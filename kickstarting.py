"""Kickstarting RL (Schmitt et al., 2018) for our SAC + BipedalWalker setup.

Faithful to the paper's objective: the student trains on its own RL loss PLUS a
weighted distillation term -- the cross-entropy of the student under the teacher
policy, evaluated on the STUDENT's visited states:

    L = L_RL(student)  +  lambda_k * H(pi_teacher(.|s) || pi_student(.|s))

H(pi_T || pi_S) = E_{a ~ pi_T}[ -log pi_S(a|s) ]  (== KL(pi_T||pi_S) up to a
constant in the student's parameters), so minimizing it pulls the student toward
the teacher's action distribution. lambda_k is annealed to 0 so the student
weans off the teacher.

Adaptations from the paper (documented, not hidden):
  * Base algo: paper uses IMPALA (on-policy, V-trace); we add the term to SAC's
    actor loss instead.
  * lambda_k schedule: the paper tunes lambda_k with Population Based Training;
    we use a fixed linear decay (lambda0 -> 0 over `decay_steps`), which is the
    common practical substitute.
  * H is estimated with one teacher action sample per state (Monte Carlo), using
    SB3's squashed-Gaussian log_prob so it is correct under tanh squashing/gSDE.
Teacher is frozen (inference only) and excluded from checkpoints.
"""
from __future__ import annotations

import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update


class KickstartSAC(SAC):
    def set_kickstart(self, teacher: SAC, lambda0: float, decay_steps: int) -> None:
        self._ks_teacher = teacher
        self._ks_lambda0 = float(lambda0)
        self._ks_decay_steps = int(decay_steps)

    def _lambda_k(self) -> float:
        if not hasattr(self, "_ks_teacher") or self._ks_decay_steps <= 0:
            return 0.0
        frac = max(0.0, 1.0 - self.num_timesteps / self._ks_decay_steps)
        return self._ks_lambda0 * frac

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + [
            "_ks_teacher", "_ks_lambda0", "_ks_decay_steps"
        ]

    def _kickstart_loss(self, obs) -> th.Tensor:
        """H(pi_teacher || pi_student) = E_{a~pi_T}[ -log pi_S(a|s) ] on states `obs`."""
        with th.no_grad():
            teacher_actions, _ = self._ks_teacher.actor.action_log_prob(obs)
        mean_s, log_std_s, kwargs_s = self.actor.get_action_dist_params(obs)
        dist = self.actor.action_dist.proba_distribution(mean_s, log_std_s, **kwargs_s)
        log_prob_s = dist.log_prob(teacher_actions)
        return -log_prob_s.mean()

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Copied from SB3 SAC.train (2.9.0), with the kickstarting term added to
        # the actor loss (marked ==KICKSTART==).
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, kickstart_losses = [], [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

            # ==KICKSTART== add lambda_k * H(pi_teacher || pi_student)
            lam = self._lambda_k()
            if lam > 0.0:
                ks = self._kickstart_loss(replay_data.observations)
                actor_loss = actor_loss + lam * ks
                kickstart_losses.append(ks.item())

            actor_losses.append(actor_loss.item())
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("ksrl/lambda_k", self._lambda_k())
        if kickstart_losses:
            self.logger.record("ksrl/kickstart_loss", np.mean(kickstart_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
