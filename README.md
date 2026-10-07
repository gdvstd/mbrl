# Transfer-RL Study: Reproducing & Extending EBTL

A study of **when teacher guidance helps a student policy** — centered on
reproducing **EBTL** (Energy-Based Transfer for Reinforcement Learning,
[arXiv:2506.16590](https://arxiv.org/abs/2506.16590)) and its baselines
(Fine-Tuning, Action Advising, JSRL, Kickstarting), then extending the
question toward VLA-based manipulation.

The repo contains four experiment tracks, in the order they were built.
Each track's design decisions, pitfalls, and results are documented in the
module docstrings of its files.

## Track 1 — SAC / BipedalWalker (continuous-control warm-up)

Teacher `BipedalWalker-v3` → student `BipedalWalkerHardcore-v3`.

- `sac_common.py` — shared SAC config (single source of truth)
- `train_teacher.py` / `train_student.py` — training entry points
- `transfer.py` — pluggable transfer seam (`--transfer none|weight_init|jsrl|reward_shaping|ksrl`)
- `jsrl.py`, `kickstarting.py`, `reward_shaping.py` — method implementations
- `compare_runs.py`, `watch_policy.py`, `watch_latest.py`, `explore_bipedal.py` — tooling

Finding: weight-init transfer dominates; teacher-in-the-loop methods are
limited because the flat-terrain teacher is OOD on hardcore obstacles —
the covariate-shift problem EBTL targets.

## Track 2 — MiniGrid 6x6 (discrete stand-in pair)

Teacher `MiniGrid-Empty-6x6` → student `MiniGrid-DoorKey-6x6` (SB3 PPO,
7x7 egocentric partial obs).

- `grid_common.py` — env ids, CNN, PPO config (note the hard-won gotchas
  in the comments: `normalize_images=False`, `ent_coef=0.05`)
- `train_grid_teacher.py` / `train_grid_student.py`
  (`--transfer scratch|finetune|aa|jsrl|ksrl|ebtl`)
- `grid_mixed_ppo.py` — mixed teacher/student rollouts with the behavior
  log-prob stored (EBTL Eq. 3 off-policy correction via PPO's own ratio)
- `grid_strategies.py` — AA / JSRL / EBTL guidance masks
- `grid_ksrl.py` — Kickstarting (exact discrete cross-entropy)
- `grid_teacher.py` — frozen teacher + energy threshold calibration
  (energy MUST use raw `action_net` logits; torch's `Categorical.logits`
  are normalized and give logsumexp == 0)
- `grid_energy_reg.py` — teacher energy regularization (margin loss)
- `run_grid_experiments.sh`, `watch_grid.py` (GIF/live rollouts)

Finding: EBTL is the only guidance method that never hurts (3/3 seeds),
but this surrogate pair has no true shared ID region, which energy
regularization exposes by shutting the gate entirely.

## Track 3 — Paper-faithful 11x11 Four-Room (main reproduction)

The paper's actual GridWorld, reconstructed from Fig. 2a + Appendix A/B:
3x3 rooms, center doorways, full-grid 11x11 observation, sparse reward
(exactly 1), action masking, MaskablePPO with the paper's Table-3
hyperparameters and Fig-8a architecture (25.7K params).

- `fourroom_env.py` — AltGoal & Locked scenarios (+ `--ego` egocentric
  variant with occlusion)
- `paper_common.py` — wrappers, hyperparams, extractor, frozen teacher,
  tau calibration
- `paper_mixed_ppo.py` / `paper_ksrl.py` / `paper_energy_reg.py` — ports
  onto sb3-contrib MaskablePPO
- `train_paper_teacher.py` / `train_paper_student.py` (`--scenario`,
  `--transfer`, `--ego`)
- `run_paper_experiments.sh`, `run_paper_ego.sh`, `run_paper_ft_full.sh`
- `viz_energy_paper.py` — Fig-5b-style advice-rate heatmaps + state
  montages; `calib_scratch.py` — No-Transfer calibration probes

Findings (3 seeds): the paper's ordering holds —
frozen-conv Fine-Tuning plateaus < Action Advising < EBTL (fastest
takeoff, only 3/3-reliable guidance method) in both scenarios; full
(unfrozen) fine-tuning, which the paper does not test, is strongest
overall. The egocentric variant flips the regime (translation/rotation
invariance makes the baseline ~10x faster) and shows EBTL's
do-no-harm property: unfiltered advice becomes 2.6-3.5x slower than
scratch while EBTL's gate shuts and stays harmless. Observation
representation was pinned down by matching No-Transfer curves to the
paper (full obs, not egocentric).

## Track 4 — VLA / LIBERO (`vla/`)

The altgoal analog in manipulation: teacher knows 5 of 10
`libero_object` pick-and-place objects; the student's target mixes all 10.

- `vla/vla_common.py` — task split, env/preprocess, local BC policy
- `vla/train_teacher_bc.py`, `vla/eval_policy.py` — local BC stand-in
  teacher (documented negative result: naive feedforward BC fails
  closed-loop by compounding error even when teacher-forcing error is low)
- `vla/vla_ppo_common.py`, `vla/vla_mixed_ppo.py`, `vla/vla_ksrl.py`,
  `vla/train_vla_student.py` — student + baselines (scratch / finetune /
  aa / jsrl / ksrl) on a task-mix env
- `vla/finetune_openvla.py` — OpenVLA-7B LoRA teacher on the seen-5 demos
  (hdf5-direct, bypassing RLDS). Two critical recipe points, learned the
  hard way: action tokens must follow the pretrained convention
  (`token = vocab_size - bin`), and LoRA must target the LLM only —
  `all-linear` wraps the vision tower and collapses the policy to a
  constant action.
- `vla/eval_openvla.py`, `vla/probe_sensitivity.py` — gate evaluation and
  cheap frame-sensitivity probes
- `vla/cluster/` — Purdue Gilbreth (Slurm) setup, data fetch, and job
  scripts; see `vla/cluster/README_GILBRETH.md`

## Setup

Gridworld tracks: `pip install -r requirements.txt` (mujoco pins matter).
VLA track: see `vla/cluster/setup_gilbreth.sh` — it encodes every
version pin this stack needs (robosuite 1.4.1 + mujoco 2.3.7, and for
OpenVLA: transformers 4.40.1 / tokenizers 0.19.1 / timm 0.9.10 /
peft 0.11.1 / torch 2.5.x).

Experiment outputs land in `runs/` (gridworld) and `vla/runs_vla/` /
cluster scratch (VLA); both are git-ignored.
