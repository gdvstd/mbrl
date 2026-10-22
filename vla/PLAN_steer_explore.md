# Steering-Driven Exploration (SDE) — research plan

Idea (Sehyun, 2026-10-22): take PDE's question — exploration for VLA RL
fine-tuning when the initial policy has ~0% success — but change the
exploration variable from the *prompt* to the *activation (concept) space*,
using the mechanistic steering recipe of arXiv 2509.00328. An LLM decomposes
each task into skills / solution procedures; early in training we steer the
VLA toward those behaviors so it accumulates successful experience, then
anneal the intervention away.

## Source papers

1. **Mechanistic interpretability for steering VLAs** (arXiv 2509.00328)
   - FFN value vectors (columns of each layer's `down_proj`) projected onto
     the vocabulary via the LM head (logit lens); top-k tokens name the
     neuron's concept.
   - Steering = override selected neuron activations with a fixed scalar
     α ∈ {2..20} via a forward hook on `down_proj` (OpenVLA/PyTorch).
   - Validated on OpenVLA + LIBERO-Long (e.g. "fast" clusters → +27.7%
     end-effector displacement, p<0.001; max +148%) and π0-FAST on a UR5.
   - Noted limitations we exploit: clusters are token-similarity-based, not
     behaviorally validated; effect of fine-tuning on steerability unknown.
2. **PDE: Prompt-Driven Exploration** (arXiv 2607.08837, MIT, 2026-09)
   - Exploration variable = the prompt. A VLM is an implicit posterior
     ρ(·|g,H): it watches rollout videos and rewrites prompts; rollouts use
     p ~ α_i δ_{p_g} + (1−α_i) ρ with α_i annealed toward the canonical
     prompt. PPO with *mixed backpropagation*: the ratio uses the average
     log-prob under exploratory and canonical prompt.
   - Setup: LIBERO / LIBERO-PRO (120 tasks) / ManiSkill PutOnPlateInScene25;
     Pi0.5-SFT (weak ckpt), GR00T, Pi0; baselines Action Noise, Parameter
     Noise, RND, Robometer dense reward; metric `success_once` (RLinf suite,
     250 eval envs/task); tasks tiered by initial SR (hard <10%, medium,
     easy). Headline: +60% relative on hard tier; microwave task 0%→98%.
   - Key diagnostic we reuse: "can the method discover a successful rollout
     within ~85 rollouts, before any weight update?"
   - Website/code: https://xinyunsunshine.github.io/prompt-rl

## Concretized method (SDE)

- **Skill decomposition**: LLM generates, per task g, a skill sequence
  (e.g. close-microwave → [reach door, contact, push]) plus concept
  keywords per skill ("push", "door", "slow", ...).
- **Concept→neuron mapping**: logit-lens the FFN value vectors; for each
  keyword collect aligned neurons across layers. **Behavioral validation**
  filter (our addition over paper 1): short steered rollouts must move
  measurable behavior metrics (displacement / height / contact), otherwise
  the cluster is discarded.
- **Exploration**: at rollout start sample a steering config
  σ = (concept cluster(s), α, time window). Posterior over σ:
  (a) bandit (UCB/Thompson on success/progress) — cheap, no VLM in loop;
  (b) VLM-feedback variant (PDE-style) later. Phase-scheduled steering
  (skill k active during phase k) is the capability prompts don't have.
- **Transfer to learning**: steered rollouts are off-policy → store the
  *behavior* log-prob from the steered forward pass and let the PPO ratio
  correct it — a direct transplant of our gridworld MixedPolicyPPO /
  EBTL-Eq.3 machinery. Steering strength annealed like PDE's α schedule.
- **Hypotheses**: H1 steering gives global, temporally coherent
  perturbations with continuous control; H2 steering beats prompts when
  instruction-following is weak (hard tier) because it bypasses the
  language pathway; H3 phase-scheduled steering reaches behaviors neither
  prompts nor noise reach; H4 prompt+steering is complementary.

## Experiment plan

### Phase 0 — steering reproduction + concept library (≈1 wk, 1 GPU, inference only)
- `vla/steer/logit_lens.py`: project every `down_proj` column through the
  LM head; search a keyword list; save concept library JSON
  (concept → [(layer, neuron, token, rank, logit)]), dump top-30 tokens of
  selected neurons for manual inspection. Run for **base OpenVLA-7B and our
  LoRA teacher (lora_ep4)** — the base-vs-LoRA steerability comparison is
  itself a contribution (open question in paper 1).
- `vla/steer/steer_probe.py`: behavior probe on libero_object (seen 0,2 +
  unseen 5): concepts {fast, slow, up, down, stop, push} × α ∈ {5, 10} vs
  baseline vs random-neuron control, fixed init states (paired), metrics =
  mean per-step EE displacement, max EE height, net displacement, success.
- **Gate**: steering produces statistically significant, direction-correct
  behavior changes; random control ≈ baseline. Report whether LoRA
  preserves steerability.

### Phase 1 — exploration quality, no RL (1–2 wks, decisive go/no-go)
- PDE's pre-RL diagnostic on ~0%-SR tasks (our unseen 5–9; LIBERO-90
  "close the microwave" to reproduce PDE Fig. 3 left): fixed budget
  (~200 rollouts/task), measure rollouts-to-first-success and success
  discovery rate for: Action Noise | prompt exploration (PDE posterior
  stage, LLM API) | **SDE (bandit over concept×α×phase)** | SDE+prompt.
- All inference-only; 1–2 A100s.
- **Gate**: SDE clearly beats action noise and is competitive with prompt
  exploration on at least some hard tasks.

### Phase 2 — exploration → learning (2–3 wks)
- Track (a), primary: **steered-success distillation** — collect successful
  exploratory rollouts, LoRA-BC/DAgger them into the policy under the
  canonical prompt (matches our "teacher-rollout BC wins" finding and the
  bcrl lesson that PPO-on-BC is fragile at our scale). Compare final SR per
  exploration method at equal rollout budget.
- Track (b), if feasible: PDE's released code / RLinf on a few tasks, add
  steering to their mixed-backprop PPO.
- Protocol adopted from PDE: `success_once`, initial-SR tiers,
  success-vs-rollout-budget curves.

### Phase 3 — analysis / extensions
- Gate steering by our φ / SAE-recon OOD scores (steer only in familiar
  states — direct bridge to the EBTL line).
- Interpret which concepts exploration discovers; extend to LIBERO-PRO.

## Risks & mitigations
- Cluster semantic ambiguity → Phase-0 behavioral validation filter.
- LoRA may damage steerability → base-vs-LoRA comparison is a deliverable
  either way.
- Full PDE reproduction is infra-heavy → adopt their protocol, not their
  infra; check their code release before Phase 1.
- 7B PPO infeasible locally → Phase-2 distillation track is the primary
  claim vehicle.
