# Deep Walkthrough: How this repo trains NCAs, simulates aging, and performs rejuvenation

This guide is written for exactly the questions:
- “How does the network work?”
- “How do they simulate aging?”
- “How do they tweak parameters?”
- “How do they push cells back toward embryonic state?”
- “What does each training/testing function do?”

---

## 1) Big picture (what is being modeled)

This project models a **tissue** as a grid of cells. Each cell runs the same neural controller (an NCA cell policy), repeatedly:

1. Look at local neighbors,
2. Decide a state update,
3. Apply update (with possible noise),
4. Repeat across many steps.

The goal is to self-assemble and maintain a target pattern (for example a smiley face).

The full stack is:
- `musepy/` = core framework (agent/env/task/experiment tools),
- `examples/evolve/` = training + evaluation by evolution,
- `examples/aging-as-loss-of-goal-directedness/` = post-training aging + recovery experiments.

---

## 2) The “network” architecture used by each cell

The main agent type is `HybridNCAAgent`.

Conceptually it has three modules:

1. **state_module** (structural genome)
   - Stores/produces initial cell state values.
2. **sensory_module**
   - Processes neighborhood observations into embeddings.
3. **policy_module**
   - Produces the action / delta-state update.

You can see this explicitly in a trained smiley model:
- `sensory_module` is feed-forward,
- `policy_module` is recurrent (`RGRN`),
- `sampling` controls decision probability (competency).

---

## 3) One full simulation step (exact mechanics)

### Environment side (`musepy/envs/hybrid_nca.py`)
1. `get_cell_neigh_states()` builds each cell’s neighborhood tensor.
2. Agent returns one action vector per cell.
3. `state.update(action, reset=...)` applies action to state array.
4. State noise is added inside `State.update(...)` via `noise_level`.
5. Task reward is computed (`StateTask.evaluate`).
6. If `incremental_reward=True`, reward is converted to per-step increment.

### Agent side (`musepy/agents/hybrid_neural_cellular_automaton_agent.py`)
1. `preprocess_observation(...)` converts obs and retrieves `sensitivity`.
2. `apply_sensor_sensitivity(...)` can close/open neighbor inputs (communication loss model).
3. `forward_sensory_module(...)` computes embeddings.
4. `forward_aggregate(...)` combines neighbor embeddings (`mean` or `flatten`).
5. Policy module outputs action.

So aging interventions can affect either environment dynamics (noise), agent execution reliability (sampling), agent weights (parameter drift), or communication channels (sensitivity masks).

---

## 4) Training pipeline: function-by-function

Entry point is `examples/evolve/main.py`.

## `train(...)`
- Resolves task/agent paths with `get_files(...)`.
- Parses optional checkpoint resume info.
- Calls `musepy.experiment.train(...)`.

## `musepy.experiment.train(...)` does:
1. Load tasks dataset JSON folder.
2. Load config yaml (`configs/train.yml` by default).
3. Build `HybridNCA` env from config sections:
   - `grid_config`, `state_config`, `task_config`, `render_config`.
4. Build agent from YAML template, resolving wildcard values like `{NUM_CELLS}`.
5. Dump generated `env.yml`, `agent.yml`, `world.yml` into run folder.
6. Launch ES optimization via `mindcraft.script.train(...)`.

Important: optimization is **evolutionary strategy** (CMAES/SimpleGA), not SGD backprop through episodes.

---

## 5) Testing and analysis pipeline: function-by-function

## `examples/evolve/main.py::test(...)`
- Loads world + agent and optional checkpoint.
- Calls `musepy.experiment.test(...)`.

## `musepy.experiment.test(...)`
1. Build `World` from saved world config.
2. Optionally load checkpoint parameters into the agent.
3. Set task mode/task index/schedule.
4. Run `world.rollout()` for `n_episodes`.
5. Return log history (per episode, per timestep fields).

## `progress(...)`
- Reads ES log and plots best/mean/std curves.

## `checkpoints(...)`
- Loads checkpoint history and evaluates multiple generations.
- Computes:
  - “genotypic/structural” fitness variants,
  - state trajectories across development,
  - optional progress/state plots.

---

## 6) Reward function and objective (how success is measured)

Reward is computed in `StateTask.evaluate(...)`:

1. Convert current NCA state to target-comparable representation (`hardmax`, `softmax`, or `mse` style).
2. Compute mismatch via `cross_entropy(...)` variant.
3. Translate mismatch into per-cell reward/cost using:
   - `individual_cost`,
   - `individual_reward`.
4. Add `completion_reward` if entire target matches exactly.
5. Optionally apply `stagnation_cost` when no progress occurs.

Therefore the evolved parameters are selected for better morphogenesis return over rollout episodes.

---

## 7) Aging simulation: exactly what is changed

Aging experiments are implemented in `examples/aging-as-loss-of-goal-directedness/aging.py::rollout_lifetime(...)`.

It builds a `schedule` mapping callback functions to per-step values. Four interventions:

1. `set_noise_level(...)`
   - Changes `world.env.state.noise_level` over time.
2. `set_competency_level(...)`
   - Changes `world.agent.sampling` over time (decision probability).
3. `update_agent_parameters(...)`
   - Adds Gaussian drift to current ANN parameters each step.
4. `update_agent_sensitivity(...)`
   - Changes `world.agent.sensitivity` + reduction/susceptibility knobs controlling intercellular communication loss.

This is the core answer to “how did they simulate aging?”

They do **post-training degradation during lifetime** rather than retraining a new aged model.

---

## 8) Rejuvenation/recovery: how they move cells toward embryonic state

Recovery is in `examples/aging-as-loss-of-goal-directedness/recovery.py`.

Most important function is `set_initial_state(...)`:

1. Measure organ state quality from masks (left eye/right eye/mouth).
2. If quality drops below threshold and cooldown window allows:
   - choose injection mask (organ-only or organ+socket),
   - copy values from `initial_state` into current tissue state,
   - optionally reset recurrent memory for those cells.

So rejuvenation here is **targeted state injection** and optional memory reset — not full retraining.

`run_recovery(...)` wires three organ-specific interventions into a schedule that checks/intervenes every step over episodes.

---

## 9) Where to tune parameters (practical knobs)

### Training knobs (`examples/evolve/configs/train.yml`)
- Evolution: `solver_config.es`, `size`, `generations`, `sigma_init`.
- Rollout horizon: `world_config.max_steps`, `world_config.n_episodes`.
- Environment stochasticity: `state_config.noise_level`.
- Objective shape: `task_config.individual_cost`, `stagnation_cost`, `completion_reward`.
- Competency baseline: `agent_config.sampling`, `sampling_scale`.

### Aging knobs (`aging.py` schedules)
- `noise_schedule`,
- `competency_schedule`,
- `ann_schedule`,
- `sensitivity_schedule`,
- `sensitivity_reduction_rate`, `sensitivity_susceptibility`.

### Recovery knobs (`recovery.py`)
- `threshold`, `development`, `horizon`,
- `inject_initial` (embryonic source state),
- `reset_memory`,
- `only_correct_wrongs`.

---

## 10) Direct concise answers to your original questions

- **How did they simulate aging?**
  By progressively degrading an evolved controller during long lifetime rollouts via schedules that increase noise, reduce competency, drift ANN parameters, and reduce communication sensitivity.

- **How did they tweak parameters?**
  Through step-wise callback schedules that mutate runtime world/agent fields at each timestep.

- **How did they revert cells toward embryonic state?**
  By local state overwrites (injecting initial/embryonic-like values into damaged regions) plus optional reset of recurrent hidden memory in those same cells.

- **How to understand training/testing functions?**
  Start at `examples/evolve/main.py` wrappers, then read `musepy/experiment.py` (`train`, `test`, `progress`, `checkpoints`), then the environment/task internals (`hybrid_nca.py`, `task.py`).

---

## 11) How to improve long-term health **without changing the 25-step development horizon**

If you want to keep `world_config.max_steps: 25` during evolution (development phase fixed), you can still add
"healthy-lifespan" pressure by changing the *evaluation protocol* or adding post-development interventions.

### A) Two-stage evaluation with fixed development window

Keep training development exactly as-is (`max_steps=25`), but score each candidate with:

1. **Stage 1 (development):** 25 steps, current objective (reach pattern fast).
2. **Stage 2 (maintenance):** continue rollout under perturbations/noise and score pattern retention.

This preserves developmental timing while selecting for post-development robustness.

### B) Add maintenance-only objective term (post-step-25)

Use a weighted fitness:

- `fitness = w_dev * score(steps 1..25) + w_maint * score(steps 26..T)`

with `w_dev` high enough to preserve normal development and `w_maint` high enough to prevent fade.

### C) Triggered intervention policy (closed-loop therapy)

Instead of always degrading until collapse, add intervention rules after development:

- monitor morphology quality,
- if quality drops below threshold, trigger a mild correction (state injection / memory reset / temporary noise reduction),
- enforce cooldown so interventions are sparse.

This mimics health maintenance medicine without changing early growth dynamics.

### D) Keep development fixed, train recovery competence

During training/evaluation, occasionally perturb only after step 25 and reward fast return to target.
That makes the controller resilient in adulthood while preserving early development behavior.

### E) Practical note for this repo

The aging/recovery scripts already support post-development interventions via `development` threshold in recovery logic,
so this is a natural place to implement lifespan-extending strategies without editing the 25-step developmental config.

### F) "Dumb" C variant implemented: memory-reset only when target is reached

A simple intervention variant is now available in `rollout_lifetime(...)`:
- `memory_reset_on_completion=True`
- `memory_reset_cooldown=<int>`
- `memory_reset_match_threshold=<float in [0,1]>`

Behavior:
- At each step, compute the fraction of grid cells whose type matches target,
- if this fraction is >= `memory_reset_match_threshold`, reset policy-module recurrent memory,
- do nothing otherwise,
- enforce optional cooldown between resets.

This is intentionally minimal and lets you test C-style intervention separately from A-style training/evaluation changes.

