# Bioelectric Control Policy Toy Model

A first, deliberately small implementation of the idea that cellular rejuvenation can be framed as a **control-policy search over an upstream physiological layer** rather than as delivery of a genetic payload.

This is not a real biophysical simulator and it is not evidence for rejuvenation. It is a toy model that makes the hypothesis executable:

> Find a timed perturbation that moves aged cells through endogenous “logic gates” into a younger, identity-preserving state.

## What is modeled

Each cell in a 2D tissue grid has normalized state variables:

- `vmem`: membrane-potential-like polarity, where `1` is polarized and `0` is depolarized.
- `calcium`: a downstream messenger proxy activated by depolarization.
- `gap`: gap-junction-like neighbor coupling.
- `youth`: a rejuvenation / repair proxy.
- `identity`: cell-identity retention proxy.
- `morphology`: discrete tissue/cell-type state compared with a target morphology.
- `depol_memory`: short-term memory of prior depolarization.

The core “natural logic gate” is:

```text
short depolarization -> plasticity gate opens
then timed repolarization -> repair gate fires
only if identity is still high -> youth and morphology improve
prolonged plasticity / low identity -> stress and identity loss
```

This maps the current repo’s NCA idea onto a more bioelectric vocabulary:

```text
neighbor sensing / NCA policy / tissue state
        becomes
Vmem / coupling / calcium-like messenger / identity + youth state
```

## What the search does

The search sweeps simple depolarization→repolarization policies:

```text
start time
x depolarization duration
x repolarization duration
x depolarization strength
x repolarization strength
```

It scores policies by:

```text
youth + identity retention + morphology match - identity-loss penalty
```

So a policy that makes cells “younger” by destroying identity should not win.

## Run

From the repository root:

```bash
python examples/bioelectric-control-policy/bioelectric_control_policy.py --search --plot
```

Run one demo policy:

```bash
python examples/bioelectric-control-policy/bioelectric_control_policy.py --policy demo --plot
```

Run the aged baseline without intervention:

```bash
python examples/bioelectric-control-policy/bioelectric_control_policy.py --policy none --plot
```

The plot is saved by default to:

```text
examples/bioelectric-control-policy/data/rollout.png
```

## Why this belongs here

The existing Artificial-Aging repo already models aging as loss of goal-directed tissue control in a Neural Cellular Automaton. This example adds a first bridge toward the bioelectric-control-policy idea:

- aging = drift toward depolarization, lower coupling, lower youth, weaker identity;
- intervention = timed physiological control signal;
- rejuvenation = restoration of youth/morphology under identity constraints;
- failure mode = over-plasticity, stress, or identity loss.

## Next steps

Useful upgrades:

1. Replace the hand-written toy dynamics with equations closer to Vmem/channel/coupling models.
2. Replace grid search with Bayesian optimization or evolutionary search.
3. Add spatial policies, not only global pulses.
4. Couple the toy to real NCA target recovery experiments.
5. Replace `youth` with transcriptomic/epigenetic proxy vectors once real data exists.
