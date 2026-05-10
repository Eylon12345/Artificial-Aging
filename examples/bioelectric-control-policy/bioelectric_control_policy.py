"""Toy bioelectric control-policy search.

This is not a biophysical simulator. It is a small, inspectable model that
turns the Artificial-Aging / NCA idea into a first bioelectric-control toy:
local cells have membrane-potential-like state, gap-junction-like coupling,
calcium/plasticity gates, identity confidence, and a youth marker. A control
policy is a timed depolarization -> repolarization pulse. The search asks which
policy improves youth while preserving identity and morphology.

Run:
    python bioelectric_control_policy.py --search --plot
    python bioelectric_control_policy.py --policy demo --plot
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class ModelParams:
    """Parameters for the toy dynamical system.

    All state variables are normalized to [0, 1]. For vmem, 1 means polarized
    and 0 means depolarized. This avoids pretending that the toy has real mV
    units while preserving the qualitative polarity logic.
    """

    size: int = 32
    baseline_vmem: float = 0.78
    age_depolarization_rate: float = 0.0010
    age_youth_decay_rate: float = 0.0013
    age_gap_decay_rate: float = 0.0008
    gap_diffusion_rate: float = 0.16
    voltage_relaxation_rate: float = 0.24
    calcium_relaxation_rate: float = 0.55
    calcium_from_depol: float = 1.35
    plasticity_threshold: float = 0.54
    repair_gain: float = 0.060
    identity_loss_gain: float = 0.035
    stress_loss_gain: float = 0.045
    identity_recovery_gain: float = 0.020
    morphology_repair_gain: float = 0.055
    morphology_noise_rate: float = 0.007
    min_identity_for_repair: float = 0.50
    seed: int = 7


@dataclass
class BioelectricState:
    """Grid state for the toy tissue."""

    vmem: Array
    calcium: Array
    gap: Array
    youth: Array
    identity: Array
    morphology: Array
    target: Array
    depol_memory: Array


def sigmoid(x: Array | float, k: float = 12.0) -> Array | float:
    return 1.0 / (1.0 + np.exp(-k * x))


def clip01(x: Array) -> Array:
    return np.clip(x, 0.0, 1.0)


def neighbor_mean(x: Array) -> Array:
    """Von-Neumann local neighborhood average with wraparound boundaries."""
    return (
        np.roll(x, 1, axis=0)
        + np.roll(x, -1, axis=0)
        + np.roll(x, 1, axis=1)
        + np.roll(x, -1, axis=1)
    ) / 4.0


def make_smiley_target(size: int) -> Array:
    """Return a simple 2D target morphology with background, face, and organs.

    0 = background, 1 = face/body, 2 = organ/feature.
    """
    yy, xx = np.mgrid[:size, :size]
    cx = cy = (size - 1) / 2.0
    radius = size * 0.40
    target = np.zeros((size, size), dtype=np.int8)
    face = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    target[face] = 1

    eye_r = max(1, int(size * 0.045))
    left_eye = (int(size * 0.36), int(size * 0.40))
    right_eye = (int(size * 0.64), int(size * 0.40))
    for ex, ey in (left_eye, right_eye):
        target[(xx - ex) ** 2 + (yy - ey) ** 2 <= eye_r**2] = 2

    mouth = (
        (yy > size * 0.60)
        & (yy < size * 0.72)
        & (np.abs(xx - cx) < size * 0.18)
        & (((yy - size * 0.60) / (size * 0.12)) > np.abs(xx - cx) / (size * 0.18))
    )
    target[mouth] = 2
    return target


def init_state(params: ModelParams, aged: bool = True) -> BioelectricState:
    rng = np.random.default_rng(params.seed)
    target = make_smiley_target(params.size)
    morphology = target.copy()

    if aged:
        # Small morphology defects: some organ/body cells drift to the wrong type.
        defect_mask = rng.random(target.shape) < 0.08
        morphology = np.where(defect_mask, rng.integers(0, 3, target.shape), morphology)
        vmem = clip01(params.baseline_vmem - 0.22 + 0.05 * rng.standard_normal(target.shape))
        youth = clip01(0.48 + 0.10 * rng.standard_normal(target.shape))
        identity = clip01(0.70 + 0.12 * rng.standard_normal(target.shape))
        gap = clip01(0.55 + 0.15 * rng.standard_normal(target.shape))
    else:
        vmem = clip01(params.baseline_vmem + 0.03 * rng.standard_normal(target.shape))
        youth = clip01(0.92 + 0.03 * rng.standard_normal(target.shape))
        identity = clip01(0.95 + 0.02 * rng.standard_normal(target.shape))
        gap = clip01(0.85 + 0.05 * rng.standard_normal(target.shape))

    return BioelectricState(
        vmem=vmem,
        calcium=np.zeros_like(vmem),
        gap=gap,
        youth=youth,
        identity=identity,
        morphology=morphology,
        target=target,
        depol_memory=np.zeros_like(vmem),
    )


def policy_signal(
    t: int,
    depol_start: int,
    depol_duration: int,
    repol_duration: int,
    depol_strength: float,
    repol_strength: float,
) -> float:
    """Timed control signal: negative depolarizes, positive repolarizes."""
    if depol_start <= t < depol_start + depol_duration:
        return -float(depol_strength)
    if depol_start + depol_duration <= t < depol_start + depol_duration + repol_duration:
        return float(repol_strength)
    return 0.0


def step(state: BioelectricState, params: ModelParams, control: float, rng: np.random.Generator) -> None:
    """Advance one timestep.

    The qualitative logic is:
    - aging drifts cells toward depolarization, lower youth, and lower coupling;
    - gap-like coupling pulls a cell's vmem toward its neighbors' vmem;
    - depolarization raises a calcium/plasticity gate;
    - a short depol memory followed by repolarization creates a repair gate;
    - too much plasticity without identity confidence causes identity loss/stress.
    """
    depol = 1.0 - state.vmem
    neighborhood_v = neighbor_mean(state.vmem)
    gap_flow = state.gap * (neighborhood_v - state.vmem)

    # Voltage dynamics: endogenous drift + community coupling + external action.
    dv = params.voltage_relaxation_rate * (
        params.baseline_vmem - state.vmem - params.age_depolarization_rate
    )
    dv += params.gap_diffusion_rate * gap_flow
    dv += control
    state.vmem[...] = clip01(state.vmem + dv)

    depol = 1.0 - state.vmem
    state.calcium[...] = clip01(
        (1.0 - params.calcium_relaxation_rate) * state.calcium
        + params.calcium_relaxation_rate * sigmoid(params.calcium_from_depol * (depol - 0.34), k=8)
    )

    plasticity = sigmoid(depol - params.plasticity_threshold, k=14)
    repolarizing = sigmoid(state.vmem - neighbor_mean(state.vmem), k=8) + max(control, 0.0)
    state.depol_memory[...] = clip01(0.90 * state.depol_memory + plasticity)
    repair_gate = clip01(state.depol_memory * repolarizing * state.identity)

    # Youth increases only when the timed depol->repol gate fires and identity is retained.
    youth_gain = params.repair_gain * repair_gate * (1.0 - state.youth)
    stress_loss = params.stress_loss_gain * plasticity * (1.0 - state.identity)
    age_loss = params.age_youth_decay_rate * (1.0 + 0.8 * depol)
    state.youth[...] = clip01(state.youth + youth_gain - stress_loss - age_loss)

    # Identity has a plasticity danger zone but recovers under polarized, high-youth conditions.
    identity_loss = params.identity_loss_gain * plasticity * (1.0 - state.youth)
    identity_recovery = params.identity_recovery_gain * state.vmem * state.youth * (1.0 - state.identity)
    state.identity[...] = clip01(state.identity + identity_recovery - identity_loss)

    # Gap-like coupling decays with age but partially recovers when cells are healthier.
    state.gap[...] = clip01(state.gap - params.age_gap_decay_rate + 0.004 * repair_gate)

    # Morphology repair: incorrect cells can snap back to target when repair + identity are high.
    wrong = state.morphology != state.target
    repair_probability = params.morphology_repair_gain * repair_gate * (state.identity > params.min_identity_for_repair)
    repaired = wrong & (rng.random(wrong.shape) < repair_probability)
    state.morphology[repaired] = state.target[repaired]

    # Morphology noise: weak identity + low youth lets local type drift.
    drift_probability = params.morphology_noise_rate * (1.0 - state.identity) * (1.0 - state.youth)
    drift = rng.random(wrong.shape) < drift_probability
    state.morphology[drift] = rng.integers(0, 3, size=int(drift.sum()))


def metrics(state: BioelectricState) -> Dict[str, float]:
    morphology_match = float(np.mean(state.morphology == state.target))
    youth = float(np.mean(state.youth))
    identity = float(np.mean(state.identity))
    gap = float(np.mean(state.gap))
    vmem = float(np.mean(state.vmem))
    score = youth + 0.9 * identity + 0.7 * morphology_match - 0.6 * max(0.0, 0.82 - identity)
    return {
        "score": score,
        "youth": youth,
        "identity": identity,
        "morphology_match": morphology_match,
        "gap": gap,
        "vmem": vmem,
    }


def rollout(
    params: ModelParams,
    steps: int,
    policy: Tuple[int, int, int, float, float] | None = None,
    aged: bool = True,
) -> Tuple[BioelectricState, Dict[str, List[float]], List[float]]:
    rng = np.random.default_rng(params.seed)
    state = init_state(params, aged=aged)
    history: Dict[str, List[float]] = {k: [] for k in metrics(state).keys()}
    controls: List[float] = []

    for t in range(steps):
        control = 0.0
        if policy is not None:
            control = policy_signal(t, *policy)
        controls.append(control)
        step(state, params, control, rng)
        for key, value in metrics(state).items():
            history[key].append(value)
    return state, history, controls


def search_policies(params: ModelParams, steps: int = 180) -> List[Dict[str, float]]:
    """Small grid search over depol/repol policy parameters."""
    results: List[Dict[str, float]] = []
    depol_starts = [10]
    depol_durations = [4, 8, 12, 20, 32]
    repol_durations = [4, 8, 12, 20, 32]
    depol_strengths = [0.025, 0.045, 0.070, 0.095]
    repol_strengths = [0.025, 0.045, 0.070, 0.095]

    _, baseline_history, _ = rollout(params, steps=steps, policy=None, aged=True)
    baseline = {f"baseline_{k}": v[-1] for k, v in baseline_history.items()}

    for ds in depol_starts:
        for dd in depol_durations:
            for rd in repol_durations:
                for dep_s in depol_strengths:
                    for rep_s in repol_strengths:
                        policy = (ds, dd, rd, dep_s, rep_s)
                        _, history, _ = rollout(params, steps=steps, policy=policy, aged=True)
                        final = {k: v[-1] for k, v in history.items()}
                        results.append(
                            {
                                "depol_start": ds,
                                "depol_duration": dd,
                                "repol_duration": rd,
                                "depol_strength": dep_s,
                                "repol_strength": rep_s,
                                **final,
                                **baseline,
                                "delta_score": final["score"] - baseline["baseline_score"],
                                "delta_youth": final["youth"] - baseline["baseline_youth"],
                                "delta_identity": final["identity"] - baseline["baseline_identity"],
                                "delta_morphology": final["morphology_match"] - baseline["baseline_morphology_match"],
                            }
                        )
    return sorted(results, key=lambda r: r["delta_score"], reverse=True)


def plot_rollout(history: Dict[str, List[float]], controls: List[float], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(controls))
    fig, ax = plt.subplots(figsize=(11, 6))
    for key in ("youth", "identity", "morphology_match", "gap", "vmem"):
        ax.plot(x, history[key], label=key)
    ax2 = ax.twinx()
    ax2.plot(x, controls, linestyle="--", label="control")
    ax.set_xlabel("timestep")
    ax.set_ylabel("state metrics")
    ax2.set_ylabel("control signal")
    ax.set_title("Toy bioelectric control-policy rollout")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def print_top(results: Iterable[Dict[str, float]], n: int = 10) -> None:
    keys = [
        "delta_score",
        "delta_youth",
        "delta_identity",
        "delta_morphology",
        "depol_duration",
        "repol_duration",
        "depol_strength",
        "repol_strength",
        "youth",
        "identity",
        "morphology_match",
    ]
    print("\t".join(keys))
    for row in list(results)[:n]:
        print("\t".join(f"{row[k]:.4f}" if isinstance(row[k], float) else str(row[k]) for k in keys))


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy bioelectric control-policy search")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--search", action="store_true", help="run grid search over timed depol/repol policies")
    parser.add_argument("--policy", choices=["demo", "none"], default="demo")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("examples/bioelectric-control-policy/data/rollout.png"))
    args = parser.parse_args()

    params = ModelParams()

    if args.search:
        results = search_policies(params, steps=args.steps)
        print_top(results, n=10)
        best = results[0]
        policy = (
            int(best["depol_start"]),
            int(best["depol_duration"]),
            int(best["repol_duration"]),
            float(best["depol_strength"]),
            float(best["repol_strength"]),
        )
        print("\nBest policy:", policy)
    else:
        policy = None if args.policy == "none" else (10, 12, 20, 0.070, 0.070)

    _, history, controls = rollout(params, steps=args.steps, policy=policy, aged=True)
    print("\nFinal metrics:")
    for key, values in history.items():
        print(f"{key}: {values[-1]:.4f}")

    if args.plot:
        plot_rollout(history, controls, args.out)
        print(f"\nSaved plot to {args.out}")


if __name__ == "__main__":
    main()
