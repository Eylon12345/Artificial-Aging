from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aging import rollout_lifetime, save_data


MAX_STEPS = 1000
NUM_EPISODES = 5
SEED = 123
NOISE_END = 0.30
OUT_DIR = Path(__file__).resolve().parent / "data"
WORLD_CONFIG = "agents/smiley_16x16/world.yml"


def save_summary_plot(data: dict, png_path: Path) -> None:
    reward = data["reward"]
    cumulative = data["cumulative_reward"]
    types = data["type"]

    x_r = np.arange(reward.shape[1])
    x_c = np.arange(cumulative.shape[1])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    axes[0, 0].plot(x_r, reward.mean(axis=0), color="black", label="mean reward")
    axes[0, 0].fill_between(
        x_r,
        reward.mean(axis=0) - reward.std(axis=0),
        reward.mean(axis=0) + reward.std(axis=0),
        color="black",
        alpha=0.2,
        label="±1 std",
    )
    axes[0, 0].set_title("Per-step reward")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[1, 0].plot(x_c, cumulative.mean(axis=0), color="tab:blue", label="mean cumulative")
    axes[1, 0].fill_between(
        x_c,
        cumulative.mean(axis=0) - cumulative.std(axis=0),
        cumulative.mean(axis=0) + cumulative.std(axis=0),
        color="tab:blue",
        alpha=0.2,
        label="±1 std",
    )
    axes[1, 0].set_title("Cumulative reward")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Cumulative reward")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    snapshot_steps = [0, min(MAX_STEPS // 2, types.shape[1] - 1), min(MAX_STEPS, types.shape[1] - 1)]
    for col, step in enumerate(snapshot_steps, start=1):
        img = types[0, step]
        axes[0, col].imshow(img, cmap="viridis", vmin=0, vmax=2)
        axes[0, col].set_title(f"Episode 1 grid @ step {step}")
        axes[0, col].axis("off")

        img_last = types[-1, step]
        axes[1, col].imshow(img_last, cmap="viridis", vmin=0, vmax=2)
        axes[1, col].set_title(f"Episode {types.shape[0]} grid @ step {step}")
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    noise_schedule = np.linspace(0.0, NOISE_END, MAX_STEPS)

    print("Running BASELINE (no memory reset)...")
    print(f" - max_steps={MAX_STEPS}, episodes={NUM_EPISODES}, seed={SEED}, noise_end={NOISE_END}")

    data = rollout_lifetime(
        world_config=WORLD_CONFIG,
        max_steps=MAX_STEPS,
        num_episodes=NUM_EPISODES,
        seed=SEED,
        noise_schedule=noise_schedule,
        memory_reset_on_completion=False,
        log_fields=("reward", "state"),
        log_foos={"state": "observation[:, 0].reshape(16, 16, -1)"},
        verbose=True,
        render=True,
    )

    h5_path = OUT_DIR / f"baseline-1000-noreset-{ts}.h5"
    png_path = OUT_DIR / f"baseline-1000-noreset-{ts}.png"
    summary_path = OUT_DIR / f"baseline-1000-noreset-summary-{ts}.json"

    save_data(data, str(h5_path))
    save_summary_plot(data, png_path)

    summary = {
        "config": {
            "max_steps": MAX_STEPS,
            "num_episodes": NUM_EPISODES,
            "seed": SEED,
            "noise_end": NOISE_END,
            "memory_reset_on_completion": False,
        },
        "results": {
            "final_cumulative_mean": float(data["cumulative_reward"][:, -1].mean()),
            "final_cumulative_std": float(data["cumulative_reward"][:, -1].std()),
            "memory_reset_count": int(data["memory_reset_count"]),
        },
        "files": {"h5": str(h5_path), "plot": str(png_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print("\nSaved artifacts:")
    print(f" - {h5_path}")
    print(f" - {png_path}")
    print(f" - {summary_path}")


if __name__ == "__main__":
    main()
