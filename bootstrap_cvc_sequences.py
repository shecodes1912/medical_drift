
from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path("outputs/cvc_evaluation/cvc_scores_all.csv")
OUT_DIR = Path("outputs/cvc_evaluation")

N_BOOTSTRAPS = 5000
SEED = 42
FAILURE_THRESHOLD = 0.5

df = pd.read_csv(CSV_PATH)
df["failure"] = df["dice"] < FAILURE_THRESHOLD

# Each sequence is one resampling unit.
groups = [
    group.copy()
    for _, group in df.groupby("sequence_id", sort=False, dropna=False)
]

rng = np.random.default_rng(SEED)

observed_frame_dice = df["dice"].mean()
observed_frame_failure = df["failure"].mean()
observed_sequence_dice = np.mean(
    [g["dice"].mean() for g in groups]
)
observed_sequence_failure = np.mean(
    [g["failure"].mean() for g in groups]
)

frame_dice_boot = []
frame_failure_boot = []
sequence_dice_boot = []
sequence_failure_boot = []

n_groups = len(groups)

for _ in range(N_BOOTSTRAPS):
    # Draw whole sequences with replacement.
    sampled_indices = rng.integers(0, n_groups, size=n_groups)
    sampled_groups = [groups[i] for i in sampled_indices]

    # Frame-weighted metrics: longer sequences contribute more.
    sampled_dice = np.concatenate(
        [g["dice"].to_numpy() for g in sampled_groups]
    )
    sampled_failures = np.concatenate(
        [g["failure"].to_numpy(dtype=float) for g in sampled_groups]
    )

    frame_dice_boot.append(sampled_dice.mean())
    frame_failure_boot.append(sampled_failures.mean())

    # Sequence-weighted metrics: each sampled sequence contributes equally.
    sequence_dice_boot.append(
        np.mean([g["dice"].mean() for g in sampled_groups])
    )
    sequence_failure_boot.append(
        np.mean([g["failure"].mean() for g in sampled_groups])
    )


def report(name, observed, samples):
    low, high = np.percentile(samples, [2.5, 97.5])
    print(f"{name}: {observed:.4f} "
          f"(sequence-bootstrap 95% CI {low:.4f}–{high:.4f})")


print("=== CVC sequence-cluster bootstrap ===")
print(f"Frames: {len(df)}")
print(f"Sequences: {n_groups}")
print(f"Bootstrap replicates: {N_BOOTSTRAPS}")
print(f"Seed: {SEED}")

print("\nFrame-weighted metrics:")
report("Mean Dice", observed_frame_dice, frame_dice_boot)
report("Failure rate (Dice < 0.5)",
       observed_frame_failure, frame_failure_boot)

print("\nSequence-weighted metrics:")
report("Mean of sequence-level mean Dice",
       observed_sequence_dice, sequence_dice_boot)
report("Mean sequence-level failure fraction",
       observed_sequence_failure, sequence_failure_boot)

print(
    "\nInterpretation: intervals account for within-sequence clustering "
    "by resampling sequences. They do not account for dataset-selection "
    "bias or establish performance on new hospitals or devices."
)
