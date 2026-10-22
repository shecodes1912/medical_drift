
from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path("outputs/cvc_evaluation/cvc_scores_all.csv")
OUT_DIR = Path("outputs/cvc_evaluation")
FAILURE_THRESHOLD = 0.5

df = pd.read_csv(CSV_PATH)

required = {"sequence_id", "dice", "empty_prediction"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df["failure"] = df["dice"] < FAILURE_THRESHOLD

# One row per sequence, including its number of frames and failures.
summary = (
    df.groupby("sequence_id", dropna=False)
    .agg(
        frames=("dice", "size"),
        mean_dice=("dice", "mean"),
        median_dice=("dice", "median"),
        failures=("failure", "sum"),
        empty_predictions=("empty_prediction", "sum"),
    )
    .reset_index()
    .sort_values(["failures", "frames"], ascending=[False, False])
)

print("=== CVC sequence structure ===")
print(f"Total frames: {len(df)}")
print(f"Unique sequences: {df['sequence_id'].nunique(dropna=False)}")
print(f"Frames per sequence: median={summary['frames'].median():.1f}, "
      f"min={summary['frames'].min()}, max={summary['frames'].max()}")
print(f"Total Dice < {FAILURE_THRESHOLD}: {int(df['failure'].sum())}")
print(f"Sequences containing at least one failure: "
      f"{int((summary['failures'] > 0).sum())}")

print("\n=== Sequences with failures (or the lowest mean Dice) ===")
print(summary.head(15).to_string(index=False))

# List every individual failure so we can trace its sequence.
failure_cols = [
    col for col in [
        "sequence_id", "frame_id", "image_path", "dice", "iou",
        "mean_entropy_nats", "p90_pixel_entropy_nats",
        "low_confidence_fraction", "connected_component_count",
        "pred_foreground_fraction", "empty_prediction",
    ]
    if col in df.columns
]

print("\n=== Individual failure frames ===")
failures = df.loc[df["failure"]].sort_values("dice")
if failures.empty:
    print("No failures at the selected threshold.")
else:
    print(failures[failure_cols].to_string(index=False))

# Descriptive frame-level vs sequence-level failure rates.
print("\n=== Descriptive rates ===")
print(f"Frame-level failure rate: {df['failure'].mean():.4f}")
print(f"Sequence-level fraction with any failure: "
      f"{(summary['failures'] > 0).mean():.4f}")
print("Note: these are descriptive, not independent-sample confidence estimates.")

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "cvc_sequence_summary.csv"
summary.to_csv(out_path, index=False)
print(f"\nSaved sequence summary: {out_path}")
