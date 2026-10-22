
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("outputs")
FAILURE_THRESHOLD = 0.5

FILES = {
    "Kvasir held-out": BASE / "sequential_baseline_v2" / "kvasir_holdout_alarms.csv",
    "CVC": BASE / "sequential_baseline_v2" / "cvc_alarms.csv",
    "ETIS": BASE / "sequential_baseline_v2" / "etis_alarms.csv",
}

SCORES = [
    "mean_entropy_nats",
    "p90_pixel_entropy_nats",
    "low_confidence_fraction",
    "pred_foreground_fraction",
]


def main():
    for dataset, path in FILES.items():
        df = pd.read_csv(path)
        df["failure"] = df["dice"] < FAILURE_THRESHOLD

        print(f"\n=== {dataset} ===")
        print(f"Images: {len(df)}")
        print(f"Failures (Dice < {FAILURE_THRESHOLD}): "
              f"{int(df['failure'].sum())}")

        for score in SCORES:
            alarm_col = f"{score}_alarm"
            if alarm_col not in df.columns:
                continue

            alarm = df[alarm_col].fillna(False).astype(bool).to_numpy()
            failure = df["failure"].to_numpy(dtype=bool)

            tp = int(np.sum(alarm & failure))
            fp = int(np.sum(alarm & ~failure))
            fn = int(np.sum(~alarm & failure))
            tn = int(np.sum(~alarm & ~failure))

            precision = tp / (tp + fp) if tp + fp else np.nan
            recall = tp / (tp + fn) if tp + fn else np.nan
            specificity = tn / (tn + fp) if tn + fp else np.nan

            print(
                f"{score:32s} "
                f"alarm_images={int(alarm.sum()):3d} "
                f"TP={tp:3d} FP={fp:3d} FN={fn:3d} TN={tn:3d} "
                f"precision={precision:.3f} "
                f"recall={recall:.3f} "
                f"specificity={specificity:.3f}"
            )

        if "sequence_id" in df.columns:
            per_sequence = (
                df.groupby("sequence_id")
                .agg(
                    frames=("dice", "size"),
                    failures=("failure", "sum"),
                )
            )
            print(
                f"Sequences: {len(per_sequence)}; "
                f"sequences with failures: "
                f"{int((per_sequence['failures'] > 0).sum())}"
            )

    print("\nCAVEATS:")
    print("- These are image-level associations, not event-level detection delay.")
    print("- Rolling alarm windows overlap, so alarmed images are dependent.")
    print("- CVC results must be recomputed with windows reset per sequence.")
    print("- ETIS image ordering is not assumed to be real deployment chronology.")
    print("- A low precision can reflect a useful early warning, but must be "
          "evaluated against an explicit false-alarm budget.")


if __name__ == "__main__":
    main()
