
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("outputs")
OUT_DIR = BASE / "sequential_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAL_PATH = BASE / "kvasir_calibration" / "kvasir_scores_all.csv"
CVC_PATH = BASE / "cvc_evaluation" / "cvc_scores_all.csv"
ETIS_PATH = BASE / "etis_evaluation" / "etis_full196.csv"

SEED = 42

# Candidate label-free monitoring scores.
SCORES = [
    "mean_entropy_nats",
    "p90_pixel_entropy_nats",
    "low_confidence_fraction",
    "pred_foreground_fraction",
]

# We monitor deviations in both directions for foreground fraction.
# For uncertainty scores, this first baseline only checks unusually high values.
HIGHER_IS_SUSPICIOUS = {
    "mean_entropy_nats": True,
    "p90_pixel_entropy_nats": True,
    "low_confidence_fraction": True,
    "pred_foreground_fraction": False,
}

# Empirical reference quantile used to define a threshold.
# This is a baseline choice, not a guarantee of a particular false-alarm rate.
REFERENCE_QUANTILE = 0.95

# Alarm if at least this many of the last W scores cross the threshold.
WINDOW_SIZE = 10
MIN_CROSSES = 3


def load_data():
    kvasir = pd.read_csv(CAL_PATH)
    calibration = kvasir.loc[kvasir["split"] == "calibration"].copy()
    holdout = kvasir.loc[kvasir["split"] == "source_holdout"].copy()

    cvc = pd.read_csv(CVC_PATH).copy()
    etis = pd.read_csv(ETIS_PATH).copy()

    # Preserve sequence order for CVC. Sort within each sequence by image ID.
    cvc["image_id_numeric"] = pd.to_numeric(cvc["image_id"], errors="coerce")
    cvc = cvc.sort_values(["sequence_id", "image_id_numeric"]).reset_index(drop=True)

    # ETIS image IDs are numeric; retain their natural order.
    etis["image_id_numeric"] = pd.to_numeric(etis["image_id"], errors="coerce")
    etis = etis.sort_values("image_id_numeric").reset_index(drop=True)

    return calibration, holdout, cvc, etis


def get_threshold(reference, score):
    values = reference[score].dropna().to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError(f"No reference values for {score}")

    higher = HIGHER_IS_SUSPICIOUS[score]

    if higher:
        return float(np.quantile(values, REFERENCE_QUANTILE))
    return float(np.quantile(values, 1 - REFERENCE_QUANTILE))


def calculate_crossings(df, score, threshold):
    values = df[score].to_numpy(dtype=float)
    higher = HIGHER_IS_SUSPICIOUS[score]

    if higher:
        crossed = values > threshold
    else:
        crossed = values < threshold

    return crossed


def run_stream(name, df, thresholds):
    print(f"\n=== {name} ===")
    print(f"Images: {len(df)}")

    result = df.copy()

    for score in SCORES:
        threshold = thresholds[score]
        crossed = calculate_crossings(result, score, threshold)

        rolling_count = (
            pd.Series(crossed.astype(int))
            .rolling(WINDOW_SIZE, min_periods=WINDOW_SIZE)
            .sum()
            .to_numpy()
        )

        alarms = rolling_count >= MIN_CROSSES
        alarm_positions = np.flatnonzero(alarms)

        first_alarm = (
            int(alarm_positions[0] + 1)
            if len(alarm_positions) else None
        )

        result[f"{score}_crossed"] = crossed
        result[f"{score}_rolling_crosses"] = rolling_count
        result[f"{score}_alarm"] = alarms

        print(
            f"{score}: threshold={threshold:.6g}, "
            f"crossings={int(crossed.sum())}/{len(crossed)}, "
            f"alarm_windows={int(alarms.sum())}, "
            f"first_alarm_position={first_alarm}"
        )

    return result


def main():
    calibration, holdout, cvc, etis = load_data()

    print("=== Dataset sizes ===")
    print(f"Kvasir calibration: {len(calibration)}")
    print(f"Kvasir held-out: {len(holdout)}")
    print(f"CVC frames: {len(cvc)}")
    print(f"CVC sequences: {cvc['sequence_id'].nunique()}")
    print(f"ETIS images: {len(etis)}")

    thresholds = {
        score: get_threshold(calibration, score)
        for score in SCORES
    }

    print("\n=== Frozen thresholds from Kvasir calibration ===")
    for score, threshold in thresholds.items():
        print(f"{score}: {threshold:.8g}")

    streams = {
        "kvasir_holdout": holdout,
        "cvc": cvc,
        "etis": etis,
    }

    for name, df in streams.items():
        result = run_stream(name, df, thresholds)
        out_path = OUT_DIR / f"{name}_alarms.csv"
        result.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

    print("\n=== Important interpretation limits ===")
    print(
        "This is an exploratory threshold-crossing baseline. "
        "Overlapping rolling windows are dependent."
    )
    print(
        "The reference quantile does not guarantee a sequential false-alarm "
        "rate, and results depend on stream ordering."
    )
    print(
        "For CVC, alarms should later be assessed with sequence-aware "
        "analysis. Ground-truth Dice is not used by the alarm rule."
    )


if __name__ == "__main__":
    main()
