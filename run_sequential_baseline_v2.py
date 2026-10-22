
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("outputs")
OUT = BASE / "sequential_baseline_v2"
OUT.mkdir(parents=True, exist_ok=True)

CALIBRATION_FILE = (
    BASE / "kvasir_calibration" / "kvasir_scores_all.csv"
)
HOLDOUT_FILE = (
    BASE / "kvasir_calibration" / "kvasir_scores_all.csv"
)
CVC_FILE = BASE / "cvc_evaluation" / "cvc_scores_all.csv"
ETIS_FILE = BASE / "etis_evaluation" / "etis_full196.csv"

SCORES = [
    "mean_entropy_nats",
    "p90_pixel_entropy_nats",
    "low_confidence_fraction",
    "pred_foreground_fraction",
]

WINDOW = 10
MIN_CROSSINGS = 3

# High values are suspicious for these three signals.
HIGH_IS_SUSPICIOUS = {
    "mean_entropy_nats": True,
    "p90_pixel_entropy_nats": True,
    "low_confidence_fraction": True,
    "pred_foreground_fraction": False,
}

calibration = pd.read_csv(CALIBRATION_FILE)
calibration = calibration[
    calibration["split"].eq("calibration")
]

if len(calibration) == 0:
    raise ValueError("No calibration rows found.")

# Freeze thresholds using only the source calibration split.
thresholds = {}

for score in SCORES:
    q = 0.95 if HIGH_IS_SUSPICIOUS[score] else 0.05
    thresholds[score] = float(
        calibration[score].quantile(q)
    )

print("Frozen Kvasir calibration thresholds:")
for score, threshold in thresholds.items():
    print(f"  {score}: {threshold:.8f}")


def add_alarms(df, groups=None):
    """Compute threshold crossings and rolling alarms.
    Each group gets an independent rolling window.
    """
    result = df.copy()

    for score in SCORES:
        threshold = thresholds[score]

        if HIGH_IS_SUSPICIOUS[score]:
            crossing = result[score] > threshold
        else:
            crossing = result[score] < threshold

        crossing = crossing.fillna(False).astype(int)

        result[f"{score}_crossing"] = crossing
        result[f"{score}_alarm"] = False

        if groups is None:
            alarm = (
                crossing.rolling(
                    WINDOW, min_periods=1
                ).sum() >= MIN_CROSSINGS
            )
            result[f"{score}_alarm"] = alarm.to_numpy()
        else:
            # Reset the rolling window at every sequence boundary.
            for _, indices in result.groupby(
                groups, sort=False
            ).groups.items():
                local_crossing = crossing.loc[indices]
                alarm = (
                    local_crossing.rolling(
                        WINDOW, min_periods=1
                    ).sum() >= MIN_CROSSINGS
                )
                result.loc[indices, f"{score}_alarm"] = (
                    alarm.to_numpy()
                )

    return result


# Kvasir held-out images: exploratory ordered sequence only.
kvasir = pd.read_csv(HOLDOUT_FILE)
kvasir = kvasir[kvasir["split"].eq("source_holdout")].copy()
kvasir = kvasir.sort_values("image_id").reset_index(drop=True)
kvasir = add_alarms(kvasir)
kvasir.to_csv(OUT / "kvasir_holdout_alarms.csv", index=False)

# CVC: sort frames within sequence and reset rolling windows.
cvc = pd.read_csv(CVC_FILE)
cvc["_image_sort"] = pd.to_numeric(
    cvc["image_id"], errors="coerce"
)
cvc = cvc.sort_values(
    ["sequence_id", "_image_sort"]
).reset_index(drop=True)

cvc = add_alarms(cvc, groups="sequence_id")
cvc = cvc.drop(columns=["_image_sort"])
cvc.to_csv(OUT / "cvc_alarms.csv", index=False)

# ETIS image IDs provide a reproducible ordering, not verified
# real-world deployment chronology.
etis = pd.read_csv(ETIS_FILE)
etis["_image_sort"] = pd.to_numeric(
    etis["image_id"], errors="coerce"
)
etis = etis.sort_values("_image_sort").reset_index(drop=True)
etis = add_alarms(etis)
etis = etis.drop(columns=["_image_sort"])
etis.to_csv(OUT / "etis_alarms.csv", index=False)

print("\nAlarm counts:")
for name, df in [
    ("Kvasir held-out", kvasir),
    ("CVC", cvc),
    ("ETIS", etis),
]:
    print(f"\n{name}: {len(df)} images")

    for score in SCORES:
        crossings = int(df[f"{score}_crossing"].sum())
        alarmed = int(df[f"{score}_alarm"].sum())
        print(
            f"  {score}: crossings={crossings}, "
            f"alarmed_images={alarmed}"
        )

print(f"\nSaved corrected outputs to: {OUT}")
