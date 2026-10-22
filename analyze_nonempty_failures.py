
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import rankdata

CSV_PATH = Path("outputs/etis_evaluation/etis_full196.csv")
OUT_DIR = Path("outputs/etis_evaluation")
N_BOOTSTRAPS = 5000
SEED = 42
FAILURE_THRESHOLD = 0.5

df = pd.read_csv(CSV_PATH)

required = {
    "dice",
    "empty_prediction",
    "mean_entropy_nats",
    "p90_pixel_entropy_nats",
    "low_confidence_fraction",
    "connected_component_count",
    "pred_foreground_fraction",
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")


def auroc(y, score):
    """Mann–Whitney AUROC; higher score must indicate higher failure risk."""
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=float)

    positives = y.sum()
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return np.nan

    ranks = rankdata(score)
    return (
        ranks[y].sum() - positives * (positives + 1) / 2
    ) / (positives * negatives)


# Restrict to nonempty predictions to study failures that an empty-mask
# alarm alone would miss.
nonempty = df.loc[~df["empty_prediction"].astype(bool)].copy()
y = (nonempty["dice"].to_numpy() < FAILURE_THRESHOLD)

scores = {
    "Mean entropy (higher risk)": (
        nonempty["mean_entropy_nats"].to_numpy(), 1
    ),
    "P90 entropy (higher risk)": (
        nonempty["p90_pixel_entropy_nats"].to_numpy(), 1
    ),
    "Low-confidence fraction (higher risk)": (
        nonempty["low_confidence_fraction"].to_numpy(), 1
    ),
    "Component count (higher risk)": (
        nonempty["connected_component_count"].to_numpy(), 1
    ),
    "Foreground fraction (lower risk)": (
        nonempty["pred_foreground_fraction"].to_numpy(), -1
    ),
}

rng = np.random.default_rng(SEED)
results = []

print("=== ETIS nonempty-prediction failure detection ===")
print(f"All ETIS images: {len(df)}")
print(f"Nonempty predictions: {len(nonempty)}")
print(f"Failures (Dice < {FAILURE_THRESHOLD}): {int(y.sum())}")
print(f"Non-failures: {int((~y).sum())}")
print(f"Bootstrap replicates: {N_BOOTSTRAPS}; seed={SEED}")

for name, (raw_score, direction) in scores.items():
    oriented_score = raw_score * direction
    observed = auroc(y, oriented_score)

    boot = []
    positive_indices = np.flatnonzero(y)
    negative_indices = np.flatnonzero(~y)

    # Stratified bootstrap: keep the observed number of failures and
    # non-failures fixed in each replicate.
    for _ in range(N_BOOTSTRAPS):
        sampled_pos = rng.choice(
            positive_indices, size=len(positive_indices), replace=True
        )
        sampled_neg = rng.choice(
            negative_indices, size=len(negative_indices), replace=True
        )
        indices = np.concatenate([sampled_pos, sampled_neg])
        boot.append(auroc(y[indices], oriented_score[indices]))

    low, high = np.percentile(boot, [2.5, 97.5])
    results.append({
        "score": name,
        "n_nonempty": len(nonempty),
        "n_failures": int(y.sum()),
        "auroc": observed,
        "ci95_low": low,
        "ci95_high": high,
    })
    print(f"{name:42s} AUROC={observed:.3f} "
          f"95% bootstrap CI=[{low:.3f}, {high:.3f}]")

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "etis_nonempty_auroc_ci.csv"
pd.DataFrame(results).to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(
    "These are image-level descriptive estimates, not independent "
    "validation of a deployment-time drift detector."
)
