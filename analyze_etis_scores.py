from pathlib import Path
import csv
import numpy as np

CSV_PATH = Path("outputs/etis_evaluation/etis_sample50.csv")

with CSV_PATH.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise RuntimeError("No evaluation rows found.")

def column(name):
    return np.array([float(r[name]) for r in rows], dtype=float)

def ranks(x):
    order = np.argsort(x, kind="mergesort")
    result = np.empty(len(x), dtype=float)
    i = 0

    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        result[order[i:j]] = (i + j - 1) / 2 + 1
        i = j

    return result

def corr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def auc_rank(score, failure):
    # AUROC via ranks; higher scores are assumed to indicate failure.
    # Ties receive average ranks.
    positive = int(failure.sum())
    negative = len(failure) - positive

    if positive == 0 or negative == 0:
        return float("nan")

    r = ranks(score)
    rank_sum = r[failure].sum()
    return float(
        (rank_sum - positive * (positive + 1) / 2)
        / (positive * negative)
    )

dice = column("dice")
failure = dice < 0.5

signals = {
    "Mean entropy": column("mean_entropy_nats"),
    "P90 pixel entropy": column("p90_pixel_entropy_nats"),
    "Low-confidence fraction": column("low_confidence_fraction"),
    "Connected-component count": column("connected_component_count"),
    "Empty prediction indicator": np.array(
    [r["empty_prediction"].strip().lower() == "true" for r in rows],
    dtype=float
),
}

print("=== ETIS monitoring-signal analysis ===")
print(f"Images: {len(rows)}")
print(f"Failures (Dice < 0.5): {failure.sum()}/{len(rows)}")
print(f"Successful cases: {(~failure).sum()}/{len(rows)}")

print("\n--- Association with Dice ---")
for name, score in signals.items():
    print(
        f"{name:30s} "
        f"Pearson={corr(score, dice): .4f}  "
        f"Spearman={corr(ranks(score), ranks(dice)): .4f}"
    )

print("\n--- Failure ranking (higher score = more risk) ---")
for name, score in signals.items():
    print(f"{name:30s} AUROC={auc_rank(score, failure):.4f}")

# Foreground fraction is retained as a descriptive signal.
# Its relationship with failure is not necessarily monotonic.
fg = column("pred_foreground_fraction")
print("\n--- Foreground fraction ---")
print(f"Pearson vs Dice:  {corr(fg, dice):.4f}")
print(f"Spearman vs Dice: {corr(ranks(fg), ranks(dice)):.4f}")

print("\n--- Mean scores by outcome ---")
for name, score in signals.items():
    print(
        f"{name:30s} "
        f"failure={score[failure].mean():.6f}  "
        f"success={score[~failure].mean():.6f}"
    )

empty = np.array(
    [r["empty_prediction"].strip().lower() == "true" for r in rows],
    dtype=bool
)
print("\n--- Empty-prediction diagnostic ---")
print(f"Empty predictions: {empty.sum()}/{len(rows)}")
print(f"Empty predictions among failures: {empty[failure].sum()}/{failure.sum()}")
print(
    "Failure recall from empty flag: "
    f"{empty[failure].sum() / failure.sum():.1%}"
    if failure.sum() else "Failure recall unavailable"
)