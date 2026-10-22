
import csv
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, rankdata, wasserstein_distance

BASE = Path("outputs")
FILES = {
    "Kvasir calibration": BASE / "kvasir_calibration" / "kvasir_scores_all.csv",
    "Kvasir held-out": BASE / "kvasir_calibration" / "kvasir_scores_all.csv",
    "CVC": BASE / "cvc_evaluation" / "cvc_scores_all.csv",
    "ETIS": BASE / "etis_evaluation" / "etis_full196.csv",
}
OUT = BASE / "cross_dataset_analysis"
OUT.mkdir(parents=True, exist_ok=True)

SCORES = [
    "mean_entropy_nats",
    "p90_pixel_entropy_nats",
    "low_confidence_fraction",
    "connected_component_count",
    "pred_foreground_fraction",
    "mean_probability",
]

# Higher scores indicate greater risk for these three candidate signals.
RISK_DIRECTIONS = {
    "mean_entropy_nats": 1,
    "p90_pixel_entropy_nats": 1,
    "low_confidence_fraction": 1,
    "connected_component_count": 1,
    "pred_foreground_fraction": -1,  # Exploratory: unusually small masks
    "mean_probability": 1,            # Exploratory; direction not established
    "empty_prediction": 1,
}


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Check that the previous scoring runs completed."
        )
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def num(row, key):
    return float(row[key])


def auc_rank(y_true, scores):
    """Mann–Whitney AUROC with average ranks for tied scores."""
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=float)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(s, method="average")
    return float(
        (ranks[y].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def summarize_performance(name, rows):
    y = [num(r, "dice") < 0.5 for r in rows]
    n_fail = sum(y)
    print(f"\n{name}: n={len(rows)}, Dice<0.5 failures={n_fail}")

    if n_fail == 0 or n_fail == len(rows):
        print("  AUROC unavailable: both outcome classes are required.")
        return

    for score in SCORES + ["empty_prediction"]:
        values = []
        for r in rows:
            if score == "empty_prediction":
                values.append(float(r[score].lower() == "true"))
            else:
                values.append(num(r, score))

        direction = RISK_DIRECTIONS[score]
        auc = auc_rank(y, np.asarray(values) * direction)
        print(f"  {score:32s} AUROC={auc:.3f}")


def main():
    all_rows = {
        name: read_csv(path) for name, path in FILES.items()
    }

    # Split the two Kvasir subsets using the stored split column.
    kvasir_all = all_rows["Kvasir calibration"]
    calibration = [r for r in kvasir_all if r["split"] == "calibration"]
    holdout = [r for r in kvasir_all if r["split"] == "source_holdout"]

    if not calibration or not holdout:
        raise ValueError("Kvasir calibration/holdout split is missing.")

    datasets = {
        "Kvasir held-out": holdout,
        "CVC": all_rows["CVC"],
        "ETIS": all_rows["ETIS"],
    }

    print("=== Dataset and segmentation performance ===")
    print(f"Kvasir calibration reference: {len(calibration)} images")
    print(f"Kvasir held-out: {len(holdout)} images")

    for name, rows in datasets.items():
        dice = np.array([num(r, "dice") for r in rows])
        iou = np.array([num(r, "iou") for r in rows])
        empty = sum(r["empty_prediction"].lower() == "true" for r in rows)
        print(
            f"{name:18s} n={len(rows):4d} | "
            f"mean Dice={dice.mean():.4f} | "
            f"median Dice={np.median(dice):.4f} | "
            f"mean IoU={iou.mean():.4f} | empty={empty}"
        )

    print("\n=== Score-distribution comparison vs Kvasir calibration ===")
    print(
        "KS tests and Wasserstein distances are exploratory; "
        "they do not establish a sequential guarantee."
    )

    comparison_rows = []
    for name, rows in datasets.items():
        print(f"\n--- {name} ---")
        for score in SCORES:
            ref = np.array([num(r, score) for r in calibration])
            target = np.array([num(r, score) for r in rows])

            ks = ks_2samp(ref, target, alternative="two-sided", method="auto")
            q_low, q_high = np.quantile(ref, [0.025, 0.975])
            outside = np.mean((target < q_low) | (target > q_high))
            wd = wasserstein_distance(ref, target)

            print(
                f"{score:32s} "
                f"ref_med={np.median(ref):.6g} "
                f"target_med={np.median(target):.6g} "
                f"KS={ks.statistic:.3f} "
                f"KS_p={ks.pvalue:.3g} "
                f"W1={wd:.6g} "
                f"outside_ref_95pct={outside:.3f}"
            )

            comparison_rows.append({
                "dataset": name,
                "score": score,
                "reference_n": len(ref),
                "target_n": len(target),
                "reference_median": float(np.median(ref)),
                "target_median": float(np.median(target)),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "wasserstein_distance": float(wd),
                "reference_q025": float(q_low),
                "reference_q975": float(q_high),
                "target_fraction_outside_reference_95pct": float(outside),
            })

    comparison_csv = OUT / "distribution_comparison.csv"
    with comparison_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    print("\n=== Failure-detection AUROC ===")
    print("Failure is defined as Dice < 0.5; GT is used only for evaluation.")
    for name, rows in datasets.items():
        summarize_performance(name, rows)

    print(f"\nDistribution table saved to: {comparison_csv}")


if __name__ == "__main__":
    main()
