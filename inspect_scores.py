
import pandas as pd

files = {
    "Kvasir": "outputs/kvasir_calibration/kvasir_scores_all.csv",
    "CVC": "outputs/cvc_evaluation/cvc_scores_all.csv",
    "ETIS": "outputs/etis_evaluation/etis_full196.csv",
}

columns_to_show = [
    "split",
    "sequence_id",
    "frame_id",
    "image_id",
    "mean_entropy_nats",
    "p90_pixel_entropy_nats",
    "low_confidence_fraction",
    "pred_foreground_fraction",
    "dice",
]

for name, path in files.items():
    print(f"\n=== {name} ===")

    df = pd.read_csv(path)

    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    available = [c for c in columns_to_show if c in df.columns]
    print(df[available].head(5).to_string(index=False))
