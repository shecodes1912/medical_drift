import numpy as np
import torch
from config import cfg
from datasets import get_dataloader
from models import FrozenSegmenterWrapper, BaselineDummySegmenter
from nonconformity import extract_nonconformity_score
from martingale import ConformalMartingaleMonitor

def collect_scores(model_wrapper: FrozenSegmenterWrapper, dataloader, score_type: str = "S1") -> np.ndarray:
    """Collects nonconformity scores across an entire dataset loader."""
    scores = []
    for batch in dataloader:
        imgs = batch["image"]
        probs = model_wrapper.predict_probs(imgs)
        score = extract_nonconformity_score(probs, score_type=score_type)
        scores.append(score)
    return np.array(scores)


def run_experiment_1_drift_latency(
    source_scores: np.ndarray,
    target_scores: np.ndarray,
    target_name: str,
    N_cal: int = 50,
    num_runs: int = 20,
    score_type: str = "S1"
):
    """
    Runs Experiment 1: Evaluates detection latency and miss rate under domain shift.
    Proves martingale alarm fires rapidly under shift.
    """
    print(f"\n=======================================================")
    print(f"Running Experiment 1: Drift Detection Latency ({target_name})")
    print(f"Calibration Size N={N_cal} | Score Function={score_type}")
    print(f"=======================================================")
    
    latencies = []
    misses = 0
    monitor = ConformalMartingaleMonitor(delta=cfg.delta, epsilon=cfg.epsilon)
    
    for run in range(num_runs):
        # Sample calibration set randomly from source distribution
        cal_indices = np.random.choice(len(source_scores), size=N_cal, replace=False)
        cal_scores = source_scores[cal_indices]
        
        # Shuffle target stream to average over orderings
        target_stream = np.random.permutation(target_scores)
        
        # Phase 1: Offline Calibration
        monitor.fit_calibration(cal_scores)
        
        # Phase 2: Online Deployment Monitoring
        alarm_triggered = False
        for t, score in enumerate(target_stream, start=1):
            res = monitor.update(score)
            if res["alarm"]:
                latencies.append(t)
                alarm_triggered = True
                break
                
        if not alarm_triggered:
            misses += 1

    miss_rate = misses / num_runs
    mean_latency = np.mean(latencies) if latencies else float("nan")
    std_latency = np.std(latencies) if latencies else float("nan")
    
    print(f"Target Domain: {target_name}")
    print(f"Detection Latency T_alarm: {mean_latency:.2f} ± {std_latency:.2f} images")
    print(f"Miss Rate (Alarm never fired): {miss_rate * 100:.1f}%")
    
    return {
        "target": target_name,
        "mean_latency": mean_latency,
        "std_latency": std_latency,
        "miss_rate": miss_rate
    }


if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate representative entropy distributions for pipeline validation
    # Kvasir (Source): low entropy (well calibrated)
    kvasir_scores = np.random.normal(loc=0.12, scale=0.02, size=200)
    
    # CVC-ClinicDB (Mild Shift): slightly elevated entropy
    cvc_scores = np.random.normal(loc=0.19, scale=0.03, size=200)
    
    # ETIS-LaribPolypDB (Hard Shift): noticeably high entropy
    etis_scores = np.random.normal(loc=0.28, scale=0.04, size=150)
    
    # Run Mild Shift Benchmark
    run_experiment_1_drift_latency(kvasir_scores, cvc_scores, target_name="CVC-ClinicDB (Mild Shift)", N_cal=50)
    
    # Run Hard Shift Benchmark
    run_experiment_1_drift_latency(kvasir_scores, etis_scores, target_name="ETIS-LaribPolypDB (Hard Shift)", N_cal=50)