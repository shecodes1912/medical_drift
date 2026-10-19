import numpy as np
import matplotlib.pyplot as plt
import os
from config import cfg
from martingale import ConformalMartingaleMonitor

def run_experiment_3_calibration_sweep(
    source_scores: np.ndarray,
    target_scores: np.ndarray,
    target_name: str,
    cal_sizes: list = [10, 25, 50, 100, 200],
    num_runs: int = 50
):
    """
    Evaluates detection latency and miss rate across different calibration set sizes (N_cal).
    Proves that small calibration sets (N ~ 25-50) are sufficient for deployment.
    """
    print(f"\n=======================================================")
    print(f"Running Experiment 3: Calibration Size Sweep ({target_name})")
    print(f"Sweeping N_cal in {cal_sizes}")
    print(f"=======================================================")
    
    results = {}
    
    for N_cal in cal_sizes:
        if N_cal >= len(source_scores):
            print(f"Skipping N_cal={N_cal} (insufficient source samples: {len(source_scores)})")
            continue
            
        latencies = []
        misses = 0
        monitor = ConformalMartingaleMonitor(delta=cfg.delta, epsilon=cfg.epsilon)
        
        for run in range(num_runs):
            # Sample N_cal calibration images from source domain
            cal_indices = np.random.choice(len(source_scores), size=N_cal, replace=False)
            cal_scores = source_scores[cal_indices]
            
            # Target deployment stream
            target_stream = np.random.permutation(target_scores)
            
            monitor.fit_calibration(cal_scores)
            
            alarm_triggered = False
            for t, score in enumerate(target_stream, start=1):
                res = monitor.update(score)
                if res["alarm"]:
                    latencies.append(t)
                    alarm_triggered = True
                    break
                    
            if not alarm_triggered:
                misses += 1

        mean_lat = np.mean(latencies) if latencies else float("nan")
        std_lat = np.std(latencies) if latencies else float("nan")
        miss_rate = misses / num_runs
        
        results[N_cal] = {
            "mean_latency": mean_lat,
            "std_latency": std_lat,
            "miss_rate": miss_rate
        }
        
        print(f"N_cal = {N_cal:3d} | Detection Latency: {mean_lat:6.2f} ± {std_lat:5.2f} images | Miss Rate: {miss_rate*100:4.1f}%")

    return results

if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate mock score distributions for sweep testing
    # Kvasir-SEG (Source distribution)
    kvasir_scores = np.random.normal(loc=0.12, scale=0.02, size=300)
    
    # ETIS-LaribPolypDB (Hard Distribution Shift)
    etis_scores = np.random.normal(loc=0.28, scale=0.04, size=150)
    
    sweep_results = run_experiment_3_calibration_sweep(
        source_scores=kvasir_scores,
        target_scores=etis_scores,
        target_name="ETIS-LaribPolypDB (Hard Shift)",
        cal_sizes=cfg.cal_sizes,
        num_runs=50
    )
    print("\n✅ Experiment 3 Completed Successfully.")