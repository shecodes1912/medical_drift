import numpy as np
from config import cfg
from martingale import ConformalMartingaleMonitor

def run_experiment_4_score_ablation(
    source_scores_dict: dict,
    target_scores_dict: dict,
    target_name: str,
    N_cal: int = 50,
    num_runs: int = 50
):
    """
    Evaluates detection latency and miss rate across nonconformity scores S1, S2, and S3.
    Demonstrates which output statistic is most sensitive to specific domain shifts.
    """
    print(f"\n=======================================================")
    print(f"Running Experiment 4: Score Function Ablation ({target_name})")
    print(f"Calibration Size N={N_cal}")
    print(f"=======================================================")
    
    results = {}
    score_names = ["S1 (Entropy)", "S2 (Prediction Set Size)", "S3 (Fragmentation)"]
    
    for score_key in ["S1", "S2", "S3"]:
        source_scores = source_scores_dict[score_key]
        target_scores = target_scores_dict[score_key]
        
        latencies = []
        misses = 0
        monitor = ConformalMartingaleMonitor(delta=cfg.delta, epsilon=cfg.epsilon)
        
        for run in range(num_runs):
            cal_indices = np.random.choice(len(source_scores), size=N_cal, replace=False)
            cal_scores = source_scores[cal_indices]
            
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
        
        results[score_key] = {
            "mean_latency": mean_lat,
            "std_latency": std_lat,
            "miss_rate": miss_rate
        }
        
        print(f"Score {score_key:2s} | Latency: {mean_lat:6.2f} ± {std_lat:5.2f} images | Miss Rate: {miss_rate*100:4.1f}%")

    return results

if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate mock distributions simulating different score behaviors
    # S1: Entropy (Detects general uncertainty shift)
    source_S1 = np.random.normal(loc=0.12, scale=0.02, size=300)
    target_S1 = np.random.normal(loc=0.28, scale=0.04, size=150)
    
    # S2: Set Size (Detects boundary uncertainty)
    source_S2 = np.random.normal(loc=0.05, scale=0.01, size=300)
    target_S2 = np.random.normal(loc=0.18, scale=0.03, size=150)
    
    # S3: Fragmentation (Detects structural disintegration)
    source_S3 = np.random.normal(loc=1.20, scale=0.40, size=300)
    target_S3 = np.random.normal(loc=4.50, scale=1.20, size=150)
    
    source_dict = {"S1": source_S1, "S2": source_S2, "S3": source_S3}
    target_dict = {"S1": target_S1, "S2": target_S2, "S3": target_S3}
    
    run_experiment_4_score_ablation(
        source_scores_dict=source_dict,
        target_scores_dict=target_dict,
        target_name="ETIS-LaribPolypDB (Hard Shift)",
        N_cal=50,
        num_runs=50
    )
    print("\n✅ Experiment 4 Completed Successfully.")