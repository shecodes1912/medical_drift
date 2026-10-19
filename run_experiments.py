import numpy as np
from config import cfg
from martingale import ConformalMartingaleMonitor
from nonconformity import extract_nonconformity_score

def run_experiment_2_false_alarm_validation(in_dist_scores: np.ndarray, N_cal: int = 50):
    """
    Validates Ville's Inequality: P(exists t : M_t > 1/delta | H0) <= delta
    Runs 500 stream orderings on held-out in-distribution data.
    """
    print(f"--- Running Experiment 2: False Alarm Rate under H0 (N_cal={N_cal}) ---")
    monitor = ConformalMartingaleMonitor(delta=cfg.delta, epsilon=cfg.epsilon)
    false_alarm_count = 0
    
    # Filter valid dataset size
    total_samples = len(in_dist_scores)
    assert total_samples > N_cal, "Insufficient samples for calibration and test split."
    
    for run in range(cfg.num_runs):
        # Permute stream randomly to simulate streaming orderings
        permuted_scores = np.random.permutation(in_dist_scores)
        cal_scores = permuted_scores[:N_cal]
        test_scores = permuted_scores[N_cal:]
        
        monitor.fit_calibration(cal_scores)
        
        # Deploy sequentially over stream
        alarm_raised = False
        for score in test_scores:
            res = monitor.update(score)
            if res["alarm"]:
                alarm_raised = True
                break  # Alarm triggered
                
        if alarm_raised:
            false_alarm_count += 1

    empirical_far = false_alarm_count / cfg.num_runs
    print(f"Target Delta (Theoretical Upper Bound): {cfg.delta}")
    print(f"Empirical False Alarm Rate across {cfg.num_runs} runs: {empirical_far:.4f}")
    assert empirical_far <= cfg.delta + 0.02, "Empirical false alarm rate exceeds bound + slack!"
    print("✅ Experiment 2 Passed: Theoretical guarantee verified.\n")
    return empirical_far

if __name__ == "__main__":
    # Generate mock in-distribution entropy scores (e.g. mean 0.15, std 0.03) for initial verification
    np.random.seed(42)
    mock_kvasir_scores = np.random.normal(loc=0.15, scale=0.03, size=300)
    
    # Run Exp 2
    run_experiment_2_false_alarm_validation(mock_kvasir_scores, N_cal=50)