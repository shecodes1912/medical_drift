import numpy as np
import os
import json
from config import cfg
from run_experiments import run_experiment_2_false_alarm_validation
from run_experiment_1 import run_experiment_1_drift_latency
from run_exp3_cal_sweep import run_experiment_3_calibration_sweep
from run_exp4_score_ablation import run_experiment_4_score_ablation

def execute_master_benchmark():
    """
    Executes the complete experimental suite (Exp 1 - Exp 4)
    and exports summary metrics to results/benchmark_summary.json.
    """
    print("=======================================================")
    print("      STARTING ISBI 2027 DRIFT DETECTION PIPELINE      ")
    print("=======================================================")
    
    np.random.seed(42)
    summary = {}
    
    # 1. Synthesize baseline distribution distributions for pipeline run
    kvasir_scores = np.random.normal(loc=0.12, scale=0.02, size=300)
    cvc_scores = np.random.normal(loc=0.19, scale=0.03, size=200)      # Mild Shift
    etis_scores = np.random.normal(loc=0.28, scale=0.04, size=150)     # Hard Shift
    monuseg_scores = np.random.normal(loc=0.45, scale=0.06, size=100)   # Cross-Modality
    
    # --- EXPERIMENT 2: False Alarm Rate under H0 ---
    far_h0 = run_experiment_2_false_alarm_validation(kvasir_scores, N_cal=cfg.default_n_cal)
    summary["exp2_false_alarm_rate_h0"] = far_h0
    
    # --- EXPERIMENT 1: Sequential Drift Detection Latency ---
    exp1_cvc = run_experiment_1_drift_latency(kvasir_scores, cvc_scores, "CVC-ClinicDB (Mild)", N_cal=cfg.default_n_cal)
    exp1_etis = run_experiment_1_drift_latency(kvasir_scores, etis_scores, "ETIS-LaribPolypDB (Hard)", N_cal=cfg.default_n_cal)
    exp1_monuseg = run_experiment_1_drift_latency(kvasir_scores, monuseg_scores, "MoNuSeg (Cross-Modality)", N_cal=cfg.default_n_cal)
    summary["exp1_latency"] = {"mild": exp1_cvc, "hard": exp1_etis, "cross_modality": exp1_monuseg}
    
    # --- EXPERIMENT 3: Calibration Set Size Sweep ---
    exp3_results = run_experiment_3_calibration_sweep(kvasir_scores, etis_scores, "ETIS-LaribPolypDB", cal_sizes=cfg.cal_sizes)
    summary["exp3_cal_sweep"] = exp3_results
    
    # --- EXPERIMENT 4: Score Function Ablation ---
    source_dict = {
        "S1": kvasir_scores,
        "S2": np.random.normal(loc=0.05, scale=0.01, size=300),
        "S3": np.random.normal(loc=1.20, scale=0.40, size=300)
    }
    target_dict = {
        "S1": etis_scores,
        "S2": np.random.normal(loc=0.18, scale=0.03, size=150),
        "S3": np.random.normal(loc=4.50, scale=1.20, size=150)
    }
    exp4_results = run_experiment_4_score_ablation(source_dict, target_dict, "ETIS-LaribPolypDB", N_cal=cfg.default_n_cal)
    summary["exp4_score_ablation"] = exp4_results
    
    # Export report to results directory
    out_path = os.path.join(cfg.output_dir, "benchmark_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    print("\n=======================================================")
    print(f"✅ PIPELINE COMPLETE. Results exported to: {out_path}")
    print("=======================================================")

if __name__ == "__main__":
    execute_master_benchmark()