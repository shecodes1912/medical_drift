import os
import numpy as np
import matplotlib.pyplot as plt
from config import cfg
from martingale import ConformalMartingaleMonitor

def plot_martingale_trajectories(
    source_scores: np.ndarray,
    target_scores_dict: dict,
    N_cal: int = 50,
    max_steps: int = 50,
    save_path: str = os.path.join(cfg.output_dir, "fig1_martingale_trajectories.png")
):
    """
    Generates Figure 1 for the paper: Martingale value M_t over streaming time steps t.
    Compares H0 (In-Distribution) vs. Mild Shift vs. Hard Shift.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    # 1. Fit calibration once on source domain
    monitor = ConformalMartingaleMonitor(delta=cfg.delta, epsilon=cfg.epsilon)
    cal_scores = source_scores[:N_cal]
    
    # 2. Simulate stream for H0 (In-distribution held-out)
    h0_stream = source_scores[N_cal:N_cal + max_steps]
    monitor.fit_calibration(cal_scores)
    m_h0 = [1.0]
    for s in h0_stream:
        res = monitor.update(s)
        m_h0.append(res["martingale_value"])
        
    ax.plot(range(len(m_h0)), m_h0, label="H0: In-Distribution (Kvasir-SEG)", color="gray", linestyle="--", linewidth=2)
    
    # 3. Simulate stream for targets (Shifted domains)
    colors = {"CVC-ClinicDB (Mild)": "orange", "ETIS-LaribPolypDB (Hard)": "red", "MoNuSeg (Cross-Modality)": "purple"}
    
    for domain_name, target_scores in target_scores_dict.items():
        monitor.fit_calibration(cal_scores)
        m_target = [1.0]
        stream = target_scores[:max_steps]
        for s in stream:
            res = monitor.update(s)
            m_target.append(res["martingale_value"])
            
        color = colors.get(domain_name, "blue")
        ax.plot(range(len(m_target)), m_target, label=f"Shift: {domain_name}", color=color, linewidth=2.5)

    # 4. Plot Ville's Inequality Threshold (1 / delta)
    threshold = 1.0 / cfg.delta
    ax.axhline(y=threshold, color="black", linestyle=":", linewidth=2, label=f"Alarm Threshold 1/δ ({threshold:.0f})")
    
    # Formatting
    ax.set_yscale("log")
    ax.set_xlabel("Deployment Image Sequence Step (t)", fontsize=12)
    ax.set_ylabel("Martingale Value $M_t$ (Log Scale)", fontsize=12)
    ax.set_title("Sequential Conformal Drift Detection Trajectories", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Paper Figure saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    np.random.seed(42)
    
    kvasir_scores = np.random.normal(loc=0.12, scale=0.02, size=200)
    cvc_scores = np.random.normal(loc=0.19, scale=0.03, size=100)
    etis_scores = np.random.normal(loc=0.28, scale=0.04, size=100)
    monuseg_scores = np.random.normal(loc=0.45, scale=0.06, size=100)
    
    targets = {
        "CVC-ClinicDB (Mild)": cvc_scores,
        "ETIS-LaribPolypDB (Hard)": etis_scores,
        "MoNuSeg (Cross-Modality)": monuseg_scores
    }
    
    plot_martingale_trajectories(kvasir_scores, targets, N_cal=50, max_steps=40)