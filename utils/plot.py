import numpy as np
import matplotlib.pyplot as plt


def plot_pr_curve(pr_data, best_idx, save_path=None, auprc=None):
    """
    pr_data: dict with tensors/arrays
        - "precision": length N+1
        - "recall":    length N+1
        - "thresholds": length N   (no 0 or 1)
    best_idx: index into precision/recall (0..N) where your metric peaks
    save_path: path to save PNG (if None, just shows)
    auprc: optional float to print in title
    """
    # to numpy
    precision = np.asarray(pr_data["precision"])
    recall    = np.asarray(pr_data["recall"])
    thresholds = np.asarray(pr_data["thresholds"])

    # Build a threshold vector aligned to precision/recall (N+1)
    # [0.0] + thresholds + [1.0]
    thr_aligned = np.concatenate(([0.0], thresholds, [1.0]))

    # AUPRC if not provided
    if auprc is None:
        # torchmetrics returns PR sorted by recall descending; trapz works either way
        auprc = float(np.trapz(precision, recall))

    # Plot PR curve
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve (AUPRC = {auprc:.4f})")
    plt.grid(True, alpha=0.3)

    # Mark best operating point
    bx, by = recall[best_idx], precision[best_idx]
    bthr = thr_aligned[best_idx]
    plt.scatter([bx], [by], s=60)
    plt.annotate(f"best @ τ={bthr:.3f}\nP={by:.3f}, R={bx:.3f}",
                 (bx, by), textcoords="offset points", xytext=(8, -18))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()