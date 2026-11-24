import os
import re
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

LOG_DIR = os.path.join(os.path.dirname(__file__), "log")
FIG_DIR = os.path.join(LOG_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


import glob

def extract_iter19_losses(run_dir):
    losses = {}
    pattern = re.compile(r"layer\s+(\d+)\s+iter\s+19\s+loss:([0-9.eE+-]+)")
    for log_path in sorted(glob.glob(os.path.join(run_dir, "log_rank0_*.txt"))):
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    layer_idx = int(m.group(1))
                    loss_val = float(m.group(2))
                    losses[layer_idx] = loss_val
    return [losses[i] for i in range(22)]


def main():
    # 1. Load Hijack Ratio JSON for TinyLlama-1.1B
    hijack_path = os.path.join(LOG_DIR, "hijack_ratio_tinyllama_1.1b.json")
    with open(hijack_path, "r", encoding="utf-8") as f:
        hijack_data = json.load(f)

    layers = [item["layer"] for item in hijack_data["layers"]]
    r_vals = [item["r_mean"] for item in hijack_data["layers"]]
    r2_vals = [item["r2_mean"] for item in hijack_data["layers"]]

    # 2. Extract 22-layer iter 19 MSE losses across all 5 runs on TinyLlama-1.1B
    runs = {
        "orig_no_let": os.path.join(LOG_DIR, "1.1b_orig_no_let"),
        "orig_with_let": os.path.join(LOG_DIR, "1.1b_orig_with_gqa_let"),
        "osdsq_v1_no_let": os.path.join(LOG_DIR, "ours_tinyllama_w4a4"),
        "osdsq_v2_no_let": os.path.join(LOG_DIR, "1.1b_osdsq_v2_no_let"),
        "osdsq_v2_with_let": os.path.join(LOG_DIR, "1.1b_osdsq_with_gqa_let"),
    }
    mse_dict = {k: extract_iter19_losses(p) for k, p in runs.items()}

    # Save complete 22-layer MSE comparison table to CSV & JSON
    csv_path = os.path.join(FIG_DIR, "tinyllama_1.1b_layerwise_mse_and_hijack.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "layer", "r_hijack", "r2_inflation",
            "mse_orig_no_let", "mse_orig_with_let",
            "mse_osdsq_v1_no_let", "mse_osdsq_v2_no_let", "mse_osdsq_v2_with_let",
            "mse_ratio_no_let(orig/v2)", "mse_ratio_with_let(orig/v2)"
        ])
        for i in range(22):
            m_o_no = mse_dict["orig_no_let"][i]
            m_o_let = mse_dict["orig_with_let"][i]
            m_v1_no = mse_dict["osdsq_v1_no_let"][i]
            m_v2_no = mse_dict["osdsq_v2_no_let"][i]
            m_v2_let = mse_dict["osdsq_v2_with_let"][i]
            writer.writerow([
                i, f"{r_vals[i]:.4f}", f"{r2_vals[i]:.4f}",
                f"{m_o_no:.6f}", f"{m_o_let:.6f}",
                f"{m_v1_no:.6f}", f"{m_v2_no:.6f}", f"{m_v2_let:.6f}",
                f"{m_o_no / m_v2_no:.4f}", f"{m_o_let / m_v2_let:.4f}"
            ])

    # --- Figure 1: Range Hijack Ratio r and Variance Inflation r^2 across 22 layers ---
    fig, ax1 = plt.subplots(figsize=(9.5, 4.2), dpi=300)
    x = np.arange(len(layers))
    width = 0.55

    bars = ax1.bar(x, r_vals, width=width, color="#3B82F6", alpha=0.85, edgecolor="#1D4ED8", label=r"Range Hijack Ratio $r = R_{\mathrm{all}} / R_{\mathrm{low}}$")
    ax1.axhline(1.0, color="#6B7280", linestyle="--", linewidth=1.0, label="Ideal No-Hijack Baseline ($r = 1.0$)")
    ax1.set_xlabel("Transformer Decoder Layer Index (TinyLlama-1.1B)", fontsize=11, fontweight="bold")
    ax1.set_ylabel(r"Range Hijack Ratio $r = R_{\mathrm{all}} / R_{\mathrm{low}}$", fontsize=11, color="#1D4ED8", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(i) for i in layers], fontsize=9)
    ax1.tick_params(axis="y", labelcolor="#1D4ED8")
    ax1.set_ylim(0, max(r_vals) * 1.18)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    ax2 = ax1.twinx()
    line = ax2.plot(x, r2_vals, color="#DC2626", marker="o", linewidth=2.0, markersize=5, label=r"Theoretical Quantization Variance Inflation $r^2$")
    ax2.set_ylabel(r"Variance Inflation Factor $r^2$ (log scale)", fontsize=11, color="#DC2626", fontweight="bold")
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor="#DC2626")
    ax2.set_ylim(1.0, max(r2_vals) * 2.5)

    # Annotate peak layer (Layer 0) and final layer (Layer 21)
    ax1.annotate(f"Layer 0 Peak:\n$r={r_vals[0]:.2f}, r^2={r2_vals[0]:.1f}\\times$",
                 xy=(0, r_vals[0]), xytext=(2.2, r_vals[0] - 2.5),
                 arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.2),
                 fontsize=9, fontweight="bold", color="#991B1B",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#DC2626", lw=0.8))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9, framealpha=0.95)
    plt.title("Layer-wise Range Hijacking and Variance Inflation in TinyLlama-1.1B (W4A4, p=0.9)", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_hijack_ratio_1.1b.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig1_hijack_ratio_1.1b.pdf"))
    plt.close(fig)

    # --- Figure 2: Layer-wise Reconstruction MSE Loss (2x2 Matrix + v1) ---
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=300)

    ax_left.plot(x, mse_dict["orig_no_let"], color="#EF4444", linestyle="--", marker="s", markersize=4.5, linewidth=1.8, label="Orig. SpikeLLM (No LET) [Wiki2: 42.87]")
    ax_left.plot(x, mse_dict["orig_with_let"], color="#F59E0B", linestyle="-.", marker="^", markersize=4.5, linewidth=1.8, label="Orig. SpikeLLM + GQA-LET [Wiki2: 19.78]")
    ax_left.plot(x, mse_dict["osdsq_v1_no_let"], color="#8B5CF6", linestyle=":", marker="d", markersize=4.5, linewidth=1.8, label="OSDSQ v1 (No LET, Linear-only) [Wiki2: 16.02]")
    ax_left.plot(x, mse_dict["osdsq_v2_no_let"], color="#2563EB", linestyle="-", marker="o", markersize=4.5, linewidth=2.0, label="Ours: OSDSQ v2 (No LET) [Wiki2: 15.59]")
    ax_left.plot(x, mse_dict["osdsq_v2_with_let"], color="#10B981", linestyle="-", marker="*", markersize=6.5, linewidth=2.2, label="Ours: OSDSQ v2 + GQA-LET [Wiki2: 13.01]")

    ax_left.set_xlabel("Transformer Decoder Layer Index", fontsize=11, fontweight="bold")
    ax_left.set_ylabel("Block Reconstruction MSE Loss (Linear Scale)", fontsize=11, fontweight="bold")
    ax_left.set_title("(a) Layer-wise Error Accumulation & Tail Avalanche", fontsize=11.5, fontweight="bold")
    ax_left.set_xticks(x[::2])
    ax_left.grid(True, linestyle=":", alpha=0.6)
    ax_left.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    # Annotate Layer 21 reduction
    ax_left.annotate(f"L21: 0.3832 -> 0.1228\n(-68.0% MSE)",
                     xy=(21, mse_dict["osdsq_v2_with_let"][21]), xytext=(13.5, 0.28),
                     arrowprops=dict(arrowstyle="->", color="#047857", lw=1.3),
                     fontsize=9, fontweight="bold", color="#065F46",
                     bbox=dict(boxstyle="round,pad=0.3", fc="#ECFDF5", ec="#10B981", lw=0.8))

    # Subplot (b): Log-scale MSE across layers to highlight early-layer (L0-L15) suppression
    ax_right.plot(x, mse_dict["orig_no_let"], color="#EF4444", linestyle="--", marker="s", markersize=4.5, linewidth=1.8, label="Orig. SpikeLLM (No LET)")
    ax_right.plot(x, mse_dict["orig_with_let"], color="#F59E0B", linestyle="-.", marker="^", markersize=4.5, linewidth=1.8, label="Orig. SpikeLLM + GQA-LET")
    ax_right.plot(x, mse_dict["osdsq_v2_no_let"], color="#2563EB", linestyle="-", marker="o", markersize=4.5, linewidth=2.0, label="Ours: OSDSQ v2 (No LET)")
    ax_right.plot(x, mse_dict["osdsq_v2_with_let"], color="#10B981", linestyle="-", marker="*", markersize=6.5, linewidth=2.2, label="Ours: OSDSQ v2 + GQA-LET")
    ax_right.set_yscale("log")
    ax_right.set_xlabel("Transformer Decoder Layer Index", fontsize=11, fontweight="bold")
    ax_right.set_ylabel("Block Reconstruction MSE Loss (Log Scale)", fontsize=11, fontweight="bold")
    ax_right.set_title("(b) Log-Scale MSE Across All 22 Layers", fontsize=11.5, fontweight="bold")
    ax_right.set_xticks(x[::2])
    ax_right.grid(True, which="both", linestyle=":", alpha=0.6)
    ax_right.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_layerwise_mse_1.1b.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig2_layerwise_mse_1.1b.pdf"))
    plt.close(fig)

    # --- Figure 3: 2x2 Matrix Comparison on TinyLlama-1.1B and LLaMA-68M ---
    fig, (ax_1b, ax_68m) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=300)

    labels_1b = ["Orig. SpikeLLM\n(No LET)", "Orig. SpikeLLM\n(+ GQA-LET)", "Ours: OSDSQ v2\n(No LET)", "Ours: OSDSQ v2\n(+ GQA-LET)"]
    wiki_1b = [42.87, 19.78, 15.59, 13.01]
    c4_1b = [48.09, 23.40, 19.40, 16.36]

    pos = np.arange(len(labels_1b))
    bw = 0.35
    b1 = ax_1b.bar(pos - bw/2, wiki_1b, width=bw, color="#3B82F6", edgecolor="#1E40AF", label="WikiText-2 PPL ↓")
    b2 = ax_1b.bar(pos + bw/2, c4_1b, width=bw, color="#10B981", edgecolor="#065F46", label="C4 PPL ↓")
    ax_1b.set_xticks(pos)
    ax_1b.set_xticklabels(labels_1b, fontsize=9, fontweight="bold")
    ax_1b.set_ylabel("Perplexity (PPL ↓)", fontsize=10.5, fontweight="bold")
    ax_1b.set_title("(a) TinyLlama-1.1B (W4A4, 2x2 Ablation Matrix)", fontsize=11, fontweight="bold")
    ax_1b.set_ylim(0, 55)
    ax_1b.grid(axis="y", linestyle=":", alpha=0.6)
    ax_1b.legend(loc="upper right", fontsize=9)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax_1b.text(bar.get_x() + bar.get_width()/2.0, h + 0.8, f"{h:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    labels_68m = ["Orig. SpikeLLM\n(No LET)", "Orig. SpikeLLM\n(+ LET)", "Ours: OSDSQ v2\n(No LET)", "Ours: OSDSQ v2\n(+ LET)"]
    wiki_68m = [219.67, 178.56, 162.44, 156.09]
    c4_68m = [133.09, 103.00, 94.53, 90.92]

    b3 = ax_68m.bar(pos - bw/2, wiki_68m, width=bw, color="#3B82F6", edgecolor="#1E40AF", label="WikiText-2 PPL ↓")
    b4 = ax_68m.bar(pos + bw/2, c4_68m, width=bw, color="#10B981", edgecolor="#065F46", label="C4 PPL ↓")
    ax_68m.set_xticks(pos)
    ax_68m.set_xticklabels(labels_68m, fontsize=9, fontweight="bold")
    ax_68m.set_ylabel("Perplexity (PPL ↓)", fontsize=10.5, fontweight="bold")
    ax_68m.set_title("(b) LLaMA-68M (W4A4, 2x2 Ablation Matrix)", fontsize=11, fontweight="bold")
    ax_68m.set_ylim(0, 250)
    ax_68m.grid(axis="y", linestyle=":", alpha=0.6)
    ax_68m.legend(loc="upper right", fontsize=9)
    for bar in list(b3) + list(b4):
        h = bar.get_height()
        ax_68m.text(bar.get_x() + bar.get_width()/2.0, h + 3.0, f"{h:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_2x2_matrix_comparison.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig3_2x2_matrix_comparison.pdf"))
    plt.close(fig)

    # --- Figure 4: Leave-One-Out (LOO) Component Ablation & 3-Seed Robustness on LLaMA-68M (+ LET) ---
    fig, (ax_abl, ax_seed) = plt.subplots(1, 2, figsize=(12.4, 4.4), dpi=300)

    abl_labels = [
        "Full OSDSQ v2\n(Ours + LET)",
        "- KV Shield\n(Linear Only)",
        "- Step-Size\nSaliency",
        "- Cosine/Distill\nAlignment",
        "Orig. SpikeLLM\n(+ LET)",
        "OmniQuant\n(No Spike)"
    ]
    abl_wiki = [156.84, 158.45, 159.17, 161.28, 178.56, 179.78]
    abl_c4 = [90.53, 90.53, 90.63, 91.47, 103.00, 108.45]

    pa = np.arange(len(abl_labels))
    ba1 = ax_abl.bar(pa - bw/2, abl_wiki, width=bw, color="#3B82F6", edgecolor="#1E40AF", label="WikiText-2 PPL ↓")
    ba2 = ax_abl.bar(pa + bw/2, abl_c4, width=bw, color="#10B981", edgecolor="#065F46", label="C4 PPL ↓")
    ax_abl.set_xticks(pa)
    ax_abl.set_xticklabels(abl_labels, fontsize=8, fontweight="bold")
    ax_abl.set_ylabel("Perplexity (PPL ↓)", fontsize=10.5, fontweight="bold")
    ax_abl.set_title("(a) Leave-One-Out (LOO) Ablation (LLaMA-68M + LET)", fontsize=11, fontweight="bold")
    ax_abl.set_ylim(75, 195)
    ax_abl.grid(axis="y", linestyle=":", alpha=0.6)
    ax_abl.legend(loc="upper left", fontsize=8.5)
    for bar in list(ba1) + list(ba2):
        h = bar.get_height()
        ax_abl.text(bar.get_x() + bar.get_width()/2.0, h + 1.2, f"{h:.2f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold")

    # 3-Seed Mean ± Std Error Bar Plot
    seed_labels = ["Orig. SpikeLLM (+ LET)\n(3 Seeds: 0, 1, 2)", "Ours: OSDSQ v2 (+ LET)\n(3 Seeds: 0, 1, 2)"]
    wiki_means = [175.33, 155.47]
    wiki_stds = [2.56, 1.21]
    c4_means = [100.93, 90.21]
    c4_stds = [1.49, 0.30]

    ps = np.arange(len(seed_labels))
    bs1 = ax_seed.bar(ps - bw/2, wiki_means, yerr=wiki_stds, capsize=5, width=bw, color="#2563EB", edgecolor="#1E3A8A", label="WikiText-2 (Mean ± Std)")
    bs2 = ax_seed.bar(ps + bw/2, c4_means, yerr=c4_stds, capsize=5, width=bw, color="#059669", edgecolor="#064E3B", label="C4 (Mean ± Std)")
    ax_seed.set_xticks(ps)
    ax_seed.set_xticklabels(seed_labels, fontsize=9.5, fontweight="bold")
    ax_seed.set_ylabel("Perplexity (PPL ↓)", fontsize=10.5, fontweight="bold")
    ax_seed.set_title("(b) Multi-Seed Robustness (Mean ± Std across Seeds)", fontsize=11, fontweight="bold")
    ax_seed.set_ylim(70, 195)
    ax_seed.grid(axis="y", linestyle=":", alpha=0.6)
    ax_seed.legend(loc="upper right", fontsize=9)
    for bar, std_v in zip(list(bs1) + list(bs2), wiki_stds + c4_stds):
        h = bar.get_height()
        ax_seed.text(bar.get_x() + bar.get_width()/2.0, h + std_v + 1.8, f"{h:.2f}±{std_v:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_ablation_and_multiseed_68m.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig4_ablation_and_multiseed_68m.pdf"))
    plt.close(fig)

    # --- Figure 5: Zero-Shot 6-Task Reasoning & Multi-Step Spike Simulation on TinyLlama-1.1B ---
    fig, (ax_zs, ax_mb) = plt.subplots(1, 2, figsize=(12.6, 4.5), dpi=300)

    tasks_labels = ["PIQA", "ARC-E", "ARC-C", "BoolQ", "WinoG.", "HellaS.", "AVG ↑"]
    orig_no_let_zs = [54.73, 33.50, 24.66, 60.31, 49.09, 35.40, 42.95]
    orig_let_zs = [60.23, 37.46, 27.82, 59.82, 52.57, 44.40, 47.05]
    v2_no_let_zs = [59.47, 36.78, 26.96, 55.41, 50.04, 46.00, 45.78]
    v2_let_zs = [63.33, 38.43, 27.22, 58.23, 53.75, 45.90, 47.81]

    pz = np.arange(len(tasks_labels))
    w_z = 0.20
    ax_zs.bar(pz - 1.5*w_z, orig_no_let_zs, width=w_z, color="#EF4444", label="Orig. (No LET) [Avg: 42.95%]")
    ax_zs.bar(pz - 0.5*w_z, v2_no_let_zs, width=w_z, color="#3B82F6", label="OSDSQ v2 (No LET) [Avg: 45.78%]")
    ax_zs.bar(pz + 0.5*w_z, orig_let_zs, width=w_z, color="#F59E0B", label="Orig. (+ GQA-LET) [Avg: 47.05%]")
    b_top = ax_zs.bar(pz + 1.5*w_z, v2_let_zs, width=w_z, color="#10B981", label="OSDSQ v2 (+ GQA-LET) [Avg: 47.81%]")
    ax_zs.set_xticks(pz)
    ax_zs.set_xticklabels(tasks_labels, fontsize=9, fontweight="bold")
    ax_zs.set_ylabel("Zero-Shot Accuracy (% ↑)", fontsize=10.5, fontweight="bold")
    ax_zs.set_title("(a) 6-Task Zero-Shot Reasoning on TinyLlama-1.1B (W4A4)", fontsize=11, fontweight="bold")
    ax_zs.set_ylim(20, 70)
    ax_zs.grid(axis="y", linestyle=":", alpha=0.6)
    ax_zs.legend(loc="upper right", fontsize=8)
    for bar in b_top:
        h = bar.get_height()
        ax_zs.text(bar.get_x() + bar.get_width()/2.0, h + 0.6, f"{h:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold", color="#065F46")

    # Subplot (b): Multi-Step Spike Simulation (fake_binary vs fake_multibit vs real_multibit fixed) on 1.1B
    mb_labels = ["Orig. + GQA-LET\n(Real Spike Fixed)", "OSDSQ v2 + GQA-LET\n(Binary Sim.)", "OSDSQ v2 + GQA-LET\n(Fake Multi-Bit)", "OSDSQ v2 + GQA-LET\n(Real Spike Fixed)"]
    mb_wiki = [22.39, 13.01, 14.36, 14.36]
    mb_c4 = [28.47, 16.36, 18.09, 18.13]
    pm = np.arange(len(mb_labels))
    bm1 = ax_mb.bar(pm - bw/2, mb_wiki, width=bw, color="#2563EB", edgecolor="#1E3A8A", label="WikiText-2 PPL ↓")
    bm2 = ax_mb.bar(pm + bw/2, mb_c4, width=bw, color="#059669", edgecolor="#064E3B", label="C4 PPL ↓")
    ax_mb.set_xticks(pm)
    ax_mb.set_xticklabels(mb_labels, fontsize=8.5, fontweight="bold")
    ax_mb.set_ylabel("Perplexity (PPL ↓)", fontsize=10.5, fontweight="bold")
    ax_mb.set_title("(b) Real Multi-Step SNN Spike Simulation (TinyLlama-1.1B)", fontsize=11, fontweight="bold")
    ax_mb.set_ylim(0, 33)
    ax_mb.grid(axis="y", linestyle=":", alpha=0.6)
    ax_mb.legend(loc="upper right", fontsize=8.5)
    for bar in list(bm1) + list(bm2):
        h = bar.get_height()
        ax_mb.text(bar.get_x() + bar.get_width()/2.0, h + 0.5, f"{h:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_zeroshot_and_multibit_1.1b.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig5_zeroshot_and_multibit_1.1b.pdf"))
    plt.close(fig)

    print("Successfully generated figures in:", FIG_DIR)


if __name__ == "__main__":
    main()
