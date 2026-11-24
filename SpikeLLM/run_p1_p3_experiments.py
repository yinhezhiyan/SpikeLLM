import os
import json
import time
import subprocess
import sys

PYTHON_EXE = sys.executable
BASE_CMD = [
    PYTHON_EXE, "main.py",
    "--model", "JackFram/llama-68m",
    "--wbits", "4", "--abits", "4",
    "--lwc", "--batch_size", "1",
    "--eval_ppl"
]

EXPERIMENTS = [
    # 1. Regression Check: Full OSDSQ v2 + LET (seed=2)
    ("reg_v2_let_seed2", ["--epochs", "20", "--let", "--aug_loss", "--seed", "2", "--addbit", "1", "--low_p", "0.9", "--output_dir", "./log/p1_reg_v2_let_s2"]),
    # 2. P1 Cumulative Ablations on 68M + LET (seed=2)
    ("abl_step1_linear_only", ["--epochs", "20", "--let", "--aug_loss", "--seed", "2", "--addbit", "1", "--low_p", "0.9", "--ablate_no_kv_shield", "--ablate_orig_saliency", "--ablate_orig_loss", "--output_dir", "./log/p1_abl_step1_linear"]),
    ("abl_step2_linear_kv", ["--epochs", "20", "--let", "--aug_loss", "--seed", "2", "--addbit", "1", "--low_p", "0.9", "--ablate_orig_saliency", "--ablate_orig_loss", "--output_dir", "./log/p1_abl_step2_linear_kv"]),
    ("abl_step3_plus_saliency", ["--epochs", "20", "--let", "--aug_loss", "--seed", "2", "--addbit", "1", "--low_p", "0.9", "--ablate_orig_loss", "--output_dir", "./log/p1_abl_step3_saliency"]),
    # 3. External Baseline: Pure OmniQuant W4A4 on 68M (seed=2)
    ("ext_omniquant_w4a4_s2", ["--epochs", "20", "--let", "--seed", "2", "--addbit", "0", "--low_p", "1.0", "--orig_spikellm", "--output_dir", "./log/p4_68m_omniquant_w4a4"]),
    # 4. Multi-seed on 68M + LET (seed=0, 1)
    ("orig_let_seed0", ["--epochs", "20", "--let", "--aug_loss", "--seed", "0", "--addbit", "1", "--low_p", "0.9", "--orig_spikellm", "--output_dir", "./log/p1_orig_let_s0"]),
    ("orig_let_seed1", ["--epochs", "20", "--let", "--aug_loss", "--seed", "1", "--addbit", "1", "--low_p", "0.9", "--orig_spikellm", "--output_dir", "./log/p1_orig_let_s1"]),
    ("v2_let_seed0", ["--epochs", "20", "--let", "--aug_loss", "--seed", "0", "--addbit", "1", "--low_p", "0.9", "--output_dir", "./log/p1_v2_let_s0"]),
    ("v2_let_seed1", ["--epochs", "20", "--let", "--aug_loss", "--seed", "1", "--addbit", "1", "--low_p", "0.9", "--output_dir", "./log/p1_v2_let_s1"]),
    # 5. P3 Multi-step Spike Simulation on 68M (resuming from reg_v2_let_seed2 & orig_let)
    ("p3_v2_fake_multibit", ["--epochs", "0", "--resume", "./log/p1_reg_v2_let_s2/omni_parameters.pth", "--let", "--aug_loss", "--seed", "2", "--addbit", "1", "--low_p", "0.9", "--sim_mode", "fake_multibit_simulate", "--output_dir", "./log/p3_68m_v2_fake_multibit"]),
    ("p3_v2_real_multibit_legacy_bug", ["--epochs", "0", "--resume", "./log/p1_reg_v2_let_s2/omni_parameters.pth", "--let", "--aug_loss", "--seed", "2", "--addbit", "1", "--low_p", "0.9", "--sim_mode", "real_multibit_simulate", "--legacy_multibit_bug", "--output_dir", "./log/p3_68m_v2_real_legacy_bug"]),
    ("p3_v2_real_multibit_fixed", ["--epochs", "0", "--resume", "./log/p1_reg_v2_let_s2/omni_parameters.pth", "--let", "--aug_loss", "--seed", "2", "--addbit", "1", "--low_p", "0.9", "--sim_mode", "real_multibit_simulate", "--output_dir", "./log/p3_68m_v2_real_fixed"]),
    ("p3_orig_real_multibit_legacy_bug", ["--epochs", "0", "--resume", "./log/p1_orig_let_s0/omni_parameters.pth", "--let", "--aug_loss", "--seed", "0", "--addbit", "1", "--low_p", "0.9", "--orig_spikellm", "--sim_mode", "real_multibit_simulate", "--legacy_multibit_bug", "--output_dir", "./log/p3_68m_orig_real_legacy_bug"]),
    ("p3_orig_real_multibit_fixed", ["--epochs", "0", "--resume", "./log/p1_orig_let_s0/omni_parameters.pth", "--let", "--aug_loss", "--seed", "0", "--addbit", "1", "--low_p", "0.9", "--orig_spikellm", "--sim_mode", "real_multibit_simulate", "--output_dir", "./log/p3_68m_orig_real_fixed"]),
    # 6. Leave-One-Out (LOO) Ablations on 68M + LET (seed=2)
    ("loo_minus_kv_shield", ["--epochs", "20", "--let", "--aug_loss", "--seed", "2", "--addbit", "1", "--low_p", "0.9", "--ablate_no_kv_shield", "--output_dir", "./log/p1_loo_minus_kv"]),
    ("loo_minus_saliency", ["--epochs", "20", "--let", "--aug_loss", "--seed", "2", "--addbit", "1", "--low_p", "0.9", "--ablate_orig_saliency", "--output_dir", "./log/p1_loo_minus_saliency"]),
]


def parse_ppl(stdout_text):
    res = {}
    for line in stdout_text.splitlines():
        if "wikitext2 :" in line:
            res["wikitext2"] = float(line.split("wikitext2 :")[1].strip())
        elif "c4 :" in line:
            res["c4"] = float(line.split("c4 :")[1].strip())
    return res


def main():
    summary_path = "./log/p1_p3_68m_summary.json"
    summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            summary = {}

    for name, extra_args in EXPERIMENTS:
        if name in summary and "wikitext2" in summary[name] and "c4" in summary[name]:
            print(f"[SKIP] {name} already completed: {summary[name]}", flush=True)
            continue
        t0 = time.time()
        cmd = BASE_CMD + extra_args
        print(f"=== Running {name} ===", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        elapsed = time.time() - t0
        ppls = parse_ppl(proc.stdout + "\n" + proc.stderr)
        ppls["elapsed_sec"] = round(elapsed, 2)
        ppls["returncode"] = proc.returncode
        summary[name] = ppls
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"=== Finished {name} in {elapsed:.1f}s: {ppls} ===", flush=True)
        if proc.returncode != 0:
            print("STDERR:", proc.stderr[-500:], flush=True)


if __name__ == "__main__":
    main()
