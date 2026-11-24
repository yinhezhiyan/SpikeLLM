"""实验环境指纹报告 —— 每次实验跑完随结果一起回传。

目的：让跨批次、跨机器的实验结果可比。任何一次结果若要与其他批次并列比较，
必须先确认本报告的关键字段一致。

输出纯文本（utf-8），可直接粘贴回传，不含任何开发机路径外的敏感信息。

用法:
    python env_report.py                    # 打印到屏幕
    python env_report.py --out env.txt      # 同时写入文件
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime

# 影响数值结果的源文件：任何一个 hash 变化都会使实验不可比
CRITICAL_FILES = [
    "main.py",
    "spike_driven_quant/spike_linear.py",
    "spike_driven_quant/spike_matmul.py",
    "spike_driven_quant/spike_omniquant.py",
    "spike_driven_quant/quantizer.py",
    "models/transformation.py",
    "datautils.py",
    "utils.py",
]

CRITICAL_PKGS = ["torch", "transformers", "accelerate", "datasets", "numpy",
                 "sentencepiece", "tokenizers", "lm_eval"]


def sha256(path, limit=None):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(1 << 20)
                if not b:
                    break
                h.update(b)
                if limit and f.tell() > limit:
                    break
        return h.hexdigest()[:16]
    except Exception as e:
        return f"ERR({type(e).__name__})"


def run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return (r.stdout or r.stderr).strip()
    except Exception:
        return "n/a"


def pkg_ver(name):
    try:
        m = __import__(name)
        return getattr(m, "__version__", "unknown")
    except Exception:
        return "NOT_INSTALLED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--note", type=str, default="", help="本次实验标识，如 68M_step2_rerun")
    args = ap.parse_args()

    L = []
    add = L.append
    add("=" * 72)
    add("EXPERIMENT ENVIRONMENT FINGERPRINT")
    add(f"time      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.note:
        add(f"tag       : {args.note}")
    add("=" * 72)

    # --- 代码状态 ---
    add("[1] CODE STATE")
    add(f"cwd       : {os.getcwd()}")
    commit = run(["git", "rev-parse", "--short", "HEAD"])
    dirty = run(["git", "status", "--porcelain"])
    git_broken = ("fatal:" in commit) or ("fatal:" in dirty) or commit == "n/a"
    if git_broken:
        add(f"git commit: UNKNOWN  <-- git 不可用（常见原因：dubious ownership）")
        add("            修复: git config --global --add safe.directory <repo_path>")
        add(f"            原始信息: {(commit or dirty).splitlines()[0][:100]}")
    else:
        add(f"git commit: {commit}")
        if dirty.strip():
            n = len([x for x in dirty.splitlines() if x.strip()])
            add(f"git dirty : YES ({n} files modified)  <-- 未提交改动会使结果不可复现")
            for line in dirty.splitlines()[:10]:
                add(f"            {line.strip()}")
        else:
            add("git dirty : no")
    add(f"git usable: {'NO' if git_broken else 'yes'}  "
        f"(为 no 时只能靠 [2] 的文件哈希判断代码是否变化)")
    add("")

    # --- 关键文件哈希 ---
    add("[2] CRITICAL FILE HASHES  (任一项变化 => 与前批次结果不可比)")
    for f in CRITICAL_FILES:
        if os.path.exists(f):
            add(f"  {sha256(f)}  {f}  ({os.path.getsize(f)} B)")
        else:
            add(f"  {'MISSING':16s}  {f}")
    add("")

    # --- 运行时 ---
    add("[3] RUNTIME")
    add(f"python    : {sys.version.split()[0]}  ({platform.architecture()[0]})")
    add(f"os        : {platform.system()} {platform.release()}  {platform.machine()}")
    for p in CRITICAL_PKGS:
        add(f"{p:<10}: {pkg_ver(p)}")

    try:
        import torch
        add(f"cuda avail: {torch.cuda.is_available()}")
        add(f"cuda ver  : {torch.version.cuda}")
        add(f"cudnn     : {torch.backends.cudnn.version()}")
        add(f"cudnn.det : {torch.backends.cudnn.deterministic}")
        add(f"cudnn.bench:{torch.backends.cudnn.benchmark}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                add(f"gpu{i}      : {prop.name}  {prop.total_memory/2**30:.1f} GiB  cc{prop.major}.{prop.minor}")
    except Exception as e:
        add(f"torch probe failed: {e}")

    nv = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    add(f"nvidia-smi: {nv}")
    add("")

    # --- 环境变量 ---
    add("[4] ENV VARS AFFECTING NUMERICS")
    keys = ["PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
            "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "HF_HOME", "TRANSFORMERS_CACHE"]
    for k in keys:
        add(f"  {k} = {os.environ.get(k, '<unset>')}")
    add("")

    # --- 缓存数据 ---
    add("[5] CALIBRATION / TEST CACHE  (校准数据若不一致，PPL 直接不可比)")
    cache_dir = "./cache"
    if os.path.isdir(cache_dir):
        for fn in sorted(os.listdir(cache_dir)):
            p = os.path.join(cache_dir, fn)
            if os.path.isfile(p):
                sz = os.path.getsize(p) / 2**20
                add(f"  {sha256(p, limit=64*2**20)}  {fn}  ({sz:.1f} MB)")
    else:
        add("  cache dir not found")
    add("")

    # --- 运行超参（若提供）---
    add("[6] RUN CONFIG (from command line if provided)")
    add(f"  argv: {' '.join(sys.argv[1:])}")
    add("=" * 72)
    add("回传说明：把本文件与结果一起发回；若 [1]/[2]/[5] 与上一批次不同，")
    add("则该批次结果不能与上一批次并列写入同一张表。")
    add("=" * 72)

    text = "\n".join(L)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
