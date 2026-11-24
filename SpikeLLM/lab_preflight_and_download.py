"""
Lab Dual-3090 Preflight Environment Check & Model/Dataset Downloader (Windows Ready)
Usage:
  python lab_preflight_and_download.py --mode check
  python lab_preflight_and_download.py --mode download_7b --model_dir ./models
  python lab_preflight_and_download.py --mode download_8b --model_dir ./models
"""
import lab_compat  # Must precede huggingface_hub/datasets cache operations
import argparse
import os
import shutil
import sys
import urllib.request


def check_network_and_set_mirror():
    if os.environ.get("HF_ENDPOINT"):
        print(f"[NET] Using pre-configured HF_ENDPOINT = {os.environ['HF_ENDPOINT']}")
        return os.environ["HF_ENDPOINT"]
    try:
        req = urllib.request.Request("https://huggingface.co", method="HEAD")
        urllib.request.urlopen(req, timeout=5)
        print("[NET] Direct connection to https://huggingface.co is OK.")
        return "https://huggingface.co"
    except Exception:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("[NET] Direct connection to huggingface.co timed out -> Auto-switched HF_ENDPOINT to https://hf-mirror.com")
        return "https://hf-mirror.com"


def run_check():
    print("=" * 68)
    print("  SpikeLLM / OSDSQ v2 - Lab Dual-3090 Windows Preflight Check")
    print("=" * 68)
    print(f"[1] Python Executable : {sys.executable}")
    print(f"    Python Version    : {sys.version.split()[0]}")

    # Check disk space
    total, used, free = shutil.disk_usage(os.getcwd())
    free_gb = free / (1024 ** 3)
    print(f"[2] Disk Free Space   : {free_gb:.1f} GB (Current Drive: {os.path.splitdrive(os.getcwd())[0]})")
    if free_gb < 35:
        print("    [WARNING] Free disk space is under 35 GB! Each 7B/8B model needs ~15 GB.")
    else:
        print("    [OK] Disk space is sufficient for 7B/8B models and checkpoints.")

    # Check PyTorch & GPUs
    try:
        import torch
        print(f"[3] PyTorch Version   : {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
        if not torch.cuda.is_available():
            print("    [ERROR] CUDA is NOT available! Please use the GPU conda/venv environment.")
            sys.exit(1)
        n_gpus = torch.cuda.device_count()
        print(f"    Detected GPUs     : {n_gpus}")
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024 ** 3)
            print(f"      - GPU {i}: {props.name} ({vram_gb:.1f} GB VRAM)")
        if n_gpus < 2:
            print("    [NOTE] Only 1 GPU detected; script will run sequentially instead of 2-GPU parallel.")
        else:
            print("    [OK] Dual-GPU parallel mode (GPU 0: Orig SpikeLLM, GPU 1: OSDSQ v2) is READY!")
    except ImportError:
        print("[ERROR] PyTorch is not installed in this Python environment!")
        sys.exit(1)

    # Check required packages
    missing = []
    for pkg in ["transformers", "datasets", "accelerate", "lm_eval", "tqdm", "numpy"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[4] Missing Packages  : {', '.join(missing)}")
        print(f"    Please install via: pip install {' '.join(missing)}")
        sys.exit(1)
    else:
        import transformers
        print(f"[4] Dependencies      : ALL OK (transformers={transformers.__version__})")

    check_network_and_set_mirror()
    print("=" * 68)
    print("  PREFLIGHT CHECK PASSED! Ready for step 2 (download_7b / smoke).")
    print("=" * 68)


def download_model_and_data(repo_id, local_dir):
    check_network_and_set_mirror()
    from huggingface_hub import snapshot_download
    print(f"[DOWNLOAD] Downloading {repo_id} -> {local_dir} (supports resume)...")
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.bin", "*.pth", "*.pt", "original/*"],  # prefer safetensors if both exist
    )
    # Check if safetensors were downloaded; if repo only has .bin, download .bin
    files = os.listdir(local_dir)
    if not any(f.endswith(".safetensors") or f.endswith(".bin") for f in files):
        print("[DOWNLOAD] No .safetensors found, downloading .bin weights...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["original/*"],
        )
    print(f"[OK] Model ready at: {os.path.abspath(local_dir)}")

    # Pre-cache WikiText-2 and C4 validation splits so parallel workers never race on HF datasets cache
    print("[DATA] Pre-caching WikiText-2 and C4 calibration/eval datasets...")
    from datautils import get_loaders
    import torch
    os.makedirs("./cache", exist_ok=True)
    net_name = os.path.basename(local_dir.rstrip("/\\"))
    model_family = net_name.split("-")[0]
    cache_dataloader = f"./cache/dataloader_{model_family}_wikitext2_128.cache"
    cache_test_wiki = f"./cache/testloader_{model_family}_wikitext2_all.cache"
    cache_test_c4 = f"./cache/testloader_{model_family}_c4_all.cache"
    if not os.path.exists(cache_dataloader) or not os.path.exists(cache_test_wiki):
        dataloader, test_wiki = get_loaders("wikitext2", nsamples=128, seed=2, model=local_dir, seqlen=2048)
        torch.save(dataloader, cache_dataloader)
        torch.save(test_wiki, cache_test_wiki)
        print(f"[OK] Saved calibration & WikiText-2 test cache: {cache_dataloader}")
    else:
        print(f"[OK] Calibration & WikiText-2 test cache already exists: {cache_dataloader}")
    if not os.path.exists(cache_test_c4):
        _, test_c4 = get_loaders("c4", nsamples=8, seed=2, model=local_dir, seqlen=2048)
        torch.save(test_c4, cache_test_c4)
        print(f"[OK] Saved C4 test cache: {cache_test_c4}")
    print("[OK] WikiText-2 and C4 datasets are cached locally!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["check", "download_7b", "download_8b"])
    parser.add_argument("--model_dir", type=str, default="./models")
    # Default to NousResearch ungated official weight mirrors so no HF Token / Meta gate approval is needed
    parser.add_argument("--repo_7b", type=str, default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--repo_8b", type=str, default="NousResearch/Meta-Llama-3-8B")
    args = parser.parse_args()

    if args.mode == "check":
        run_check()
    elif args.mode == "download_7b":
        target = os.path.join(args.model_dir, "Llama-2-7b-hf")
        download_model_and_data(args.repo_7b, target)
    elif args.mode == "download_8b":
        target = os.path.join(args.model_dir, "Meta-Llama-3-8B")
        download_model_and_data(args.repo_8b, target)


if __name__ == "__main__":
    main()
