<#
.SYNOPSIS
  Windows Dual-RTX-3090 Runner for SpikeLLM / OSDSQ v2 (LLaMA-2-7B & LLaMA-3-8B)
.DESCRIPTION
  Designed for the Lab Windows + ToDesk Dual-3090 (2x24GB) machine.
  Modes:
    -Mode check        : Verify Python, PyTorch, 2x3090 GPUs, disk space, and network/mirror
    -Mode download_7b  : Download ungated Llama-2-7b-hf weights + WikiText2/C4 datasets
    -Mode smoke_7b     : Run 8-sample 1-epoch smoke gate on LLaMA-2-7B (Orig vs OSDSQ v2)
    -Mode 7b_parallel  : Run full LLaMA-2-7B W4A4 (GPU 0: Orig SpikeLLM, GPU 1: OSDSQ v2 in parallel)
    -Mode download_8b  : Download ungated Meta-Llama-3-8B weights + datasets
    -Mode smoke_8b     : Run 8-sample 1-epoch smoke gate on LLaMA-3-8B
    -Mode 8b_parallel  : Run full LLaMA-3-8B W4A4 (GPU 0: Orig SpikeLLM, GPU 1: OSDSQ v2 in parallel)
#>
param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("check", "download_7b", "smoke_7b", "7b_parallel", "download_8b", "smoke_8b", "8b_parallel")]
    [string]$Mode = "check",

    [string]$PythonExe = "",
    [string]$ModelDir = "./models",
    [double]$Alpha = 0.75,
    [double]$LetLr = 1e-3,
    [double]$LwcLr = 1e-2,
    [int]$Epochs = 20,
    [int]$Seed = 2
)

# Native Python stderr must not become a terminating PowerShell 5.1 error.
# Every external command below is still gated by LASTEXITCODE.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
if (-not $env:HF_ENDPOINT) { $env:HF_ENDPOINT = "https://hf-mirror.com" }
if (-not $env:HF_DATASETS_TRUST_REMOTE_CODE) { $env:HF_DATASETS_TRUST_REMOTE_CODE = "1" }

# NOTE (lab machine migration 2026-10-29): the whole project now lives under
# C:\Users\admin\Desktop\spikellm\ so that it can be removed as a single folder
# when the collaboration ends. HF_HOME is pinned to the project-local cache so
# that no huggingface state leaks outside this folder. The original
# ~\.cache\huggingface is intentionally left untouched because it holds ~50 GiB
# of unrelated lab models (Qwen2.5-7B, Mistral-7B, CLIP-H, DINOv2, Llama-3-8B-Instruct);
# only this project's 1.8 GiB worth of dataset caches was copied in.
# Kept as an override: set HF_HOME yourself before calling the script if needed.
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $PSScriptRoot "..\hf_cache" }
if (-not (Test-Path $env:HF_HOME)) {
    Write-Warning "HF_HOME points to a missing directory: $env:HF_HOME"
} else {
    $env:HF_HOME = (Resolve-Path $env:HF_HOME).Path
}

# 1. Auto-detect Python executable if not specified
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $candidates = @(
        (Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"),
        (Join-Path $PSScriptRoot ".venv\Scripts\python.exe")
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) {
            $PythonExe = (Resolve-Path $c).Path
            break
        }
    }
    if ([string]::IsNullOrWhiteSpace($PythonExe)) {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($cmd) {
            $PythonExe = $cmd.Source
        } else {
            Write-Error "Could not find python.exe! Please pass -PythonExe 'C:\path\to\python.exe'"
            exit 1
        }
    }
}
Write-Host "[ENV] Using Python: $PythonExe" -ForegroundColor Cyan
Write-Host "[ENV] Working Dir : $PSScriptRoot" -ForegroundColor Cyan

function Ensure-ActStats {
    param([string]$ModelPath, [string]$NetName, [int]$NumSamples = 128)
    $scalePath = "./act_scales/$NetName.pt"
    $shiftPath = "./act_shifts/$NetName.pt"
    if ((-not (Test-Path $scalePath)) -or (-not (Test-Path $shiftPath))) {
        Write-Host "[ACT_STATS] Generating $scalePath and $shiftPath (samples=$NumSamples)..." -ForegroundColor Yellow
        & $PythonExe generate_act_scale_shift.py --model $ModelPath --num-samples $NumSamples --seed $Seed
        if ($LASTEXITCODE -ne 0) { throw "generate_act_scale_shift.py failed!" }
    } else {
        Write-Host "[ACT_STATS] Found existing $scalePath and $shiftPath" -ForegroundColor Green
    }
}

function Run-Smoke {
    param([string]$ModelPath, [string]$TagPrefix)
    $NetName = Split-Path $ModelPath -Leaf
    # Use the same 128-sample LET statistics as the formal run. Otherwise the
    # smoke cache would be silently reused by Run-FullParallel.
    Ensure-ActStats -ModelPath $ModelPath -NetName $NetName -NumSamples 128

    $origDir = "./log/${TagPrefix}_smoke_orig"
    $oursDir = "./log/${TagPrefix}_smoke_osdsq_v2"
    New-Item -ItemType Directory -Force -Path $origDir | Out-Null
    New-Item -ItemType Directory -Force -Path $oursDir | Out-Null

    Write-Host "=== [SMOKE 1/2] Running Original SpikeLLM (nsamples=8, epochs=1) ===" -ForegroundColor Yellow
    $env:CUDA_VISIBLE_DEVICES = "0"
    & $PythonExe main.py `
        --model $ModelPath --net $NetName `
        --wbits 4 --abits 4 --addbit 1 --low_p 0.9 `
        --let --lwc --aug_loss `
        --alpha $Alpha --let_lr $LetLr --lwc_lr $LwcLr `
        --nsamples 8 --epochs 1 --seed $Seed `
        --orig_spikellm --eval_ppl `
        --output_dir $origDir 2>&1 | Tee-Object -FilePath "$origDir/smoke.log"
    if ($LASTEXITCODE -ne 0) { throw "Smoke test (Original SpikeLLM) failed!" }

    Write-Host "=== [SMOKE 2/2] Running Ours: OSDSQ v2 (nsamples=8, epochs=1) ===" -ForegroundColor Yellow
    & $PythonExe main.py `
        --model $ModelPath --net $NetName `
        --wbits 4 --abits 4 --addbit 1 --low_p 0.9 `
        --let --lwc --aug_loss `
        --alpha $Alpha --let_lr $LetLr --lwc_lr $LwcLr `
        --nsamples 8 --epochs 1 --seed $Seed `
        --eval_ppl `
        --output_dir $oursDir 2>&1 | Tee-Object -FilePath "$oursDir/smoke.log"
    if ($LASTEXITCODE -ne 0) { throw "Smoke test (OSDSQ v2) failed!" }

    # Gate the exact resume/evaluation paths used by the formal suite as well.
    foreach ($config in @(
        @{ Dir = $origDir; Original = $true; Name = "Original SpikeLLM" },
        @{ Dir = $oursDir; Original = $false; Name = "OSDSQ v2" }
    )) {
        $checkpoint = Join-Path $config.Dir "omni_parameters.pth"
        if (-not (Test-Path $checkpoint)) { throw "Missing smoke checkpoint: $checkpoint" }

        Write-Host "=== [SMOKE EVAL] $($config.Name): resume + real multibit PPL ===" -ForegroundColor Yellow
        $realArgs = @(
            "main.py", "--model", $ModelPath, "--net", $NetName,
            "--wbits", "4", "--abits", "4", "--addbit", "1", "--low_p", "0.9",
            "--let", "--lwc", "--aug_loss", "--alpha", "$Alpha", "--let_lr", "$LetLr", "--lwc_lr", "$LwcLr",
            "--epochs", "0", "--resume", $checkpoint,
            "--sim_mode", "real_multibit_simulate", "--eval_ppl",
            "--output_dir", (Join-Path $config.Dir "real_multibit")
        )
        if ($config.Original) { $realArgs += "--orig_spikellm" }
        & $PythonExe @realArgs 2>&1 | Tee-Object -FilePath (Join-Path $config.Dir "smoke_real.log")
        if ($LASTEXITCODE -ne 0) { throw "Smoke real-multibit evaluation failed: $($config.Name)" }

        Write-Host "=== [SMOKE EVAL] $($config.Name): resume + PIQA ===" -ForegroundColor Yellow
        $taskArgs = @(
            "main.py", "--model", $ModelPath, "--net", $NetName,
            "--wbits", "4", "--abits", "4", "--addbit", "1", "--low_p", "0.9",
            "--let", "--lwc", "--aug_loss", "--alpha", "$Alpha", "--let_lr", "$LetLr", "--lwc_lr", "$LwcLr",
            "--epochs", "0", "--resume", $checkpoint,
            "--tasks", "piqa", "--output_dir", (Join-Path $config.Dir "zeroshot")
        )
        if ($config.Original) { $taskArgs += "--orig_spikellm" }
        & $PythonExe @taskArgs 2>&1 | Tee-Object -FilePath (Join-Path $config.Dir "smoke_piqa.log")
        if ($LASTEXITCODE -ne 0) { throw "Smoke PIQA evaluation failed: $($config.Name)" }
    }

    Write-Host "================================================================" -ForegroundColor Green
    Write-Host "  SMOKE GATE PASSED for $NetName!" -ForegroundColor Green
    Write-Host "  Logs saved in: $origDir/smoke.log and $oursDir/smoke.log" -ForegroundColor Green
    Write-Host "================================================================" -ForegroundColor Green
}

function Run-FullParallel {
    param([string]$ModelPath, [string]$TagPrefix)
    $NetName = Split-Path $ModelPath -Leaf
    Ensure-ActStats -ModelPath $ModelPath -NetName $NetName -NumSamples 128

    # 1. FP16 Baseline (skipped if done.flag exists)
    $fp16Dir = "./log/${TagPrefix}_fp16"
    New-Item -ItemType Directory -Force -Path $fp16Dir | Out-Null
    if (-not (Test-Path "$fp16Dir/done.flag")) {
        Write-Host "=== [STEP 1] Evaluating FP16 Baseline for $NetName ===" -ForegroundColor Cyan
        $env:CUDA_VISIBLE_DEVICES = "0"
        & $PythonExe main.py `
            --model $ModelPath --net $NetName `
            --wbits 16 --abits 16 --epochs 0 --eval_ppl `
            --output_dir $fp16Dir 2>&1 | Tee-Object -FilePath "$fp16Dir/run.log"
        if ($LASTEXITCODE -ne 0) { throw "FP16 baseline evaluation failed for $NetName" }
        New-Item -ItemType File -Force -Path "$fp16Dir/done.flag" | Out-Null
    } else {
        Write-Host "=== [STEP 1] FP16 Baseline already completed ($fp16Dir/done.flag) ===" -ForegroundColor Green
    }

    # 2. Dual-3090 Parallel Execution: GPU 0 = Original SpikeLLM (+LET), GPU 1 = Ours OSDSQ v2 (+LET)
    $origDir = "./log/${TagPrefix}_orig_with_let"
    $oursDir = "./log/${TagPrefix}_osdsq_v2_with_let"
    New-Item -ItemType Directory -Force -Path $origDir | Out-Null
    New-Item -ItemType Directory -Force -Path $oursDir | Out-Null

    Write-Host "=== [STEP 2] Launching Dual-3090 Parallel W4A4 Training + Multibit Eval ===" -ForegroundColor Cyan
    Write-Host "  -> GPU 0: Original SpikeLLM (+LET)  [Log: $origDir/run.log]" -ForegroundColor Yellow
    Write-Host "  -> GPU 1: Ours OSDSQ v2 (+LET)      [Log: $oursDir/run.log]" -ForegroundColor Yellow

    $jobOrig = Start-Job -ScriptBlock {
        param($wd, $py, $model, $net, $alpha, $letlr, $lwclr, $epochs, $seed, $outDir)
        Set-Location $wd
        $env:CUDA_VISIBLE_DEVICES = "0"
        if (-not (Test-Path "$outDir/done_ppl.flag")) {
            & $py main.py --model $model --net $net `
                --wbits 4 --abits 4 --addbit 1 --low_p 0.9 `
                --let --lwc --aug_loss `
                --alpha $alpha --let_lr $letlr --lwc_lr $lwclr `
                --nsamples 128 --epochs $epochs --seed $seed `
                --orig_spikellm --eval_ppl `
                --output_dir $outDir *>> "$outDir/run.log"
            if ($LASTEXITCODE -ne 0) { throw "GPU 0 Original SpikeLLM main PPL run failed" }
            New-Item -ItemType File -Force -Path "$outDir/done_ppl.flag" | Out-Null
        }
        if (-not (Test-Path "$outDir/done_multibit.flag")) {
            & $py main.py --model $model --net $net `
                --wbits 4 --abits 4 --addbit 1 --low_p 0.9 `
                --let --lwc --aug_loss --alpha $alpha --let_lr $letlr --lwc_lr $lwclr `
                --epochs 0 --resume "$outDir/omni_parameters.pth" `
                --orig_spikellm --sim_mode real_multibit_simulate --eval_ppl `
                --output_dir "$outDir/real_multibit" *>> "$outDir/run.log"
            if ($LASTEXITCODE -ne 0) { throw "GPU 0 Original SpikeLLM real_multibit run failed" }
            New-Item -ItemType File -Force -Path "$outDir/done_multibit.flag" | Out-Null
        }
        if (-not (Test-Path "$outDir/done.flag")) {
            & $py main.py --model $model --net $net `
                --wbits 4 --abits 4 --addbit 1 --low_p 0.9 `
                --let --lwc --aug_loss --alpha $alpha --let_lr $letlr --lwc_lr $lwclr `
                --epochs 0 --resume "$outDir/omni_parameters.pth" `
                --orig_spikellm --tasks piqa,arc_easy,arc_challenge,boolq,winogrande,hellaswag `
                --output_dir "$outDir/zeroshot" *>> "$outDir/run.log"
            if ($LASTEXITCODE -eq 0) { New-Item -ItemType File -Force -Path "$outDir/done.flag" | Out-Null }
        }
    } -ArgumentList $PSScriptRoot, $PythonExe, $ModelPath, $NetName, $Alpha, $LetLr, $LwcLr, $Epochs, $Seed, $origDir

    $jobOurs = Start-Job -ScriptBlock {
        param($wd, $py, $model, $net, $alpha, $letlr, $lwclr, $epochs, $seed, $outDir)
        Set-Location $wd
        $env:CUDA_VISIBLE_DEVICES = "1"
        if (-not (Test-Path "$outDir/done_ppl.flag")) {
            & $py main.py --model $model --net $net `
                --wbits 4 --abits 4 --addbit 1 --low_p 0.9 `
                --let --lwc --aug_loss `
                --alpha $alpha --let_lr $letlr --lwc_lr $lwclr `
                --nsamples 128 --epochs $epochs --seed $seed `
                --eval_ppl `
                --output_dir $outDir *>> "$outDir/run.log"
            if ($LASTEXITCODE -ne 0) { throw "GPU 1 OSDSQ v2 main PPL run failed" }
            New-Item -ItemType File -Force -Path "$outDir/done_ppl.flag" | Out-Null
        }
        if (-not (Test-Path "$outDir/done_multibit.flag")) {
            & $py main.py --model $model --net $net `
                --wbits 4 --abits 4 --addbit 1 --low_p 0.9 `
                --let --lwc --aug_loss --alpha $alpha --let_lr $letlr --lwc_lr $lwclr `
                --epochs 0 --resume "$outDir/omni_parameters.pth" `
                --sim_mode real_multibit_simulate --eval_ppl `
                --output_dir "$outDir/real_multibit" *>> "$outDir/run.log"
            if ($LASTEXITCODE -ne 0) { throw "GPU 1 OSDSQ v2 real_multibit run failed" }
            New-Item -ItemType File -Force -Path "$outDir/done_multibit.flag" | Out-Null
        }
        if (-not (Test-Path "$outDir/done.flag")) {
            & $py main.py --model $model --net $net `
                --wbits 4 --abits 4 --addbit 1 --low_p 0.9 `
                --let --lwc --aug_loss --alpha $alpha --let_lr $letlr --lwc_lr $lwclr `
                --epochs 0 --resume "$outDir/omni_parameters.pth" `
                --tasks piqa,arc_easy,arc_challenge,boolq,winogrande,hellaswag `
                --output_dir "$outDir/zeroshot" *>> "$outDir/run.log"
            if ($LASTEXITCODE -eq 0) { New-Item -ItemType File -Force -Path "$outDir/done.flag" | Out-Null }
        }
    } -ArgumentList $PSScriptRoot, $PythonExe, $ModelPath, $NetName, $Alpha, $LetLr, $LwcLr, $Epochs, $Seed, $oursDir

    while (($jobOrig.State -eq "Running") -or ($jobOurs.State -eq "Running")) {
        $t = Get-Date -Format "HH:mm:ss"
        $origTail = if (Test-Path "$origDir/run.log") { (Get-Content "$origDir/run.log" -Tail 1) } else { "waiting..." }
        $oursTail = if (Test-Path "$oursDir/run.log") { (Get-Content "$oursDir/run.log" -Tail 1) } else { "waiting..." }
        Write-Host "[$t] GPU0(Orig): $origTail" -ForegroundColor Gray
        Write-Host "[$t] GPU1(Ours): $oursTail" -ForegroundColor Green
        Start-Sleep -Seconds 30
    }

    Receive-Job -Job $jobOrig
    Receive-Job -Job $jobOurs
    if (($jobOrig.State -ne "Completed") -or ($jobOurs.State -ne "Completed")) {
        throw "Parallel run failed: Original=$($jobOrig.State), OSDSQ=$($jobOurs.State). Inspect both run.log files."
    }
    $requiredFlags = @(
        "$origDir/done_ppl.flag", "$origDir/done_multibit.flag", "$origDir/done.flag",
        "$oursDir/done_ppl.flag", "$oursDir/done_multibit.flag", "$oursDir/done.flag"
    )
    $missingFlags = $requiredFlags | Where-Object { -not (Test-Path $_) }
    if ($missingFlags) {
        throw "Parallel run is incomplete; missing flags: $($missingFlags -join ', ')"
    }
    Write-Host "=== Completed Dual-3090 Parallel Run for $NetName! ===" -ForegroundColor Green
}

switch ($Mode) {
    "check" {
        & $PythonExe lab_preflight_and_download.py --mode check
    }
    "download_7b" {
        & $PythonExe lab_preflight_and_download.py --mode download_7b --model_dir $ModelDir
    }
    "smoke_7b" {
        Run-Smoke -ModelPath "$ModelDir/Llama-2-7b-hf" -TagPrefix "7b"
    }
    "7b_parallel" {
        Run-FullParallel -ModelPath "$ModelDir/Llama-2-7b-hf" -TagPrefix "7b"
    }
    "download_8b" {
        & $PythonExe lab_preflight_and_download.py --mode download_8b --model_dir $ModelDir
    }
    "smoke_8b" {
        Run-Smoke -ModelPath "$ModelDir/Meta-Llama-3-8B" -TagPrefix "8b"
    }
    "8b_parallel" {
        Run-FullParallel -ModelPath "$ModelDir/Meta-Llama-3-8B" -TagPrefix "8b"
    }
}
