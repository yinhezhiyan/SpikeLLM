<#
.SYNOPSIS
  P0 protocol replication: Original SpikeLLM vs OSDSQ v2, WITHOUT --aug_loss.
.DESCRIPTION
  Mirrors run_lab_dual3090_7b_8b.ps1 -Mode 7b_parallel exactly, except that the
  `--aug_loss` flag is omitted from every stage (the paper's protocol-replication
  arm). Everything else is identical to the formal run:
    wbits=4 abits=4 addbit=1 low_p=0.9, --let --lwc, alpha=0.75,
    let_lr=1e-3, lwc_lr=1e-2, nsamples=128, epochs=20, seed=2,
    calibration subset reused from ./cache (same as the formal run),
    same three gates per arm: fake_binary PPL -> real_multibit PPL -> 6 zero-shot.
  The FP16 baseline is NOT re-run: log/7b_fp16 already holds 5.4722 / 6.9727 and
  --aug_loss only affects the W4A4 calibration loss, not the FP16 forward pass.

  Outputs:
    log/p0_noaug_orig        (GPU 0, Original SpikeLLM)
    log/p0_noaug_osdsq_v2    (GPU 1, OSDSQ v2)
  Each gate writes a done_*.flag so a re-run resumes instead of recomputing.

  NOTE (lab machine migration 2026-10-29): nothing here changes the quantization
  algorithm; only the loss-augmentation flag differs from the formal run.
#>
param(
    [string]$PythonExe = "C:\Users\admin\.conda\envs\spikellm\python.exe",
    [string]$ModelDir  = "./models",
    [string]$TagPrefix = "p0_noaug",
    [double]$Alpha     = 0.75,
    [double]$LetLr     = 1e-3,
    [double]$LwcLr     = 1e-2,
    [int]$Epochs       = 20,
    [int]$Seed         = 2
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $PSScriptRoot "..\hf_cache" }
if (-not $env:HF_ENDPOINT) { $env:HF_ENDPOINT = "https://hf-mirror.com" }
if (-not $env:HF_DATASETS_TRUST_REMOTE_CODE) { $env:HF_DATASETS_TRUST_REMOTE_CODE = "1" }
$env:HF_HOME = (Resolve-Path $env:HF_HOME).Path

$ModelPath = (Resolve-Path (Join-Path $ModelDir "Llama-2-7b-hf")).Path
$NetName   = Split-Path $ModelPath -Leaf

$origDir = Join-Path $PSScriptRoot "log\${TagPrefix}_orig"
$oursDir = Join-Path $PSScriptRoot "log\${TagPrefix}_osdsq_v2"
New-Item -ItemType Directory -Force -Path $origDir, $oursDir | Out-Null

Write-Host "=== P0 protocol replication (NO --aug_loss) ===" -ForegroundColor Cyan
Write-Host "  model   : $ModelPath"
Write-Host "  HF_HOME : $env:HF_HOME"
Write-Host "  python  : $PythonExe"
Write-Host "  GPU 0   : Original SpikeLLM  -> $origDir"
Write-Host "  GPU 1   : OSDSQ v2           -> $oursDir"
Write-Host "  hyper   : alpha=$Alpha let_lr=$LetLr lwc_lr=$LwcLr nsamples=128 epochs=$Epochs seed=$Seed"

$body = {
    param($cfg)
    Set-Location $cfg.wd
    $env:HF_HOME = $cfg.hf_home
    $env:HF_ENDPOINT = $cfg.hf_endpoint
    $env:HF_DATASETS_TRUST_REMOTE_CODE = "1"
    $env:CUDA_VISIBLE_DEVICES = $cfg.gpu
    $py = $cfg.py
    $outDir = $cfg.outDir

    # Base argv shared by all three gates. NOTE: no --aug_loss on purpose.
    $base_ = @(
        "main.py",
        "--model", $cfg.model,
        "--net", $cfg.net,
        "--wbits", "4", "--abits", "4", "--addbit", "1", "--low_p", "0.9",
        "--let", "--lwc",
        "--alpha", "$($cfg.alpha)",
        "--let_lr", "$($cfg.letlr)",
        "--lwc_lr", "$($cfg.lwclr)"
    )
    $armArgs = @()
    if ($cfg.orig) { $armArgs += "--orig_spikellm" }

    # Gate 1: calibration + fake_binary PPL
    if (-not (Test-Path "$outDir/done_ppl.flag")) {
        $argv = $base_ + @(
            "--nsamples", "128", "--epochs", "$($cfg.epochs)", "--seed", "$($cfg.seed)",
            "--eval_ppl", "--output_dir", $outDir
        ) + $armArgs
        & $py @argv *>> "$outDir/run.log"
        if ($LASTEXITCODE -ne 0) { throw "GATE1 failed for $outDir" }
        New-Item -ItemType File -Force -Path "$outDir/done_ppl.flag" | Out-Null
    }

    # Gate 2: real_multibit PPL from the checkpoint
    if (-not (Test-Path "$outDir/done_multibit.flag")) {
        $argv = $base_ + @(
            "--epochs", "0", "--resume", "$outDir/omni_parameters.pth",
            "--sim_mode", "real_multibit_simulate", "--eval_ppl",
            "--output_dir", "$outDir/real_multibit"
        ) + $armArgs
        & $py @argv *>> "$outDir/run.log"
        if ($LASTEXITCODE -ne 0) { throw "GATE2 failed for $outDir" }
        New-Item -ItemType File -Force -Path "$outDir/done_multibit.flag" | Out-Null
    }

    # Gate 3: six zero-shot tasks
    if (-not (Test-Path "$outDir/done.flag")) {
        $argv = $base_ + @(
            "--epochs", "0", "--resume", "$outDir/omni_parameters.pth",
            "--tasks", "piqa,arc_easy,arc_challenge,boolq,winogrande,hellaswag",
            "--output_dir", "$outDir/zeroshot"
        ) + $armArgs
        & $py @argv *>> "$outDir/run.log"
        if ($LASTEXITCODE -eq 0) { New-Item -ItemType File -Force -Path "$outDir/done.flag" | Out-Null }
    }
    "ARM DONE: $outDir"
}

function New-ArmCfg {
    param([string]$Gpu, [string]$OutDir, [bool]$Orig)
    return @{
        wd = $PSScriptRoot; py = $PythonExe; model = $ModelPath; net = $NetName
        alpha = $Alpha; letlr = $LetLr; lwclr = $LwcLr; epochs = $Epochs; seed = $Seed
        hf_home = $env:HF_HOME; hf_endpoint = $env:HF_ENDPOINT
        gpu = $Gpu; outDir = $OutDir; orig = $Orig
    }
}

$jobOrig = Start-Job -ScriptBlock $body -ArgumentList (New-ArmCfg -Gpu "0" -OutDir $origDir -Orig $true)
$jobOurs = Start-Job -ScriptBlock $body -ArgumentList (New-ArmCfg -Gpu "1" -OutDir $oursDir -Orig $false)

Write-Host "  job ids : orig=$($jobOrig.Id)  ours=$($jobOurs.Id)" -ForegroundColor DarkGray
Write-Host "  START   : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor DarkGray

Wait-Job $jobOrig, $jobOurs | Out-Null
foreach ($j in @($jobOrig, $jobOurs)) {
    Write-Host "--- job $($j.Id) state=$($j.State) ---"
    Receive-Job $j 2>&1 | Select-Object -Last 15
}
Remove-Job $jobOrig, $jobOurs -Force

$okOrig = Test-Path "$origDir/done.flag"
$okOurs = Test-Path "$oursDir/done.flag"
Write-Host "=== P0 finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  orig=$okOrig ours=$okOurs ===" -ForegroundColor Cyan
if (-not ($okOrig -and $okOurs)) { exit 1 }
