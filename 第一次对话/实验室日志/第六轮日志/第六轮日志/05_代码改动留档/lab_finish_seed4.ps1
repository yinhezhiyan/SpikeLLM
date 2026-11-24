<#
.SYNOPSIS
    Finish the remaining steps of 7b_seed4_orig (real_multibit + done flags).

.DESCRIPTION
    Context (lab machine, 2026-11-01 10:20):
    The GPU1 leg of lab_round6_runner.ps1 died right after "get_wikitext2" on its first
    launch at 10:12 -- no Python traceback, GPU never allocated, log simply stops, so the
    parent runner recorded that leg as failed. The identical argv was then re-run by hand
    on GPU1 and it passed data loading and reached layer 0 normally.

    Therefore this round is completed as:
      1) fake PPL already running by hand  -> ./log/7b_seed4_orig/run.log
      2) this script waits for it, then runs real_multibit
      3) done_ppl / done_multibit / done.flag are written only if both stages succeeded

    After that, re-running `-Mode run` sees all flags present and only writes the result
    file, so the 8h PPL stage is never repeated.

    NOTE: does not change any numerical path; it only replays step 2 and 3 of the runner
    body (Invoke-TrainBody).
#>
param(
    [string]$Tag = "7b_seed4_orig",
    [string]$PythonExe = "C:\Users\admin\.conda\envs\spikellm\python.exe",
    [int]$Gpu = 1,
    [double]$TimeoutHours = 14
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path (Split-Path $PSScriptRoot -Parent) "hf_cache" }
if (-not $env:HF_ENDPOINT) { $env:HF_ENDPOINT = "https://hf-mirror.com" }
if (-not $env:HF_DATASETS_TRUST_REMOTE_CODE) { $env:HF_DATASETS_TRUST_REMOTE_CODE = "1" }
$env:CUDA_VISIBLE_DEVICES = "$Gpu"

$dir = "./log/$Tag"
$report = "$dir/finish_seed4.log"

function Write-Log([string]$m) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
    Add-Content -Path $report -Value $line -Encoding utf8
}

# The training process holds run.log / log_rank0_*.txt with a write handle (PowerShell
# redirection), so plain ReadAllText fails; open with FileShare ReadWrite instead.
function Read-Shared([string]$path) {
    try {
        $fs = [System.IO.File]::Open($path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
        $sr = New-Object System.IO.StreamReader($fs)
        $t = $sr.ReadToEnd()
        $sr.Close()
        $fs.Close()
        return $t
    } catch { return "" }
}

Write-Log ("waiting for fake PPL of " + $Tag + " (timeout " + $TimeoutHours + "h)")

$deadline = (Get-Date).AddHours($TimeoutHours)
$done = $false
while ((Get-Date) -lt $deadline) {
    $files = @(Get-ChildItem "$dir/log_rank0_*.txt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)
    foreach ($f in $files) {
        $t = Read-Shared $f.FullName
        if (($t -match "wikitext2 :") -and ($t -match "c4 :")) { $done = $true; break }
    }
    if ($done) { break }
    Start-Sleep -Seconds 60
}

if (-not $done) {
    Write-Log "TIMEOUT: fake PPL did not finish in time, aborting"
    exit 2
}

Write-Log "fake PPL finished, waiting 45s for process exit"
Start-Sleep -Seconds 45

if (-not (Test-Path "$dir/omni_parameters.pth")) {
    Write-Log "FAIL: omni_parameters.pth missing, will not write flags"
    exit 3
}

$base = @(
    "main.py",
    "--model", (Join-Path $PSScriptRoot "models\Llama-2-7b-hf"),
    "--net", "Llama-2-7b-hf",
    "--wbits", "4", "--abits", "4", "--addbit", "1", "--low_p", "0.9",
    "--let", "--lwc", "--aug_loss",
    "--alpha", "0.75", "--let_lr", "0.001", "--lwc_lr", "0.01",
    "--nsamples", "128", "--epochs", "20", "--seed", "4",
    "--cache_dir", "./cache_seed4", "--orig_spikellm"
)

if (-not (Test-Path "$dir/done_multibit.flag")) {
    Write-Log "start real_multibit"
    $argv2 = $base + @(
        "--epochs", "0", "--resume", "$dir/omni_parameters.pth",
        "--sim_mode", "real_multibit_simulate", "--eval_ppl",
        "--output_dir", "$dir/real_multibit"
    )
    & $PythonExe @argv2 *>> "$dir/run.log"
    if ($LASTEXITCODE -ne 0) {
        Write-Log ("FAIL: real_multibit exit=" + $LASTEXITCODE + ", will not write flags")
        exit 4
    }
    New-Item -ItemType File -Force -Path "$dir/done_multibit.flag" | Out-Null
    Write-Log "real_multibit done"
}

New-Item -ItemType File -Force -Path "$dir/done_ppl.flag" | Out-Null
New-Item -ItemType File -Force -Path "$dir/done.flag" | Out-Null
Write-Log "flags written: done_ppl.flag, done_multibit.flag, done.flag"
