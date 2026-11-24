# Watchdog for the long 7b_parallel run (lab machine, 2026-10-28).
# Appends a compact snapshot every 10 minutes to .\lab_watch.log so that a
# multi-hour / multi-day unattended run leaves a readable trail:
#   - current calibration layer per arm
#   - cumulative count of `norm:nan` lines (AMP transient overflow counter)
#   - last (layer, iter, loss, norm) tuple
#   - GPU memory / utilisation
# It never touches the training processes; it only reads logs and nvidia-smi.
#
# NOTE (2026-10-29): parameterised so it can watch any pair of arms / any log file.
#   -Arms "p0_noaug_orig","p0_noaug_osdsq_v2" -LogName lab_watch_p0.log -Hours 40
param(
    [string[]]$Arms = @("7b_orig_with_let", "7b_osdsq_v2_with_let"),
    [string]$LogName = "lab_watch.log",
    [double]$Hours = 30
)

$ErrorActionPreference = "Continue"

# NOTE (2026-10-28): use an absolute base directory. `Set-Location` only changes the
# PowerShell provider location; .NET APIs such as [System.IO.File]::ReadAllText resolve
# relative paths against the process working directory, which stays at the caller's cwd
# (this bit us once: ReadAllText looked for the log under Desktop\spikellm and silently
# produced "layer=- epochs_logged=0" rows). Same reason every path below is absolute.
$Base = $PSScriptRoot
Set-Location $Base

$log    = Join-Path $Base $LogName
$deadline = (Get-Date).AddHours($Hours)
# NOTE (2026-10-29): `powershell -File script.ps1 -Arms "a","b"` does NOT produce a
# two-element array -- -File arguments arrive as raw strings, so the whole "a","b"
# lands in $Arms[0] as one blob. Split on commas here so both call styles work.
$arms = @(
    $Arms | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ }
)
if ($arms.Count -eq 0) { throw "watchdog: no arm names given" }

function Read-SharedText {
    # The training processes write run.log through PowerShell's `*>>` redirection, which
    # holds the file open with a *write* handle. [System.IO.File]::ReadAllText and
    # Get-Content then fail with "being used by another process", and because
    # $ErrorActionPreference is Continue the failure is silent -> every field comes back
    # empty (layer=- , epochs_logged=0). Opening with FileShare::ReadWrite fixes it.
    param([string]$Path)

    $text = ""
    try {
        $fs = [System.IO.File]::Open(
            $Path,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read,
            [System.IO.FileShare]::ReadWrite
        )
        $sr = New-Object System.IO.StreamReader($fs, [System.Text.Encoding]::UTF8, $true)
        $text = $sr.ReadToEnd()
        $sr.Dispose()          # also closes $fs
    } catch {
        return ""
    }

    if ($text.IndexOf([char]0) -ge 0) {
        # UTF-16LE payload without a BOM: re-decode properly.
        try {
            $fs2 = [System.IO.File]::Open(
                $Path,
                [System.IO.FileMode]::Open,
                [System.IO.FileAccess]::Read,
                [System.IO.FileShare]::ReadWrite
            )
            $sr2 = New-Object System.IO.StreamReader($fs2, [System.Text.Encoding]::Unicode, $false)
            $text = $sr2.ReadToEnd()
            $sr2.Dispose()
        } catch {
            return ""
        }
    }
    return $text
}

function Get-Snapshot {
    param([string]$Tag)

    $out = @()
    $out += "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Tag"

    foreach ($arm in $arms) {
        $p = Join-Path $Base "log\$arm\run.log"
        if (-not (Test-Path $p)) { $out += "  ${arm}: (no log yet)"; continue }

        $raw = Read-SharedText -Path $p
        if ([string]::IsNullOrEmpty($raw)) {
            $out += "  ${arm}: (log unreadable right now, size=$((Get-Item $p).Length))"
            continue
        }

        $layerHits = [regex]::Matches($raw, "Start quantize layer (\d+)")
        $curLayer  = if ($layerHits.Count -gt 0) { $layerHits[$layerHits.Count - 1].Groups[1].Value } else { "-" }

        $iters = [regex]::Matches($raw, "layer (\d+) iter (\d+) loss:(\S+) norm:(\S+)")
        $lastIter = if ($iters.Count -gt 0) { $iters[$iters.Count - 1].Groups[0].Value } else { "(none yet)" }

        $nanCount = ([regex]::Matches($raw, "norm:nan")).Count
        $infCount = ([regex]::Matches($raw, "norm:inf")).Count

        $doneFlags = @()
        foreach ($f in @("done_ppl.flag", "done_multibit.flag", "done.flag")) {
            if (Test-Path (Join-Path $Base "log\$arm\$f")) { $doneFlags += $f }
        }

        $out += "  ${arm}: layer=$curLayer epochs_logged=$($iters.Count) norm_nan=$nanCount norm_inf=$infCount"
        $out += "    last: $lastIter"
        if ($doneFlags.Count -gt 0) { $out += "    flags: $($doneFlags -join ',')" }
    }

    $gpu = (nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader) -join " || "
    $out += "  GPU: $gpu"

    $out | Out-File -FilePath $log -Append -Encoding utf8
}

Out-File -FilePath $log -Append -Encoding utf8 -InputObject "===== watchdog started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="

while ((Get-Date) -lt $deadline) {

    $py = @(Get-Process -Name python -ErrorAction SilentlyContinue)
    Get-Snapshot -Tag "python_procs=$($py.Count)"

    $origDone = Test-Path (Join-Path $Base "log\$($arms[0])\done.flag")
    $oursDone = Test-Path (Join-Path $Base "log\$($arms[-1])\done.flag")

    if ($py.Count -eq 0 -and (-not $origDone -or -not $oursDone)) {
        Get-Snapshot -Tag "FATAL: no python process while done.flag is missing (orig=$origDone ours=$oursDone)"
        Out-File -FilePath $log -Append -Encoding utf8 -InputObject "===== watchdog stopped: abnormal end ====="
        break
    }

    if ($origDone -and $oursDone) {
        Get-Snapshot -Tag "COMPLETE: both arms finished"
        Out-File -FilePath $log -Append -Encoding utf8 -InputObject "===== watchdog stopped: all done ====="
        break
    }

    Start-Sleep -Seconds 600
}
