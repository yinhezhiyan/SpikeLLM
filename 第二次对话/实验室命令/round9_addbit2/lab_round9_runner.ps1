<#
================================================================================
第九轮实验执行器（实验室 Windows 双 RTX 3090）· 兼本轮说明文档
生成：WorkBuddy  2026-11-07
自定位：所有路径由 $PSScriptRoot 推导，脚本放在 SpikeLLM 根目录即可运行。

--------------------------------------------------------------------------------
【一、上一轮（第八轮）发生了什么】

第八轮把 addbit 从 1 提到 3（高显著分支 4->7 bit，KV 的 Refine 锁定 5 bit 不变）：
  7B  Orig   10.5726 -> 10.1386  (-4.10%)
  7B  Ours    7.9792 ->  7.8978  (-1.02%)
  同配对降幅：T=2 为 24.53%，a3 为 22.10%（>5%，方法在高预算档仍有效）
结论：给显著通道加预算，边际收益只有 4%；修复统计域（我们的方法）拿 24%。
"为什么不直接调大 addbit" 这一问已经有硬答案，但前沿曲线只有两个点。

--------------------------------------------------------------------------------
【二、本轮任务】

  门禁（约 40 分钟，串行 GPU0，全部 68M）
    r9_68m_orig_a1   68M  --orig_spikellm --addbit 1   <- 实验室机上从未跑过 68M 的
                                                          Orig 基线，补上它，统一口径
    r9_68m_orig_a2   68M  --orig_spikellm --addbit 2   <- 68M 前沿中点
    r9_68m_ours_a2   68M                --addbit 2   <- 68M 前沿中点

  主实验（约 7~8 小时，双卡并行）
    r9_7b_orig_a2    7B   --orig_spikellm --addbit 2    (GPU0)
    r9_7b_ours_a2    7B                 --addbit 2    (GPU1)

  只测 fake PPL（WikiText-2 + C4），零样本关闭，不跑 real_multibit。
  除 --addbit 外，参数与第七、八轮逐字相同。

  预期（写在这里仅供对表，不是判据）：7B Orig@a2 应落在 10.14~10.57 之间，
  7B Ours@a2 应落在 7.90~7.98 之间。若出现非单调（比如 a2 比 a1 和 a3 都差），
  原样回传，不要自行解释。

--------------------------------------------------------------------------------
【三、怎么跑】

  1. 本脚本放到 SpikeLLM 根目录（和 main.py 同级）。
  2. powershell -ExecutionPolicy Bypass -File .\lab_round9_runner.ps1 -Mode preflight
     powershell -ExecutionPolicy Bypass -File .\lab_round9_runner.ps1 -Mode run
  3. 可断线，重跑 -Mode run 自动跳过已完成的腿。
  4. 结束后回传：lab_round9_result.txt + log\r9_*\log_rank0_*.txt +
     env_round9_preflight.txt。

  env_report 指纹缺失时结果文件会标记【不可入账】，请务必确认它在。
================================================================================
#>

param(
    [ValidateSet("preflight", "run", "gate", "status", "runleg")]
    [string]$Mode = "status",
    [string]$Tag = "",
    [string]$Py = "",
    [string]$ModelDir = ""
)

$ErrorActionPreference = "Continue"
$Root = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Root)) { $Root = (Get-Location).Path }
Set-Location $Root

if ([string]::IsNullOrWhiteSpace($ModelDir)) { $ModelDir = Join-Path $Root "models" }
if ([string]::IsNullOrWhiteSpace($Py)) {
    $cands = @(
        (Join-Path $Root ".venv\Scripts\python.exe"),
        (Join-Path (Split-Path $Root -Parent) ".venv\Scripts\python.exe"),
        (Join-Path $Root "venv\Scripts\python.exe")
    )
    $Py = "python"
    foreach ($c in $cands) { if (Test-Path $c) { $Py = $c; break } }
}

$Net7B    = "Llama-2-7b-hf"
$Model7B  = Join-Path $ModelDir $Net7B
$Net68M   = "llama-68m"
$Model68M = "JackFram/llama-68m"

$Jobs = @(
    @{ Tag = "r9_68m_orig_a1"; Model = $Model68M; Net = $Net68M; Seed = 2; Cache = "./cache"; Addbit = 1;
       Extra = "--orig_spikellm"; Gpu = 0; Group = "gate";
       Note = "68M Orig 基线 addbit=1，实验室机首跑，统一 68M 口径（开发机为 172.29/175.14）" },
    @{ Tag = "r9_68m_orig_a2"; Model = $Model68M; Net = $Net68M; Seed = 2; Cache = "./cache"; Addbit = 2;
       Extra = "--orig_spikellm"; Gpu = 0; Group = "gate";
       Note = "68M Orig@a2 前沿中点，参照 Orig@a1(本轮) 与 Orig@a3=176.0957" },
    @{ Tag = "r9_68m_ours_a2"; Model = $Model68M; Net = $Net68M; Seed = 2; Cache = "./cache"; Addbit = 2;
       Extra = ""; Gpu = 0; Group = "gate";
       Note = "68M Ours@a2 前沿中点，参照 Ours@a1=156.84 与 Ours@a3=153.4056" },
    @{ Tag = "r9_7b_orig_a2"; Model = $Model7B; Net = $Net7B; Seed = 2; Cache = "./cache"; Addbit = 2;
       Extra = "--orig_spikellm"; Gpu = 0; Group = "main";
       Note = "7B Orig@a2 前沿中点，参照 a1=10.5726 / a3=10.1386" },
    @{ Tag = "r9_7b_ours_a2"; Model = $Model7B; Net = $Net7B; Seed = 2; Cache = "./cache"; Addbit = 2;
       Extra = ""; Gpu = 1; Group = "main";
       Note = "7B Ours@a2 前沿中点，参照 a1=7.9792 / a3=7.8978" }
)

function Write-Step($msg) {
    Write-Host ""
    Write-Host ("=" * 74)
    Write-Host "  $msg"
    Write-Host ("=" * 74)
}

function Fail($msg) {
    Write-Host "[FAIL] $msg" -ForegroundColor Red
    throw $msg
}

# ---------------------------------------------------------------- 单腿执行体
function Invoke-TrainBody {
    param($Wd, $PyExe, $Model, $Net, $JobTag, $Seed, $CacheDir, $Gpu, $ExtraStr, $Addbit)

    if ([string]::IsNullOrWhiteSpace($ExtraStr)) { $Extra = @() }
    else { $Extra = $ExtraStr -split "\|" }

    Set-Location $Wd
    $env:CUDA_VISIBLE_DEVICES = "$Gpu"
    $dir = "./log/$JobTag"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null

    $base = @(
        "main.py", "--model", $Model, "--net", $Net,
        "--wbits", "4", "--abits", "4", "--addbit", "$Addbit", "--low_p", "0.9",
        "--let", "--lwc", "--aug_loss",
        "--alpha", "0.75", "--let_lr", "1e-3", "--lwc_lr", "1e-2",
        "--nsamples", "128", "--epochs", "20", "--seed", "$Seed",
        "--cache_dir", $CacheDir
    )
    if ($Extra) { $base += $Extra }

    Write-Output ("[{0}] addbit={1}  fake=高显著{2}-bit / 真实脉冲 T={3}  GPU{4}" -f $JobTag, $Addbit, (4 + $Addbit), [Math]::Pow(2, $Addbit), $Gpu)

    if (-not (Test-Path "$dir/done_ppl.flag")) {
        Write-Output "[$JobTag] fake PPL 开始"
        $argv = $base + @("--eval_ppl", "--output_dir", $dir)
        & $PyExe @argv *>> "$dir/run.log"
        if ($LASTEXITCODE -ne 0) {
            Move-Item "$dir/run.log" "$dir/run.log.failed" -Force -ErrorAction SilentlyContinue
            throw "[$JobTag] fake PPL 失败 (exit $LASTEXITCODE)，见 $dir/run.log.failed"
        }
        New-Item -ItemType File -Force -Path "$dir/done_ppl.flag" | Out-Null
        Write-Output "[$JobTag] fake PPL 完成"
    } else {
        Write-Output "[$JobTag] done_ppl 已存在，跳过"
    }

    New-Item -ItemType File -Force -Path "$dir/done.flag" | Out-Null
}

# ---------------------------------------------------------------- preflight
function Invoke-Preflight {
    Write-Step "第九轮 preflight"
    $ok = $true

    Write-Host "1/6 Python 与依赖"
    $v = & $Py -c "import torch,transformers,sys;print(sys.version.split()[0],torch.__version__,torch.version.cuda,transformers.__version__)"
    Write-Host "     python / torch / cuda / transformers = $v"
    if ($LASTEXITCODE -ne 0) { $ok = $false }

    Write-Host "2/6 模型权重"
    if (Test-Path $Model7B) { Write-Host "     7B  OK  $Model7B" }
    else { Write-Host "     7B  缺失 $Model7B" -ForegroundColor Yellow; $ok = $false }
    Write-Host "     68M 走 HF 缓存或本地路径：$Model68M"

    Write-Host "3/6 act_scales / act_shifts（--let 必需）"
    foreach ($n in @($Net7B, $Net68M)) {
        foreach ($sub in @("act_scales", "act_shifts")) {
            $p = Join-Path $Root "$sub\$n.pt"
            if (Test-Path $p) { Write-Host "     OK  $sub\$n.pt" }
            else { Write-Host "     缺失 $sub\$n.pt" -ForegroundColor Yellow; $ok = $false }
        }
    }

    Write-Host "4/6 缓存"
    foreach ($f in @("dataloader_llama_wikitext2_128.cache", "testloader_llama_wikitext2_all.cache", "testloader_llama_c4_all.cache")) {
        $p = Join-Path $Root "cache\$f"
        if (Test-Path $p) { Write-Host "     OK  cache\$f" }
        else { Write-Host "     缺失 cache\$f" -ForegroundColor Yellow }
    }

    Write-Host "5/6 磁盘余量"
    $drive = (Get-Item $Root).PSDrive
    $free = [Math]::Round((Get-PSDrive $drive.Name).Free / 1GB, 1)
    Write-Host ("     {0}: 剩余 {1} GB" -f $drive.Name, $free)
    if ($free -lt 40) { Write-Host "     余量偏低，建议先清理 log\ 下的旧 checkpoint" -ForegroundColor Yellow }

    Write-Host "6/6 环境指纹"
    if (Test-Path (Join-Path $Root "env_report.py")) {
        & $Py (Join-Path $Root "env_report.py") --out (Join-Path $Root "env_round9_preflight.txt") | Out-Null
        if (Test-Path (Join-Path $Root "env_round9_preflight.txt")) { Write-Host "     OK  env_round9_preflight.txt" }
        else { Write-Host "     env_report.py 执行未产出文件" -ForegroundColor Yellow }
    } else {
        Write-Host "     env_report.py 不存在，跳过" -ForegroundColor Yellow
    }

    Write-Host ""
    if ($ok) { Write-Host "preflight 通过，可以 -Mode run" -ForegroundColor Green }
    else { Write-Host "preflight 有缺失项，请先补齐再跑" -ForegroundColor Yellow }
}

# ---------------------------------------------------------------- 结果汇总
function New-ResultFile {
    $out = Join-Path $Root "lab_round9_result.txt"
    $L = New-Object System.Collections.ArrayList
    [void]$L.Add("SpikeLLM / OSDSQ v2  第九轮实验结果（实验室双 3090）")
    [void]$L.Add("生成时间: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    [void]$L.Add("项目路径: " + $Root)
    [void]$L.Add("本轮任务: addbit=2 前沿中点 + 68M Orig 基线统一（同机）")
    [void]$L.Add("")
    [void]$L.Add("---- 已有参照（fake_binary, 128x20, alpha=0.75, let_lr=1e-3, seed=2）----")
    [void]$L.Add("  7B  FP16            5.4722 / 6.9727")
    [void]$L.Add("  7B  Orig  a1=10.5726 / 13.7738    a3=10.1386 / 12.8944")
    [void]$L.Add("  7B  Ours  a1= 7.9792 / 10.6792    a3= 7.8978 / 10.5766")
    [void]$L.Add("  68M Ours  a1=156.84 / 90.53       a3=153.4056 / 88.9996")
    [void]$L.Add("  68M Orig  a3=176.0957 / 99.8575   (a1 基线本轮补)")
    [void]$L.Add("")

    foreach ($j in $Jobs) {
        $dir = Join-Path $Root ("log\" + $j.Tag)
        [void]$L.Add("### " + $j.Tag + "  (GPU" + $j.Gpu + ", seed=" + $j.Seed + ", addbit=" + $j.Addbit + ")")
        [void]$L.Add("  说明: " + $j.Note)
        $f = Get-ChildItem -Path $dir -Filter "log_rank0_*.txt" -ErrorAction SilentlyContinue |
             Sort-Object LastWriteTime | Select-Object -Last 1
        if ($f) {
            $w = (Select-String -Path $f.FullName -Pattern "INFO wikitext2 :" | Select-Object -Last 1)
            $c = (Select-String -Path $f.FullName -Pattern "INFO c4 :" | Select-Object -Last 1)
            $wv = if ($w) { ($w.Line -split ":")[-1].Trim() } else { "NA" }
            $cv = if ($c) { ($c.Line -split ":")[-1].Trim() } else { "NA" }
            [void]$L.Add("  [fake] wikitext2 : " + $wv)
            [void]$L.Add("  [fake] c4 : " + $cv)
        } else {
            [void]$L.Add("  [fake] 未找到 log_rank0_*.txt")
        }
        $nan = 0
        $watch = Join-Path $dir "run.log"
        if (Test-Path $watch) {
            $nan = (Select-String -Path $watch -Pattern "norm:nan" -ErrorAction SilentlyContinue | Measure-Object).Count
        }
        [void]$L.Add("  norm:nan 次数: " + $nan)
        [void]$L.Add("")
    }

    [void]$L.Add("---- 对表（非判据）----")
    [void]$L.Add("  7B Orig@a2 预期落在 10.14 ~ 10.57；Ours@a2 预期落在 7.90 ~ 7.98")
    [void]$L.Add("  68M Orig@a1 预期接近开发机的 172.29~175.14（若差 >3% 属机器差异，如实记录）")
    [void]$L.Add("  若任一点非单调（a2 比 a1 和 a3 都差），原样回传，不要自行解释")
    [void]$L.Add("")
    [void]$L.Add("---- 环境指纹 ----")
    if (Test-Path (Join-Path $Root "env_round9_preflight.txt")) {
        [void]$L.Add("  见 env_round9_preflight.txt（随日志一并回传）")
    } else {
        [void]$L.Add("  未生成。本轮结果按规程【不可入账】，请补跑：python env_report.py --out env_round9_preflight.txt")
    }

    Set-Content -Path $out -Value $L -Encoding UTF8
    Write-Host "结果已写入 $out"
}

# ---------------------------------------------------------------- 状态
function Invoke-Status {
    Write-Step "第九轮状态"
    foreach ($j in $Jobs) {
        $dir = Join-Path $Root ("log\" + $j.Tag)
        $p = Test-Path (Join-Path $dir "done_ppl.flag")
        $d = Test-Path (Join-Path $dir "done.flag")
        $flag = if ($d) { "DONE" } elseif ($p) { "fake完成" } else { "未开始" }
        Write-Host ("  {0,-18} addbit={1}  {2}" -f $j.Tag, $j.Addbit, $flag)
    }
    Write-Host ""
    $txt = Join-Path $Root "lab_round9_result.txt"
    if (Test-Path $txt) { Get-Content $txt -Encoding UTF8 | Select-Object -First 40 }
    else { Write-Host "  （尚未生成结果文件）" }
}

# ---------------------------------------------------------------- 主入口
switch ($Mode) {

    "preflight" { Invoke-Preflight }

    "status" { Invoke-Status }

    "runleg" {
        $j = $Jobs | Where-Object { $_.Tag -eq $Tag }
        if (-not $j) { Fail "未知的 Tag: $Tag" }
        Invoke-TrainBody -Wd $Root -PyExe $Py -Model $j.Model -Net $j.Net -JobTag $j.Tag `
            -Seed $j.Seed -CacheDir $j.Cache -Gpu $j.Gpu -ExtraStr $j.Extra -Addbit $j.Addbit
    }

    "gate" {
        Write-Step "只跑 68M 门禁（约 40 分钟）"
        foreach ($j in ($Jobs | Where-Object { $_.Group -eq "gate" })) {
            Invoke-TrainBody -Wd $Root -PyExe $Py -Model $j.Model -Net $j.Net -JobTag $j.Tag `
                -Seed $j.Seed -CacheDir $j.Cache -Gpu 0 -ExtraStr $j.Extra -Addbit $j.Addbit
        }
        New-ResultFile
    }

    "run" {
        Write-Step "开始执行（门禁 40 分钟 + 两条 7B 并行约 7~8 小时）"
        $start = Get-Date

        Write-Host "步骤 1/2：68M 门禁（失败不阻塞主实验）"
        foreach ($j in ($Jobs | Where-Object { $_.Group -eq "gate" })) {
            try {
                Invoke-TrainBody -Wd $Root -PyExe $Py -Model $j.Model -Net $j.Net -JobTag $j.Tag `
                    -Seed $j.Seed -CacheDir $j.Cache -Gpu 0 -ExtraStr $j.Extra -Addbit $j.Addbit
            } catch {
                Write-Host "[WARN] 门禁腿 $($j.Tag) 失败：$_" -ForegroundColor Yellow
            }
        }
        New-ResultFile

        Write-Host ""
        Write-Host "步骤 2/2：两条 7B 腿并行（GPU0 = Orig@a2，GPU1 = Ours@a2）"
        $legs = @("r9_7b_orig_a2", "r9_7b_ours_a2")
        $psExe = (Get-Process -Id $PID).Path
        $procs = @()
        foreach ($lg in $legs) {
            $jj = $Jobs | Where-Object { $_.Tag -eq $lg }
            $argList = @(
                "-ExecutionPolicy", "Bypass", "-NoProfile", "-File", $PSCommandPath,
                "-Mode", "runleg", "-Tag", $lg, "-Py", $Py, "-ModelDir", $ModelDir
            )
            $p = Start-Process -FilePath $psExe -ArgumentList $argList -PassThru -WindowStyle Hidden
            $procs += $p
            Write-Host ("  启动 {0}  PID={1}  GPU{2}" -f $lg, $p.Id, $jj.Gpu)
        }
        foreach ($p in $procs) { $p.WaitForExit() }

        $elapsed = (Get-Date) - $start
        Write-Host ""
        Write-Host ("全部结束，用时 {0:N1} 小时" -f $elapsed.TotalHours)
        New-ResultFile

        Write-Step "结果预览"
        Get-Content (Join-Path $Root "lab_round9_result.txt") -Encoding UTF8 | Select-Object -First 60
    }
}
