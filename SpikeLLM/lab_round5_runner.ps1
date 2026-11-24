<#
================================================================================
第五轮实验执行器（实验室 Windows 双 RTX 3090）· 兼本轮协作日志
生成：GLM  2026-10-31
脚本自定位：所有路径由 $PSScriptRoot 推导，本机（开发机）路径不出现在脚本里。
================================================================================

【一、本轮用户指令（原话摘要）】
"你以后不要给我这个md文件了，这个md文件你里面指向的是这台电脑的路径，我把这个文件给实验室的电脑，
 他有的东西找不到，现在重新写一份给实验室的命令同时也是你们新一轮的协作日志。至于写成什么形式你自己看着办。"

→ 结论：不再输出"操作指南类 md"。改为一个可直接执行的 .ps1，路径全部相对项目根，
  头部注释即本轮协作记录，运行结束自动产出 txt 结果文件（不用手写报告）。

【二、截至本轮已有的 7B 证据（全部 fake_binary_simulate，seed=2，128x20，alpha=0.75）】
  FP16                     5.4722 / 6.9727
  Original (+LET+aug)     10.5726 / 13.7738
  OSDSQ v2 完整 A+B+C+D    7.9792 / 10.6792   ← 主结果，-24.5% / -22.5%
  去掉 D（等权第二目标）    8.2403 / 11.1289   ← 第三轮：D 边际 +2%~+4.3%，方向为正
  去掉 B（KV 屏蔽）         8.1102 / 10.6918   ← 第四轮 LOO
  去掉 C（步长显著性）      8.0901 / 10.7502   ← 第四轮 LOO
  real_multibit 口径：Original 11.7665 / 15.3012；OSDSQ v2 8.5797 / 11.6531

  → 判断：B / C / D 的单项效应只有 0.10~0.26 PPL，与单种子噪声同量级；
    真正能解释主增益的是 **A（低显著子集均值锚定屏蔽）**，而 A 在 7B 上**还没有独立消融**。
    开发机 68M 同版本对照：完整 159.28 / 91.74，关闭 A 后 174.22 / 101.65（回退 +9.4% / +10.8%）。
    若 7B 上 A 关掉后没有明显回退，论文的因果链就是空的，因此必须优先跑。

【三、本轮任务（两卡各一条队列，PPL + real_multibit，不跑 zero-shot，约 8 小时）】
  GPU0：A-off  —— 完整配置 + --ablate_no_linear_shield（seed=2）
                 输出 log/7b_ablate_no_linear_shield
  GPU1：seed3-Original —— 原版 SpikeLLM + --orig_spikkelm 拼写见下（seed=3，独立缓存 cache_seed3）
                 输出 log/7b_seed3_orig
  说明：seed3 先跑 Original，下一轮再跑 seed3-Ours，凑成成对种子；
        seed3 用独立 --cache_dir，避免与 seed2 的校准数据互相覆盖。

【四、为什么不跑】zero-shot 占单腿时间约 60%，本轮只做 PPL 与真实脉冲；
  8B 需要先下载，且下载可与训练并行，但本轮不启动，等 7B 结论出来再排。

【五、门禁（脚本会强制检查，不通过直接中止）】
  1. main.py / spike_linear.py 中必须存在 ablate_no_linear_shield 字样（说明 A 补丁已覆盖）；
  2. models/Llama-2-7b-hf/config.json 必须存在；
  3. act_scales / act_shifts 必须有 Llama-2-7b-hf.pt（缺了脚本会用 128 样本重建）；
  4. 两张卡显存占用必须低于 3000 MiB，否则视为有他人任务在跑，直接中止；
  5. 每条腿必须同时生成 done_ppl.flag / done_multibit.flag / done.flag，缺一即报错，
     不允许打印 "Completed"（防上一轮出现过的"假成功"）。

【六、需要 Antigravity 接续的事（本轮之后）】
  1. ACE / 发放率统计脚本（SNN 论文硬门槛，只读统计，不改数值路径）；
  2. seed3-Ours 与 seed4 成对排期；
  3. 8B（Meta-Llama-3-8B base，非 Instruct）下载与冒烟，缓存用 --cache_dir ./cache_8b 隔离；
  4. 论文 Related Work 里 QuaRot / SpinQuant 在 LLaMA-2-7B W4A4 的 WikiText-2 数字需要核实后再写。

================================================================================
#>

param(
    [ValidateSet("preflight", "run", "status")]
    [string]$Mode = "preflight",

    [string]$PythonExe = "",
    [string]$ModelDir = "./models",
    [double]$Alpha = 0.75,
    [double]$LetLr = 1e-3,
    [double]$LwcLr = 1e-2,
    [int]$Epochs = 20
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
if (-not $env:HF_ENDPOINT) { $env:HF_ENDPOINT = "https://hf-mirror.com" }
if (-not $env:HF_DATASETS_TRUST_REMOTE_CODE) { $env:HF_DATASETS_TRUST_REMOTE_CODE = "1" }
# 第五轮实验室反馈：本脚本未设 HF_HOME，会导致数据集缓存落到公共目录。
# 这里默认锚定到项目内 hf_cache，可用环境变量覆盖。
if (-not $env:HF_HOME) {
    $env:HF_HOME = (Join-Path $PSScriptRoot "hf_cache")
    New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null
}

$NetName   = "Llama-2-7b-hf"
$ModelPath = Join-Path $ModelDir $NetName

# ---------------------------------------------------------------- 工具函数
function Write-Step([string]$msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Fail([string]$msg) {
    Write-Host "[FAIL] $msg" -ForegroundColor Red
    throw $msg
}

function Resolve-PythonExe {
    if (-not [string]::IsNullOrWhiteSpace($PythonExe)) { return $PythonExe }
    $candidates = @(
        "C:\Users\admin\.conda\envs\spikellm\python.exe",
        (Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"),
        (Join-Path $PSScriptRoot ".venv\Scripts\python.exe")
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return (Resolve-Path $c).Path }
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    Fail "找不到 python.exe，请用 -PythonExe 'C:\path\to\python.exe' 指定"
}

function Test-PatchPresent {
    $mainHit  = Select-String -Path (Join-Path $PSScriptRoot "main.py") -Pattern "ablate_no_linear_shield" -Quiet
    $linearHit = Select-String -Path (Join-Path $PSScriptRoot "spike_driven_quant\spike_linear.py") -Pattern "ablate_no_linear_shield" -Quiet
    return ($mainHit -and $linearHit)
}

function Get-GpuState {
    $raw = & nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
    return $raw
}

function Prepare-Cache([string]$dir) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $src = Join-Path $PSScriptRoot "cache"
    if (Test-Path $src) {
        # 只复制测试集缓存（与 seed 无关）；dataloader 缓存含校准子集索引，必须由新 seed 重新生成
        Get-ChildItem -Path $src -Filter "testloader_*.cache" -ErrorAction SilentlyContinue | ForEach-Object {
            $dst = Join-Path $dir $_.Name
            if (-not (Test-Path $dst)) { Copy-Item $_.FullName $dst -Force }
        }
    }
}

# ---------------------------------------------------------------- 作业定义
# 每条腿：Tag / GPU / Seed / CacheDir / ExtraArgs / OrigFlag
$Jobs = @(
    [pscustomobject]@{
        Tag       = "7b_ablate_no_linear_shield"
        Gpu       = 0
        Seed      = 2
        CacheDir  = "./cache"
        Extra     = @("--ablate_no_linear_shield")
        Orig      = $false
        Note      = "A-off：关闭低显著子集均值锚定屏蔽（论文因果链核心）"
    },
    [pscustomobject]@{
        Tag       = "7b_seed3_orig"
        Gpu       = 1
        Seed      = 3
        CacheDir  = "./cache_seed3"
        Extra     = @()
        Orig      = $true
        Note      = "seed3 原版基线（为 seed3 成对比较做准备）"
    }
)

# ---------------------------------------------------------------- preflight
function Invoke-Preflight {
    Write-Step "1/6 环境"
    $script:Py = Resolve-PythonExe
    Write-Host "[PY ] $script:Py"
    & $script:Py -c "import torch,transformers,sys;print('python',sys.version.split()[0]);print('torch',torch.__version__);print('cuda',torch.version.cuda);print('gpu',torch.cuda.device_count());print('transformers',transformers.__version__)"
    if ($LASTEXITCODE -ne 0) { Fail "Python 环境不可用或 torch 导入失败" }

    Write-Step "2/6 GPU"
    Get-GpuState | ForEach-Object { Write-Host $_ }
    $busy = @()
    Get-GpuState | ForEach-Object {
        $parts = $_ -split ","
        $idx = $parts[0].Trim()
        $used = [int](($parts[2].Trim()) -replace " MiB", "")
        if ($used -gt 3000) { $busy += "GPU$idx used=${used}MiB" }
    }
    if ($busy.Count -gt 0) { Fail ("有显存被占用，先确认无人使用再跑：" + ($busy -join "; ")) }

    Write-Step "3/6 代码与模型"
    foreach ($p in @("main.py", "spike_driven_quant\spike_linear.py", "spike_driven_quant\spike_matmul.py")) {
        if (-not (Test-Path (Join-Path $PSScriptRoot $p))) { Fail "缺少文件：$p" }
    }
    if (-not (Test-PatchPresent)) {
        Fail "A 消融开关不存在：请先把补丁包里的 main.py / spike_linear.py / spike_matmul.py 覆盖到项目根目录"
    }
    Write-Host "[OK ] ablate_no_linear_shield 补丁已就位"
    if (-not (Test-Path (Join-Path $ModelPath "config.json"))) {
        Fail "模型不完整：$ModelPath\config.json 不存在"
    }
    Write-Host "[OK ] 模型：$ModelPath"

    Write-Step "4/6 激活统计"
    $scale = "./act_scales/$NetName.pt"
    $shift = "./act_shifts/$NetName.pt"
    if ((Test-Path $scale) -and (Test-Path $shift)) {
        Write-Host "[OK ] act_scales / act_shifts 已存在"
    } else {
        Write-Host "[..] 缺 LET 激活统计，开始用 128 样本生成（约 20-40 分钟）"
        & $script:Py generate_act_scale_shift.py --model $ModelPath --num-samples 128 --seed 2
        if ($LASTEXITCODE -ne 0) { Fail "激活统计生成失败" }
        if (-not ((Test-Path $scale) -and (Test-Path $shift))) { Fail "激活统计文件未生成" }
    }

    Write-Step "5/6 磁盘与缓存"
    Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Name -eq (Split-Path $PSScriptRoot -Qualifier).TrimEnd(':') } |
        ForEach-Object { Write-Host ("[DISK] {0}: free {1:N1} GB" -f $_.Name, ($_.Free / 1GB)) }
    Prepare-Cache "./cache_seed3"
    Write-Host "[OK ] cache_seed3 就绪"

    Write-Step "6/6 计划"
    foreach ($j in $Jobs) {
        Write-Host ("GPU{0}  seed={1}  cache={2}  {3}" -f $j.Gpu, $j.Seed, $j.CacheDir, $j.Tag) -ForegroundColor Yellow
        Write-Host ("        {0}" -f $j.Note) -ForegroundColor DarkGray
    }
    Write-Host "`npreflight 全部通过。执行正式运行：" -ForegroundColor Green
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\lab_round5_runner.ps1 -Mode run" -ForegroundColor Green
}

# ---------------------------------------------------------------- run
function Invoke-JobBody {
    # Extra 以 "|" 连接传入，避免空数组在 Start-Job -ArgumentList 中被展平导致参数错位
    param($Wd, $Py, $Model, $Net, $Tag, $Seed, $CacheDir, $Alpha, $LetLr, $LwcLr, $Epochs, $Gpu, $ExtraStr, $Orig)

    if ([string]::IsNullOrWhiteSpace($ExtraStr)) { $Extra = @() } else { $Extra = $ExtraStr -split "\|" }
    Set-Location $Wd
    $env:CUDA_VISIBLE_DEVICES = "$Gpu"
    $dir = "./log/$Tag"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null

    $base = @(
        "main.py", "--model", $Model, "--net", $Net,
        "--wbits", "4", "--abits", "4", "--addbit", "1", "--low_p", "0.9",
        "--let", "--lwc", "--aug_loss",
        "--alpha", "$Alpha", "--let_lr", "$LetLr", "--lwc_lr", "$LwcLr",
        "--nsamples", "128", "--epochs", "$Epochs", "--seed", "$Seed",
        "--cache_dir", $CacheDir
    )
    if ($Extra) { $base += $Extra }
    if ($Orig)  { $base += "--orig_spikellm" }

    if (-not (Test-Path "$dir/done_ppl.flag")) {
        $argv = $base + @("--eval_ppl", "--output_dir", $dir)
        & $Py @argv *>> "$dir/run.log"
        if ($LASTEXITCODE -ne 0) {
            Move-Item "$dir/run.log" "$dir/run.log.failed" -Force -ErrorAction SilentlyContinue
            throw "[$Tag] 主 PPL 运行失败 (exit $LASTEXITCODE)，见 $dir/run.log.failed"
        }
        New-Item -ItemType File -Force -Path "$dir/done_ppl.flag" | Out-Null
    } else {
        Write-Output "[$Tag] done_ppl.flag 已存在，跳过训练"
    }

    if (-not (Test-Path "$dir/done_multibit.flag")) {
        $argv2 = $base + @(
            "--epochs", "0", "--resume", "$dir/omni_parameters.pth",
            "--sim_mode", "real_multibit_simulate", "--eval_ppl",
            "--output_dir", "$dir/real_multibit"
        )
        & $Py @argv2 *>> "$dir/run.log"
        if ($LASTEXITCODE -ne 0) { throw "[$Tag] real_multibit 评测失败 (exit $LASTEXITCODE)，见 $dir/run.log" }
        New-Item -ItemType File -Force -Path "$dir/done_multibit.flag" | Out-Null
    }

    New-Item -ItemType File -Force -Path "$dir/done.flag" | Out-Null
    Write-Output "[$Tag] 完成"
}

function Invoke-RunAll {
    Write-Step "启动双卡并行（PPL + real_multibit，约 8 小时）"
    Write-Host "提示：中途不要关闭窗口；若断电或被杀，重新执行本脚本会从 done_*.flag 续跑。" -ForegroundColor DarkGray

    $script:Py = Resolve-PythonExe
    $started = @()
    foreach ($j in $Jobs) {
        if (-not $j.Orig -and -not (Test-PatchPresent)) { Fail "A 消融开关缺失，请先覆盖补丁文件" }
        Prepare-Cache $j.CacheDir
        $extraStr = ($j.Extra -join "|")
        $job = Start-Job -ScriptBlock ${function:Invoke-JobBody} `
            -ArgumentList $PSScriptRoot, $script:Py, $ModelPath, $NetName, $j.Tag, $j.Seed,
                          $j.CacheDir, $Alpha, $LetLr, $LwcLr, $Epochs, $j.Gpu, $extraStr, $j.Orig
        $started += [pscustomobject]@{ Job = $job; Meta = $j }
        Write-Host ("[START] GPU{0} -> {1}" -f $j.Gpu, $j.Tag) -ForegroundColor Yellow
    }

    while ($true) {
        $running = $started | Where-Object { $_.Job.State -eq "Running" }
        if (-not $running) { break }
        $t = Get-Date -Format "HH:mm:ss"
        foreach ($s in $started) {
            $log = Join-Path $PSScriptRoot ("log\" + $s.Meta.Tag + "\run.log")
            $tail = "waiting..."
            if (Test-Path $log) {
                $line = Get-Content $log -Tail 1 -ErrorAction SilentlyContinue
                if ($line) { $tail = $line }
            }
            if ($tail.Length -gt 110) { $tail = $tail.Substring(0, 110) }
            Write-Host ("[{0}] {1,-30} {2}" -f $t, $s.Meta.Tag, $tail) -ForegroundColor Gray
        }
        Start-Sleep -Seconds 60
    }

    $failed = @()
    foreach ($s in $started) {
        $out = Receive-Job -Job $s.Job -Keep -ErrorAction SilentlyContinue
        if ($s.Job.State -ne "Completed") {
            $err = ($s.Job.ChildJobs[0].Error | Out-String)
            $failed += ("{0}: state={1} {2}" -f $s.Meta.Tag, $s.Job.State, $err.Trim())
        }
    }
    if ($failed.Count -gt 0) {
        Write-Host "`n[FAIL] 以下任务未正常完成：" -ForegroundColor Red
        $failed | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
        throw "本轮实验未全部完成，请保留 log 目录并回报，不要重复启动"
    }

    $missing = @()
    foreach ($s in $started) {
        foreach ($f in @("done_ppl.flag", "done_multibit.flag", "done.flag")) {
            $p = Join-Path $PSScriptRoot ("log\" + $s.Meta.Tag + "\" + $f)
            if (-not (Test-Path $p)) { $missing += "$($s.Meta.Tag)/$f" }
        }
    }
    if ($missing.Count -gt 0) { throw ("缺少完成标记：" + ($missing -join ", ")) }

    Write-Host "`n=== 两条腿全部完成（flags 齐全）===" -ForegroundColor Green
    Write-ResultReport
}

function Write-ResultReport {
    $lines = @()
    $lines += "SpikeLLM / OSDSQ v2  第五轮实验结果（实验室双 3090）"
    $lines += ("生成时间: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    $lines += ("项目路径: " + $PSScriptRoot)
    $lines += ""
    foreach ($j in $Jobs) {
        $dir = Join-Path $PSScriptRoot ("log\" + $j.Tag)
        $lines += ("{0}  (GPU{1}, seed={2}, cache={3})" -f $j.Tag, $j.Gpu, $j.Seed, $j.CacheDir)
        $lines += ("  说明: " + $j.Note)
        $main = Get-ChildItem -Path $dir -Filter "log_rank0_*.txt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($main) {
            $hits = Select-String -Path $main.FullName -Pattern "wikitext2|c4 :" | Select-Object -Last 2
            foreach ($h in $hits) { $lines += ("  [fake] " + $h.Line.Trim()) }
        }
        $realDir = Join-Path $dir "real_multibit"
        if (Test-Path $realDir) {
            $r = Get-ChildItem -Path $realDir -Filter "log_rank0_*.txt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($r) {
                $hits2 = Select-String -Path $r.FullName -Pattern "wikitext2|c4 :" | Select-Object -Last 2
                foreach ($h in $hits2) { $lines += ("  [real] " + $h.Line.Trim()) }
            }
        }
        $nan = 0
        $logFile = Join-Path $dir "run.log"
        if (Test-Path $logFile) {
            $nan = (Select-String -Path $logFile -Pattern "norm:\s*nan" -AllMatches).Count
        }
        $lines += ("  norm:nan 次数: " + $nan)
        $lines += ""
    }
    $out = Join-Path $PSScriptRoot "lab_round5_result.txt"
    $lines | Set-Content -Path $out -Encoding UTF8
    Write-Host "结果已写入：$out" -ForegroundColor Green
    Write-Host "请把该 txt 原样发回开发机，不要手写摘要。" -ForegroundColor Green
}

# ---------------------------------------------------------------- status
function Invoke-Status {
    foreach ($j in $Jobs) {
        $dir = Join-Path $PSScriptRoot ("log\" + $j.Tag)
        Write-Step $j.Tag
        $flags = @("done_ppl.flag", "done_multibit.flag", "done.flag") | ForEach-Object {
            $p = Join-Path $dir $_
            if (Test-Path $p) { "$_=Y" } else { "$_=N" }
        }
        Write-Host ("flags: " + ($flags -join "  "))
        $log = Join-Path $dir "run.log"
        if (Test-Path $log) {
            Write-Host "-- tail --"
            Get-Content $log -Tail 3
        }
    }
    if (Test-Path (Join-Path $PSScriptRoot "lab_round5_result.txt")) {
        Write-Step "结果文件"
        Get-Content (Join-Path $PSScriptRoot "lab_round5_result.txt")
    }
}

# ---------------------------------------------------------------- main
switch ($Mode) {
    "preflight" { Invoke-Preflight }
    "run"       { $script:Py = Resolve-PythonExe; Invoke-RunAll }
    "status"    { Invoke-Status }
}
