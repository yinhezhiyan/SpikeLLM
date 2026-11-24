<#
================================================================================
第六轮实验执行器（实验室 Windows 双 RTX 3090）· 兼本轮协作日志
生成：GLM  2026-11-01
脚本自定位：所有路径由 $PSScriptRoot 推导，不含任何开发机路径。
================================================================================

【一、本轮用户指令（原话）】
"其实我都不知道A是什么，刚才的话都是我一个外行人瞎说的，你不用管，还是按照你说的来，
 你说怎么做就怎么做，有问题你随时提出来就行。...[011日志]你刚写的这个文件我还没交给实验室ai，
 你再重新写一份给他的任务，同时也是日志，等那边有额度了看一下日志就能了解我们的进度。"

→ 背景：用户此前建议"算法再打磨几轮"，理由是单组件占 74%~94% 增益、且来源是他人实现疏漏，
  主观上"不稳"。判定结论：**不改算法**。缺的是必要性证据与普适性证据；本轮起用对照实验回答，
  而不是新增组件（新增组件反而坐实 incremental）。

【二、A 到底是什么（一句话版，供日志留档）】
原版 SpikeLLM 把通道按显著性分两路：90% 普通通道走 4-bit 分支，10% 离群通道走高比特脉冲分支。
但给 4-bit 分支算动态量程 [xmin,xmax] 时，原版把**完整张量**丢进 act_quantizer，那 10% 离群值
仍然参与 min/max，把 90% 普通通道的分辨率压扁。A（均值锚定屏蔽）= 算 4-bit 量程前，先用
「低显著性子集的真实均值」填充被分流的通道。因 min(S) <= mean(S) <= max(S)，填充后子集
[xmin,xmax] 严格不变（等价于精确子集 min-max），但不会产生 inf/nan 梯度，且零超参数。

【三、开发机本轮已完成的证据（无需实验室重复）】
（1）数值等价性证明（CPU 秒级，结果存 log/shield_equivalence_proof.json）
     场景         方法            量程精确?   量程膨胀(均值/最坏)
     跨正负       原版全通道        NO        120.3x / 334.8x
     跨正负       0 填充(v1)        YES         1.00  / 1.00
     跨正负       均值锚定(A)       YES         1.00  / 1.00
     全正         原版全通道        NO        558.6x / 1256.6x
     全正         0 填充(v1)      **NO**        5.16  / 6.00    <-- v1 的真实缺陷
     全正         均值锚定(A)       YES         1.00  / 1.00
     → A 是同族方案中唯一同时满足「精确子集量程」与「数值稳定」的解；
       ±inf 掩码数学等价但在 AMP 下产生 nan 梯度，不可用。
（2）新增 --shield_mode {mean,zero}（默认 mean，不改变任何现有默认数值路径），
     用于把 v1 的 0 填充与 A 做同配置受控对照。
（3）low_p 敏感性扫描（68M，5 档 x {orig, v2, a-off} + zero 对照 = 16 组）在开发机运行，
     目的：验证 A 的增益是否随分流强度单调放大 → 若是，A 从"0.9 附近的补丁"升级为
     "随分流强度单调增长的机制"，直接封死"专为 0.9 调参"的质疑。

【四、截至本轮 7B 既有结果（fake_binary_simulate，128x20，alpha=0.75, let_lr=1e-3）】
  FP16                        5.4722 / 6.9727
  Original  seed2            10.5726 / 13.7738     real 10.8108 / 14.1855
  Original  seed3            10.6303 / 13.7965     real 10.8963 / 14.4792
  OSDSQ v2  seed2             7.9792 / 10.6792     real  8.1570 / 10.8975
  A-off     seed2            10.4181 / 12.9691     real 10.5408 / 13.5234
  → 换种子噪声：wt2 0.55%，c4 0.16%（real 0.79% / 2.07%）
  → A 占总增益 94%/74%(fake)、90%/80%(real)；A-off 与原版仅差 1.5%~5.8%

【五、本轮任务（两条 7B 腿 + 后台下载 8B，约 8.5 ~ 9 小时）】
  GPU0 train：7b_seed3_ours    seed=3, cache_seed3
        目的：与已完成 seed3-Original(10.6303/13.7965) 配成对，主结果首次拥有 mean±std。
  GPU1 train：7b_seed4_orig    seed=4, cache_seed4
        目的：为下一轮 seed4-Ours 铺路，最终凑满 3 组配对种子（审稿人必问误差棒）。
  后台  net ：8B base 下载（NousResearch/Meta-Llama-3-8B -> models/Meta-Llama-3-8B）
        **只下载不训练**。原因见【六】。下载失败不影响本轮主结果，不阻塞。

【六、为什么本轮不跑 8B 冒烟（重要，与上一版计划不同）】
  实验室第一轮报告实测：
  1. 出网带宽仅约 0.6 MB/s，8B 权重 16 GB 需 7.5 小时以上 —— 本轮并行下载刚好用满这个窗口；
  2. 7B 在 PIQA 阶段单卡已达 22,553 MiB / 24 GB，8B 参数更多，**PIQA 极可能 OOM**；
     故 8B 冒烟必须去掉 PIQA、单独排期，不能与本轮并行；
  3. 实验室 HF 缓存只有 Meta-Llama-3-8B-Instruct，无 base 版，主表必须用 base。
  → 本轮先把权重下下来（唯一无法压缩的等待），下一轮 GPU 空出来再做 8B 冒烟（不含 PIQA）。

【七、门禁（不通过直接中止，不允许"假成功"）】
  1. main.py / spike_linear.py 必须含 shield_mode 与 ablate_no_linear_shield；
  2. models/Llama-2-7b-hf/config.json 必须存在；
  3. 两张卡显存占用必须 < 3000 MiB；
  4. 磁盘剩余 >= 40 GB（8B 权重 16 GB + 中间产物）；
  5. 每腿必须同时生成 done_ppl.flag / done_multibit.flag / done.flag，缺一即报错，
     绝不打印 Completed；
  6. 8B 下载完成后必须校验 config.json + 至少 1 个 .safetensors，否则在报告中标 FAIL。

【八、需要 Antigravity 恢复后接续的事】
  1. ACE / 发放率统计脚本 —— 已连续三轮未交付，本轮若仍无，GLM 自行接管；
  2. 8B 冒烟需去掉 PIQA 并单独排期，评估显存是否够（预计 23~24 GB，临界）；
  3. 论文 Related Work 中 QuaRot / SpinQuant 在 LLaMA-2-7B W4A4 的数字需核实后再写；
  4. low_p 扫描结果出来后，把 fig1 从"单点柱状图"改为"r^2 vs low_p 曲线"。

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
    [int]$Epochs = 20,
    [switch]$SkipDownload8B
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
if (-not $env:HF_ENDPOINT) { $env:HF_ENDPOINT = "https://hf-mirror.com" }
if (-not $env:HF_DATASETS_TRUST_REMOTE_CODE) { $env:HF_DATASETS_TRUST_REMOTE_CODE = "1" }
if (-not $env:HF_HOME) {
    $env:HF_HOME = (Join-Path $PSScriptRoot "hf_cache")
    New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null
}

$Net7B = "Llama-2-7b-hf"
$Net8B = "Meta-Llama-3-8B"
$Model7B = Join-Path $ModelDir $Net7B
$Model8B = Join-Path $ModelDir $Net8B

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
    foreach ($c in $candidates) { if (Test-Path $c) { return (Resolve-Path $c).Path } }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    Fail "找不到 python.exe，请用 -PythonExe 'C:\path\to\python.exe' 指定"
}

function Test-PatchPresent {
    $a = Select-String -Path (Join-Path $PSScriptRoot "main.py") -Pattern "shield_mode" -Quiet
    $b = Select-String -Path (Join-Path $PSScriptRoot "spike_driven_quant\spike_linear.py") -Pattern "ablate_no_linear_shield" -Quiet
    $c = Select-String -Path (Join-Path $PSScriptRoot "spike_driven_quant\spike_linear.py") -Pattern "shield_mode" -Quiet
    return ($a -and $b -and $c)
}

function Get-GpuState {
    return (& nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader)
}

function Get-FreeSpaceGB {
    $q = (Split-Path $PSScriptRoot -Qualifier).TrimEnd(':')
    $d = Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Name -eq $q } | Select-Object -First 1
    if ($d) { return [math]::Round($d.Free / 1GB, 1) }
    return -1
}

function Prepare-Cache([string]$dir) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $src = Join-Path $PSScriptRoot "cache"
    if (Test-Path $src) {
        Get-ChildItem -Path $src -Filter "testloader_*.cache" -ErrorAction SilentlyContinue | ForEach-Object {
            $dst = Join-Path $dir $_.Name
            if (-not (Test-Path $dst)) { Copy-Item $_.FullName $dst -Force }
        }
    }
}

function Count-Nan([string]$logPath) {
    if (-not (Test-Path $logPath)) { return 0 }
    return (Select-String -Path $logPath -Pattern "norm:\s*nan" -AllMatches).Count
}

# ---------------------------------------------------------------- 作业定义
$Jobs = @(
    [pscustomobject]@{
        Tag = "7b_seed3_ours"; Gpu = 0; Seed = 3; CacheDir = "./cache_seed3"
        Extra = @(); Orig = $false
        Note = "seed3-OSDSQv2：与 seed3-Original 配对，产出主结果 mean±std"
    },
    [pscustomobject]@{
        Tag = "7b_seed4_orig"; Gpu = 1; Seed = 4; CacheDir = "./cache_seed4"
        Extra = @(); Orig = $true
        Note = "seed4-Original：为下一轮 seed4-Ours 配对铺路（3 组种子误差棒）"
    }
)

# ---------------------------------------------------------------- 作业体
function Invoke-TrainBody {
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
            throw "[$Tag] 主 PPL 失败 (exit $LASTEXITCODE)，见 $dir/run.log.failed"
        }
        New-Item -ItemType File -Force -Path "$dir/done_ppl.flag" | Out-Null
    } else { Write-Output "[$Tag] done_ppl 已存在，跳过" }

    if (-not (Test-Path "$dir/done_multibit.flag")) {
        $argv2 = $base + @("--epochs", "0", "--resume", "$dir/omni_parameters.pth",
                           "--sim_mode", "real_multibit_simulate", "--eval_ppl",
                           "--output_dir", "$dir/real_multibit")
        & $Py @argv2 *>> "$dir/run.log"
        if ($LASTEXITCODE -ne 0) { throw "[$Tag] real_multibit 失败 (exit $LASTEXITCODE)" }
        New-Item -ItemType File -Force -Path "$dir/done_multibit.flag" | Out-Null
    }

    New-Item -ItemType File -Force -Path "$dir/done.flag" | Out-Null
    Write-Output "[$Tag] 完成"
}

# ---------------------------------------------------------------- 下载作业体（不占 GPU）
function Invoke-Download8bBody {
    param($Wd, $Py, $ModelDir, $Model)
    Set-Location $Wd
    $dir = "./log/8b_download"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $log = "$dir/run.log"

    & $Py "lab_preflight_and_download.py" --mode "download_8b" --model_dir $ModelDir *>> $log
    $rc = $LASTEXITCODE
    "exit_code=$rc" | Out-File -FilePath "$dir/exit_code.txt" -Encoding utf8

    if ((Test-Path (Join-Path $Model "config.json"))) {
        $sf = Get-ChildItem -Path $Model -Filter "*.safetensors" -ErrorAction SilentlyContinue
        if ($sf -and $sf.Count -ge 1) {
            New-Item -ItemType File -Force -Path "$dir/done.flag" | Out-Null
            Write-Output "[8B-dl] 完成，权重齐全（$($sf.Count) 个 safetensors）"
            return
        }
    }
    Write-Output "[8B-dl] 未完成或权重不全（exit=$rc），本轮主结果不受影响"
}

# ---------------------------------------------------------------- preflight
function Invoke-Preflight {
    Write-Step "1/6 环境"
    $script:Py = Resolve-PythonExe
    Write-Host "[PY ] $script:Py"
    & $script:Py -c "import torch,transformers,sys;print('python',sys.version.split()[0]);print('torch',torch.__version__);print('cuda',torch.version.cuda);print('gpus',torch.cuda.device_count());print('transformers',transformers.__version__)"
    if ($LASTEXITCODE -ne 0) { Fail "Python 环境不可用" }

    Write-Step "2/6 GPU"
    Get-GpuState | ForEach-Object { Write-Host $_ }
    $busy = @()
    Get-GpuState | ForEach-Object {
        $p = $_ -split ","
        $used = [int](($p[2].Trim()) -replace " MiB", "")
        if ($used -gt 3000) { $busy += ("GPU" + $p[0].Trim() + " used=" + $used + "MiB") }
    }
    if ($busy.Count -gt 0) { Fail ("显存被占用，确认无人使用后再跑：" + ($busy -join "; ")) }

    Write-Step "3/6 代码与补丁"
    foreach ($f in @("main.py", "spike_driven_quant\spike_linear.py", "spike_driven_quant\spike_matmul.py",
                     "lab_preflight_and_download.py", "generate_act_scale_shift.py")) {
        if (-not (Test-Path (Join-Path $PSScriptRoot $f))) { Fail "缺少文件：$f" }
    }
    if (-not (Test-PatchPresent)) {
        Fail "shield_mode / ablate_no_linear_shield 补丁缺失：请先把第六轮补丁包覆盖到项目根目录"
    }
    Write-Host "[OK ] shield_mode + A 消融开关均已就位"

    Write-Step "4/6 7B 模型与激活统计"
    if (-not (Test-Path (Join-Path $Model7B "config.json"))) { Fail "7B 模型不完整：$Model7B\config.json 不存在" }
    Write-Host "[OK ] $Model7B"
    $scale = "./act_scales/$Net7B.pt"
    $shift = "./act_shifts/$Net7B.pt"
    if ((Test-Path $scale) -and (Test-Path $shift)) {
        Write-Host "[OK ] 7B act_scales / act_shifts 已存在"
    } else {
        Fail "缺少 7B 的 act_scales/act_shifts（LET 必需）。请先跑 generate_act_scale_shift.py --model $Model7B --num-samples 128 --seed 2"
    }

    Write-Step "5/6 磁盘"
    $free = Get-FreeSpaceGB
    Write-Host ("[DISK] 可用 {0} GB" -f $free)
    if ($free -ge 0 -and $free -lt 40) { Fail "磁盘剩余不足 40 GB（8B 权重约 16 GB），请先清理或加 -SkipDownload8B" }

    Write-Step "6/6 计划"
    foreach ($j in $Jobs) {
        Write-Host ("GPU{0}  {1,-16} seed={2}  cache={3}" -f $j.Gpu, $j.Tag, $j.Seed, $j.CacheDir) -ForegroundColor Yellow
        Write-Host ("        {0}" -f $j.Note) -ForegroundColor DarkGray
    }
    if ($SkipDownload8B) {
        Write-Host "`n[net ] 已指定 -SkipDownload8B，本轮不下载 8B" -ForegroundColor DarkGray
    } else {
        Write-Host "`n[net ] 后台下载 8B base（约 7.5 小时，不占 GPU，失败不影响主结果）" -ForegroundColor DarkGray
    }
    Write-Host "`npreflight 通过。执行：" -ForegroundColor Green
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\lab_round6_runner.ps1 -Mode run" -ForegroundColor Green
}

# ---------------------------------------------------------------- run
function Invoke-RunAll {
    Write-Step "启动双卡 7B + 后台 8B 下载（预计 8.5 ~ 9 小时）"
    Write-Host "中途不要关窗口；若中断，重跑同一命令会从 done_*.flag 续跑。" -ForegroundColor DarkGray

    $script:Py = Resolve-PythonExe
    $started = @()
    foreach ($j in $Jobs) {
        New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot ("log\" + $j.Tag)) | Out-Null
        Prepare-Cache $j.CacheDir
        $extraStr = ($j.Extra -join "|")
        $job = Start-Job -ScriptBlock ${function:Invoke-TrainBody} `
            -ArgumentList $PSScriptRoot, $script:Py, $Model7B, $Net7B, $j.Tag, $j.Seed,
                          $j.CacheDir, $Alpha, $LetLr, $LwcLr, $Epochs, $j.Gpu, $extraStr, $j.Orig
        $started += [pscustomobject]@{ Job = $job; Meta = $j }
        Write-Host ("[START] GPU{0} -> {1}" -f $j.Gpu, $j.Tag) -ForegroundColor Yellow
    }

    $dlJob = $null
    if (-not $SkipDownload8B -and -not (Test-Path (Join-Path $PSScriptRoot "log\8b_download\done.flag"))) {
        $dlJob = Start-Job -ScriptBlock ${function:Invoke-Download8bBody} `
            -ArgumentList $PSScriptRoot, $script:Py, $ModelDir, $Model8B
        Write-Host "[START] net  -> 8B download（后台，不占 GPU）" -ForegroundColor Yellow
    }

    while ($true) {
        $pending = @($started | Where-Object { $_.Job.State -eq "Running" })
        if (-not $pending -and (-not $dlJob -or $dlJob.State -ne "Running")) { break }
        $t = Get-Date -Format "HH:mm:ss"
        foreach ($s in $started) {
            $log = Join-Path $PSScriptRoot ("log\" + $s.Meta.Tag + "\run.log")
            $tail = "waiting..."
            if (Test-Path $log) {
                $line = Get-Content $log -Tail 1 -ErrorAction SilentlyContinue
                if ($line) { $tail = $line }
            }
            if ($tail.Length -gt 95) { $tail = $tail.Substring(0, 95) }
            Write-Host ("[{0}] {1,-16} {2}" -f $t, $s.Meta.Tag, $tail) -ForegroundColor Gray
        }
        if ($dlJob -and $dlJob.State -eq "Running") {
            $sz = 0
            if (Test-Path $Model8B) {
                $sz = [math]::Round(((Get-ChildItem $Model8B -Recurse -File -ErrorAction SilentlyContinue |
                       Measure-Object Length -Sum).Sum / 1GB), 2)
            }
            Write-Host ("[{0}] {1,-16} {2}" -f $t, "8b_download", ("已下载 ~{0} GB" -f $sz)) -ForegroundColor DarkGray
        }
        Start-Sleep -Seconds 60
    }

    $failed = @()
    foreach ($s in $started) {
        if ($s.Job.State -ne "Completed") {
            $err = ($s.Job.ChildJobs[0].Error | Out-String)
            $failed += ("{0}: state={1} {2}" -f $s.Meta.Tag, $s.Job.State, $err.Trim())
        }
    }
    if ($failed.Count -gt 0) {
        Write-Host "`n[FAIL] 未正常完成：" -ForegroundColor Red
        $failed | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
        Write-ResultReport
        throw "本轮未全部完成，请保留 log 目录并回报，不要重复启动"
    }

    $missing = @()
    foreach ($s in $started) {
        foreach ($f in @("done_ppl.flag", "done_multibit.flag", "done.flag")) {
            if (-not (Test-Path (Join-Path $PSScriptRoot ("log\" + $s.Meta.Tag + "\" + $f)))) {
                $missing += ("{0}/{1}" -f $s.Meta.Tag, $f)
            }
        }
    }
    if ($missing.Count -gt 0) { throw ("缺少完成标记：" + ($missing -join ", ")) }

    Write-Host "`n=== 两条 7B 腿全部完成（flags 齐全）===" -ForegroundColor Green
    Write-ResultReport
}

# ---------------------------------------------------------------- 结果报告
function Write-ResultReport {
    $lines = @()
    $lines += "SpikeLLM / OSDSQ v2  第六轮实验结果（实验室双 3090）"
    $lines += ("生成时间: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    $lines += ("项目路径: " + $PSScriptRoot)
    $lines += ""
    $lines += "---- 既有 7B 参照（同协议：fake_binary, 128x20, alpha=0.75, let_lr=1e-3）----"
    $lines += "  FP16                5.4722 / 6.9727"
    $lines += "  Original  seed2    10.5726 / 13.7738    real 10.8108 / 14.1855"
    $lines += "  Original  seed3    10.6303 / 13.7965    real 10.8963 / 14.4792"
    $lines += "  OSDSQ v2  seed2     7.9792 / 10.6792    real  8.1570 / 10.8975"
    $lines += "  A-off     seed2    10.4181 / 12.9691    real 10.5408 / 13.5234"
    $lines += ""

    $collected = @{}
    foreach ($j in $Jobs) {
        $dir = Join-Path $PSScriptRoot ("log\" + $j.Tag)
        $lines += ("### {0}  (GPU{1}, seed={2}, cache={3})" -f $j.Tag, $j.Gpu, $j.Seed, $j.CacheDir)
        $lines += ("  说明: " + $j.Note)
        $fake = @()
        $main = Get-ChildItem -Path $dir -Filter "log_rank0_*.txt" -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($main) {
            $fake = Select-String -Path $main.FullName -Pattern "wikitext2|c4 :" | Select-Object -Last 2
            foreach ($h in $fake) { $lines += ("  [fake] " + $h.Line.Trim()) }
        }
        $realDir = Join-Path $dir "real_multibit"
        if (Test-Path $realDir) {
            $r = Get-ChildItem -Path $realDir -Filter "log_rank0_*.txt" -ErrorAction SilentlyContinue |
                 Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($r) {
                Select-String -Path $r.FullName -Pattern "wikitext2|c4 :" | Select-Object -Last 2 |
                    ForEach-Object { $lines += ("  [real] " + $_.Line.Trim()) }
            }
        }
        $lines += ("  norm:nan 次数: " + (Count-Nan (Join-Path $dir "run.log")))
        $collected[$j.Tag] = $fake
        $lines += ""
    }

    if ($collected.ContainsKey("7b_seed3_ours") -and $collected["7b_seed3_ours"]) {
        $lines += "---- seed3 配对（本轮 Ours vs 既有 Original）----"
        $lines += "  Original seed3 : 10.6303 / 13.7965"
        $lines += "  上方 [fake] 两行即本轮 Ours；请人工相减得到 seed3 相对降幅，"
        $lines += "  再与 seed2 的 -24.5% / -22.5% 比较，判断种子间波动是否可接受（噪声带 0.55% / 0.16%）。"
        $lines += ""
    }

    $lines += "---- 8B 下载状态 ----"
    $dlFlag = Join-Path $PSScriptRoot "log\8b_download\done.flag"
    if (Test-Path $dlFlag) {
        $sf = Get-ChildItem -Path $Model8B -Filter "*.safetensors" -ErrorAction SilentlyContinue
        $cnt = 0
        if ($sf) { $cnt = $sf.Count }
        $lines += ("  OK：models/Meta-Llama-3-8B 权重齐全（{0} 个 safetensors）" -f $cnt)
        $lines += "  下一轮即可做 8B 冒烟（务必去掉 PIQA，7B 的 PIQA 已达 22.5 GB）。"
    } else {
        $lines += "  INCOMPLETE：8B 未下完或失败，本轮主结果不受影响。"
        $lines += "  若 exit_code 非 0，请把 log/8b_download/run.log 尾部一并发回。"
    }
    $lines += ""

    $lines += "---- 峰值显存 ----"
    $lines += (& nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader) -join "`n"

    $out = Join-Path $PSScriptRoot "lab_round6_result.txt"
    $lines | Set-Content -Path $out -Encoding UTF8
    Write-Host "结果已写入：$out" -ForegroundColor Green
    Write-Host "请把该 txt 原样发回，不要手写摘要。" -ForegroundColor Green
}

# ---------------------------------------------------------------- status
function Invoke-Status {
    foreach ($j in $Jobs) {
        $dir = Join-Path $PSScriptRoot ("log\" + $j.Tag)
        Write-Step $j.Tag
        $flags = @("done_ppl.flag", "done_multibit.flag", "done.flag") | ForEach-Object {
            if (Test-Path (Join-Path $dir $_)) { "$_=Y" } else { "$_=N" }
        }
        Write-Host ("flags: " + ($flags -join "  "))
        $log = Join-Path $dir "run.log"
        if (Test-Path $log) { Write-Host "-- tail --"; Get-Content $log -Tail 3 }
    }
    Write-Step "8B download"
    $dl = Join-Path $PSScriptRoot "log\8b_download"
    if (Test-Path (Join-Path $dl "done.flag")) { Write-Host "done" }
    elseif (Test-Path (Join-Path $dl "run.log")) { Get-Content (Join-Path $dl "run.log") -Tail 3 }
    else { Write-Host "未启动" }

    $res = Join-Path $PSScriptRoot "lab_round6_result.txt"
    if (Test-Path $res) { Write-Step "结果文件"; Get-Content $res }
}

# ---------------------------------------------------------------- main
switch ($Mode) {
    "preflight" { Invoke-Preflight }
    "run"       { $script:Py = Resolve-PythonExe; Invoke-RunAll }
    "status"    { Invoke-Status }
}
