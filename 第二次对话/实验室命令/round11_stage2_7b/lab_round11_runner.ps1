<#
================================================================================
第十一轮实验执行器（实验室 Windows 双 RTX 3090）· 兼本轮说明文档
生成：WorkBuddy  2026-11-09
自定位：所有路径由 $PSScriptRoot 推导，脚本放在 SpikeLLM 根目录即可运行。
前提：实验室代码应已处于第十轮打补丁后的状态（含 --adaptive_quota 开关）。
      本脚本 preflight 会先验证这件事，不对就停下来告诉你怎么办。

--------------------------------------------------------------------------------
【一、为什么是这一轮】

第十轮三件事全部通过：
  门禁一：补丁忠实（68M 全开关关复跑 157.7880/90.8312，差 0.0/0.0）
  门禁二：68M --adaptive_quota 三种子 155.96 / 155.73 / 153.96（全部优于基线 157.79）
  主体：7B A-off 三种子补齐（10.4181 / 10.1475 / 10.2797）

现在 Stage 2 只差最后一跳：**在 7B 上它还灵不灵**。这决定论文的形态：
  灵 -> 两阶段方法（Stage 1 统计域隔离 + Stage 2 等算力配额重分配），贡献翻倍；
  不灵 -> Stage 2 降级为附加分析，论文仍以 Stage 1 为主，不受影响。
所以必须现在测，不能等论文写到一半。

顺带补 Codex 006 点名的配对缺口：68M 的 uniform（不开配额）只有 seed2，
本轮补 seed3/4，让门禁二的三个数有严格配对的对照组。

--------------------------------------------------------------------------------
【二、跑什么】

  门禁（约 24 分钟，串行 GPU0，68M，信息性，不阻塞主体）
    r11_68m_uni_s3   68M  无开关  seed=3   <- 补配对：对照 aq_s3=155.7261
    r11_68m_uni_s4   68M  无开关  seed=4   <- 补配对：对照 aq_s4=153.9589

  主实验（约 8 小时，双卡并行）
    r11_7b_aq_s2     7B   --adaptive_quota  seed=2  (GPU0, ./cache)
    r11_7b_aq_s3     7B   --adaptive_quota  seed=3  (GPU1, ./cache_seed3)

  只测 fake PPL（WikiText-2 + C4），零样本关闭，不跑 real_multibit。
  除 --adaptive_quota 与 seed 外，参数与第七~十轮逐字相同（addbit=1, 128x20, alpha=0.75）。

  预期对表（仅供核对，不是判据）：
    r11_7b_aq_s2 对照 Full seed2 = 7.9792 / 10.6792
    r11_7b_aq_s3 对照 Full seed3 = 8.0948 / 10.7126
    68M 的改善幅度是 -1.16% ~ -2.43%；7B 若落在 -0.5% ~ -2% 属正常量级。
    若 7B AQ 比 Full 差（数值更高），原样回传，不要解释。

--------------------------------------------------------------------------------
【三、怎么跑】

  1. 本脚本放到 SpikeLLM 根目录（和 main.py 同级）。本轮无 patch\ 目录。
  2. powershell -ExecutionPolicy Bypass -File .\lab_round11_runner.ps1 -Mode preflight
     powershell -ExecutionPolicy Bypass -File .\lab_round11_runner.ps1 -Mode run
  3. preflight 会检查三个代码文件的 SHA-256 是否为第十轮补丁后的哈希。
     若显示"旧哈希"，说明实验室代码被还原过——把第十轮 zip 再整包解压一次
     并重跑第十轮的 -Mode patch 即可，然后回到本脚本。
  4. 可断线，重跑 -Mode run 自动跳过已完成的腿。
  5. 回传：lab_round11_result.txt + log\r11_*\log_rank0_*.txt + env_round11_preflight.txt。
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

# 第十轮补丁后的期望哈希（SHA-256 前 16 位）
$ExpectedCode = @(
    @{ Rel = "main.py";                               Sha = "49485bc7ad4e9e0b" },
    @{ Rel = "spike_driven_quant\spike_linear.py";    Sha = "857be868ad9cc740" },
    @{ Rel = "spike_driven_quant\spike_omniquant.py"; Sha = "2b17e4fcfea74fce" }
)
# 若与下面这组相符，说明代码被还原到了第十轮之前
$OldCode = @(
    @{ Rel = "main.py";                               Sha = "6842f0bb479da6bc" },
    @{ Rel = "spike_driven_quant\spike_linear.py";    Sha = "f19df5382e798d07" },
    @{ Rel = "spike_driven_quant\spike_omniquant.py"; Sha = "18cf8d473bfc4795" }
)

$Jobs = @(
    @{ Tag = "r11_68m_uni_s3"; Model = $Model68M; Net = $Net68M; Seed = 3; Cache = "./cache"; Addbit = 1;
       Extra = ""; Gpu = 0; Group = "gate";
       Note = "68M uniform seed3，补配对（aq_s3 = 155.7261 / 89.0832）" },
    @{ Tag = "r11_68m_uni_s4"; Model = $Model68M; Net = $Net68M; Seed = 4; Cache = "./cache"; Addbit = 1;
       Extra = ""; Gpu = 0; Group = "gate";
       Note = "68M uniform seed4，补配对（aq_s4 = 153.9589 / 87.2476）" },
    @{ Tag = "r11_7b_aq_s2"; Model = $Model7B; Net = $Net7B; Seed = 2; Cache = "./cache"; Addbit = 1;
       Extra = "--adaptive_quota"; Gpu = 0; Group = "main";
       Note = "7B Stage2 seed2，对照 Full s2 = 7.9792 / 10.6792" },
    @{ Tag = "r11_7b_aq_s3"; Model = $Model7B; Net = $Net7B; Seed = 3; Cache = "./cache_seed3"; Addbit = 1;
       Extra = "--adaptive_quota"; Gpu = 1; Group = "main";
       Note = "7B Stage2 seed3，对照 Full s3 = 8.0948 / 10.7126" }
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

function Get-Sha16($path) {
    if (-not (Test-Path $path)) { return "MISSING" }
    return (Get-FileHash -Path $path -Algorithm SHA256).Hash.Substring(0, 16).ToLower()
}

function Prepare-Cache([string]$dir) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $src = Join-Path $Root "cache"
    if (Test-Path $src) {
        Get-ChildItem -Path $src -Filter "testloader_*.cache" -ErrorAction SilentlyContinue | ForEach-Object {
            $dst = Join-Path $dir $_.Name
            if (-not (Test-Path $dst)) { Copy-Item $_.FullName $dst -Force }
        }
    }
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

    Write-Output ("[{0}] seed={1}  extra=[{2}]  GPU{3}" -f $JobTag, $Seed, $ExtraStr, $Gpu)

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

function Get-LegPpl([string]$JobTag) {
    $dir = Join-Path $Root ("log\" + $JobTag)
    $f = Get-ChildItem -Path $dir -Filter "log_rank0_*.txt" -ErrorAction SilentlyContinue |
         Sort-Object LastWriteTime | Select-Object -Last 1
    if (-not $f) { return $null, $null }
    $w = (Select-String -Path $f.FullName -Pattern "INFO wikitext2 :" | Select-Object -Last 1)
    $c = (Select-String -Path $f.FullName -Pattern "INFO c4 :" | Select-Object -Last 1)
    $wv = if ($w) { [double](($w.Line -split ":")[-1].Trim()) } else { -1 }
    $cv = if ($c) { [double](($c.Line -split ":")[-1].Trim()) } else { -1 }
    return $wv, $cv
}

# ---------------------------------------------------------------- preflight
function Invoke-Preflight {
    Write-Step "第十一轮 preflight"
    $ok = $true

    Write-Host "1/7 代码状态（应为第十轮补丁后）"
    $allNew = $true
    $allOld = $true
    foreach ($e in $ExpectedCode) {
        $sha = Get-Sha16 (Join-Path $Root $e.Rel)
        $isOld = $false
        $old = $OldCode | Where-Object { $_.Rel -eq $e.Rel }
        if ($sha -eq $old.Sha) { $isOld = $true }
        if ($sha -ne $e.Sha) { $allNew = $false }
        if (-not $isOld) { $allOld = $false }
        Write-Host ("     {0}  {1}" -f $sha, $(if ($sha -eq $e.Sha) { "OK  " + $e.Rel } else { "!! " + $e.Rel }))
    }
    if (-not $allNew) {
        if ($allOld) {
            Write-Host "     三个文件全部是【旧哈希】：代码被还原过。" -ForegroundColor Yellow
            Write-Host "     修复：把第十轮 zip 整包解压到 SpikeLLM 根目录，运行其 -Mode patch，然后回来重跑本脚本 preflight。" -ForegroundColor Yellow
        } else {
            Write-Host "     哈希与新旧两套都对不上：代码被未知改动，停止。" -ForegroundColor Yellow
            Write-Host "     修复：回传 env 指纹给开发组，勿跑实验。" -ForegroundColor Yellow
        }
        $ok = $false
    }

    Write-Host "2/7 Python 与依赖"
    $v = & $Py -c "import torch,transformers,sys;print(sys.version.split()[0],torch.__version__,torch.version.cuda,transformers.__version__)"
    Write-Host "     python / torch / cuda / transformers = $v"
    if ($LASTEXITCODE -ne 0) { $ok = $false }

    Write-Host "3/7 模型权重"
    if (Test-Path $Model7B) { Write-Host "     7B  OK  $Model7B" }
    else { Write-Host "     7B  缺失 $Model7B" -ForegroundColor Yellow; $ok = $false }

    Write-Host "4/7 act_scales / act_shifts（--let 必需）"
    foreach ($n in @($Net7B, $Net68M)) {
        foreach ($sub in @("act_scales", "act_shifts")) {
            $p = Join-Path $Root "$sub\$n.pt"
            if (Test-Path $p) { Write-Host "     OK  $sub\$n.pt" }
            else { Write-Host "     缺失 $sub\$n.pt" -ForegroundColor Yellow; $ok = $false }
        }
    }

    Write-Host "5/7 缓存与种子目录"
    foreach ($f in @("dataloader_llama_wikitext2_128.cache", "testloader_llama_wikitext2_all.cache", "testloader_llama_c4_all.cache")) {
        $p = Join-Path $Root "cache\$f"
        if (Test-Path $p) { Write-Host "     OK  cache\$f" }
        else { Write-Host "     缺失 cache\$f" -ForegroundColor Yellow }
    }
    foreach ($sd in @("cache_seed3")) {
        if (Test-Path (Join-Path $Root $sd)) { Write-Host "     OK  $sd 已存在" }
        else { Write-Host "     $sd 不存在，run 时自动创建" }
    }

    Write-Host "6/7 磁盘余量"
    $drive = (Get-Item $Root).PSDrive
    $free = [Math]::Round((Get-PSDrive $drive.Name).Free / 1GB, 1)
    Write-Host ("     {0}: 剩余 {1} GB" -f $drive.Name, $free)
    if ($free -lt 40) { Write-Host "     余量偏低，建议先清理 log\ 下的旧 checkpoint" -ForegroundColor Yellow }

    Write-Host "7/7 环境指纹"
    if (Test-Path (Join-Path $Root "env_report.py")) {
        & $Py (Join-Path $Root "env_report.py") --out (Join-Path $Root "env_round11_preflight.txt") | Out-Null
        if (Test-Path (Join-Path $Root "env_round11_preflight.txt")) { Write-Host "     OK  env_round11_preflight.txt" }
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
    $out = Join-Path $Root "lab_round11_result.txt"
    $L = New-Object System.Collections.ArrayList
    [void]$L.Add("SpikeLLM / OSDSQ v2  第十一轮实验结果（实验室双 3090）")
    [void]$L.Add("生成时间: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    [void]$L.Add("项目路径: " + $Root)
    [void]$L.Add("任务: Stage 2 上 7B（关键一跳）+ 68M uniform seed3/4 配对补齐")
    [void]$L.Add("")
    [void]$L.Add("---- 参照（addbit=1, fake_binary, 128x20, alpha=0.75）----")
    [void]$L.Add("  7B Full  s2 7.9792 / 10.6792    s3 8.0948 / 10.7126    s4 8.0371 / 10.7135")
    [void]$L.Add("  7B Orig  10.5806 ± 0.0462      A-off 10.2818 ± 0.1353（第十轮）")
    [void]$L.Add("  68M uniform s2 157.7880 / 90.8312")
    [void]$L.Add("  68M aq  s2 155.9571 / 89.0309   s3 155.7261 / 89.0832   s4 153.9589 / 87.2476")
    [void]$L.Add("")

    foreach ($j in $Jobs) {
        $dir = Join-Path $Root ("log\" + $j.Tag)
        [void]$L.Add("### " + $j.Tag + "  (GPU" + $j.Gpu + ", seed=" + $j.Seed + ")")
        [void]$L.Add("  说明: " + $j.Note)
        $wv, $cv = Get-LegPpl $j.Tag
        if ($wv -ge 0) {
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

    [void]$L.Add("---- 配对读数（自动算，供三方核对）----")
    $w2, $c2 = Get-LegPpl "r11_7b_aq_s2"
    $w3, $c3 = Get-LegPpl "r11_7b_aq_s3"
    if ($w2 -ge 0) {
        [void]$L.Add(("  7B AQ s2 vs Full s2:  {0} vs 7.9792  ({1}%)" -f $w2, [Math]::Round((($w2 - 7.9792) / 7.9792 * 100), 2)))
    }
    if ($w3 -ge 0) {
        [void]$L.Add(("  7B AQ s3 vs Full s3:  {0} vs 8.0948  ({1}%)" -f $w3, [Math]::Round((($w3 - 8.0948) / 8.0948 * 100), 2)))
    }
    [void]$L.Add("")
    [void]$L.Add("---- 环境指纹 ----")
    if (Test-Path (Join-Path $Root "env_round11_preflight.txt")) {
        [void]$L.Add("  见 env_round11_preflight.txt（随日志一并回传）")
    } else {
        [void]$L.Add("  未生成。本轮结果按规程【不可入账】，请补跑：python env_report.py --out env_round11_preflight.txt")
    }

    Set-Content -Path $out -Value $L -Encoding UTF8
    Write-Host "结果已写入 $out"
}

# ---------------------------------------------------------------- 状态
function Invoke-Status {
    Write-Step "第十一轮状态"
    foreach ($j in $Jobs) {
        $dir = Join-Path $Root ("log\" + $j.Tag)
        $p = Test-Path (Join-Path $dir "done_ppl.flag")
        $d = Test-Path (Join-Path $dir "done.flag")
        $flag = if ($d) { "DONE" } elseif ($p) { "fake完成" } else { "未开始" }
        Write-Host ("  {0,-18} seed={1}  {2}" -f $j.Tag, $j.Seed, $flag)
    }
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
        Write-Step "只跑 68M uniform 配对腿（约 24 分钟）"
        foreach ($j in ($Jobs | Where-Object { $_.Group -eq "gate" })) {
            Invoke-TrainBody -Wd $Root -PyExe $Py -Model $j.Model -Net $j.Net -JobTag $j.Tag `
                -Seed $j.Seed -CacheDir $j.Cache -Gpu 0 -ExtraStr $j.Extra -Addbit $j.Addbit
        }
        New-ResultFile
    }

    "run" {
        Write-Step "开始执行（68M 配对 24 分钟 + 两条 7B Stage2 并行约 8 小时）"
        $start = Get-Date

        Prepare-Cache (Join-Path $Root "cache_seed3")
        Write-Host "[OK ] cache_seed3 就绪"

        Write-Host "步骤 1/2：68M uniform seed3/4（失败不阻塞主体）"
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
        Write-Host "步骤 2/2：两条 7B Stage2 腿并行（GPU0 = AQ seed2，GPU1 = AQ seed3）"
        $legs = @("r11_7b_aq_s2", "r11_7b_aq_s3")
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
        Get-Content (Join-Path $Root "lab_round11_result.txt") -Encoding UTF8 | Select-Object -First 60
    }
}
