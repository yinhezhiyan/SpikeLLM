<#
================================================================================
第十三轮实验执行器（实验室 Windows 单卡 RTX 3090）· 兼本轮说明文档
生成：WorkBuddy  2026-11-21
自定位：所有路径由 $PSScriptRoot 推导，脚本放在 SpikeLLM 根目录即可运行。
本轮性质：LLaMA-3-8B 跨模型验证（最后一块实验）。【无补丁】——实验室代码
          应已处于第十二轮补丁后状态（含 --quota_policy 三策略），preflight 会先验证。

--------------------------------------------------------------------------------
【零、为什么是这一轮（给实验室同学的背景，不用细读）】

论文方法（OSDSQ v2）在 LLaMA-2-7B 上已经完整闭环：
  Stage 1 量程隔离  7B 三种子降 24.04%（Orig 10.5806 -> Full 8.0370）
  Stage 2 等算力配额搜索  再降 5.52%（Full 8.0370 -> AQ 7.5932，三种子均值）
  必要性消融：adaptive(7.5296) < heuristic(7.8057) < random(7.9994)，同预算。
现在只差最后一个审稿人必问的问题：**这个方法在另一代大模型上还灵不灵？**
本轮在 LLaMA-3-8B 上跑同口径四腿，回答"跨模型通用性"。

--------------------------------------------------------------------------------
【一、本轮跑什么（四腿，全部单卡 GPU0 串行，seed=2，--nsamples 64）】

  腿 1  8B FP16 参考线        --wbits 16 --abits 16         （约 0.5 小时）
  腿 2  8B Original SpikeLLM  --orig_spikellm               （约 8 小时）
  腿 3  8B Stage 1 Full       无额外开关（只 --let 等）      （约 8 小时）
  腿 4  8B Stage 1+2 AQ       --adaptive_quota              （约 8 小时）

  统一口径（本轮铁律，来自 Antigravity 012 预检）：
    - --nsamples 64           （三腿统一 64，不能拿 7B 的 128 混比）
    - --cache_dir ./cache_llama3_8b   （【致命】必须独立缓存，见【二】）
    - --seed 2
    - 8B 只跑 fake PPL（--eval_ppl），【不跑零样本】——单卡 24GB 零样本易 OOM。

  总墙钟约 24~26 小时，可断线续跑（重跑 -Mode run 自动跳过已完成腿）。

--------------------------------------------------------------------------------
【二、必须遵守的硬约束（Antigravity 012 逐行审计结论）】

  1.【致命避坑：Tokenizer 缓存碰撞】
     main.py:285 用 args.net.split('-')[0] 拼缓存文件名。
     LLaMA-3-8B 与 LLaMA-2-7b-hf 的 family 前缀都是 "Llama"，
     若 8B 沿用 ./cache，会静默加载 LLaMA-2 的 32K 词表缓存喂给 8B 的 128K 词表，
     算出全乱码的假 PPL。→ 本轮所有腿强制 --cache_dir ./cache_llama3_8b。

  2.【前置依赖：先为 8B 生成 act_scales/act_shifts】
     --let 需要 act_scales/{net}.pt 和 act_shifts/{net}.pt。
     本轮 runner 在 -Mode run 第一步会自动调用 generate_act_scale_shift.py 生成
     （约 2~3 分钟），生成到 ./act_scales/Llama-3-8B.pt 和 ./act_shifts/Llama-3-8B.pt。

  3.【显存】
     校准阶段单层上卡约 7~8GB；PPL 评测整网 8B 上卡峰值约 20.2~21.5GB / 24GB。
     跑之前请确认 GPU0 无其他占用（nvidia-smi 看显存）。
     如极端情况 CrossEntropyLoss 触发瞬态 OOM，回传环境指纹即可，开发组会给 3 行兜底补丁。

  4.【--low_p 保持 0.9 不动】
     Antigravity 已核算：8B 的 GQA 8 个 KV 头会自动受 [0.80,0.98] 上界保护维持 0.900，
     与 TinyLlama-1.1B 行为一致，无需改默认值。

--------------------------------------------------------------------------------
【三、怎么跑】

  1. 本脚本放到 SpikeLLM 根目录（和 main.py 同级）。本轮无 patch\ 目录。
  2. 先预检：
       powershell -ExecutionPolicy Bypass -File .\lab_round13_runner.ps1 -Mode preflight
  3. 确认 GPU0 空闲后开始（会先自动生成 8B act_scale，再跑四腿）：
       powershell -ExecutionPolicy Bypass -File .\lab_round13_runner.ps1 -Mode run
     或分步单腿：
       -Mode actscale     -> 只生成 8B act_scales/act_shifts
       -Mode runleg -Tag r13_8b_fp16   （单腿，见下方 Tag 表）
  4. 可断线，重跑 -Mode run 自动跳过已完成腿。
  5. 回传：lab_round13_result.txt + log\r13_*\log_rank0_*.txt + env_round13_preflight.txt。

  Tag 对照：
    r13_8b_fp16    -> 8B FP16 参考
    r13_8b_orig    -> 8B Original SpikeLLM
    r13_8b_full    -> 8B Stage 1 Full
    r13_8b_aq      -> 8B Stage 1+2 AQ
================================================================================
#>

param(
    [ValidateSet("preflight", "run", "status", "runleg", "actscale")]
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

# ---- 模型与常量 ----
$Net8B     = "Llama-3-8B"
$Model8B   = Join-Path $ModelDir $Net8B
$Cache8B   = "./cache_llama3_8b"     # 【铁律】独立缓存，防 32K/128K 词表碰撞

# 第十二轮补丁后的期望哈希（SHA-256 前 16 位，本轮无新补丁，只做基线校验）
$ExpectedCode = @(
    @{ Rel = "main.py";                               Sha = "e898ff1da6b92e91" },
    @{ Rel = "spike_driven_quant\spike_linear.py";    Sha = "857be868ad9cc740" },
    @{ Rel = "spike_driven_quant\spike_omniquant.py"; Sha = "4f7ae925a3c1bd51" }
)

$Jobs = @(
    @{ Tag = "r13_8b_fp16"; Model = $Model8B; Net = $Net8B; Seed = 2; Cache = $Cache8B;
       Wbits = 16; Abits = 16; Addbit = 1; Extra = ""; Gpu = 0; Group = "main";
       Note = "8B FP16 参考线（无量化，对照 7B FP16=5.4722）" },
    @{ Tag = "r13_8b_orig"; Model = $Model8B; Net = $Net8B; Seed = 2; Cache = $Cache8B;
       Wbits = 4; Abits = 4; Addbit = 1; Extra = "--orig_spikellm"; Gpu = 0; Group = "main";
       Note = "8B Original SpikeLLM 基线" },
    @{ Tag = "r13_8b_full"; Model = $Model8B; Net = $Net8B; Seed = 2; Cache = $Cache8B;
       Wbits = 4; Abits = 4; Addbit = 1; Extra = ""; Gpu = 0; Group = "main";
       Note = "8B Stage 1 Full（量程隔离，无配额开关）" },
    @{ Tag = "r13_8b_aq";   Model = $Model8B; Net = $Net8B; Seed = 2; Cache = $Cache8B;
       Wbits = 4; Abits = 4; Addbit = 1; Extra = "--adaptive_quota"; Gpu = 0; Group = "main";
       Note = "8B Stage 1+2 AQ（量程隔离 + 等算力配额搜索）" }
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

# ---- 生成 8B act_scales / act_shifts ----
function Invoke-ActScale {
    $scales  = Join-Path $Root "act_scales\$Net8B.pt"
    $shifts  = Join-Path $Root "act_shifts\$Net8B.pt"
    $bothOk  = (Test-Path $scales) -and (Test-Path $shifts)
    if ($bothOk) {
        Write-Host "[actscale] 已存在，跳过："
        Write-Host "   $scales"
        Write-Host "   $shifts"
        return
    }
    Write-Step "生成 LLaMA-3-8B 的 act_scales / act_shifts（约 2~3 分钟）"
    Set-Location $Root
    $env:CUDA_VISIBLE_DEVICES = "0"
    $argv = @(
        "generate_act_scale_shift.py",
        "--model", $Model8B,
        "--scales-output-path", "./act_scales/",
        "--shifts-output-path", "./act_shifts/",
        "--calib_dataset", "wikitext2",
        "--num-samples", "64",
        "--seq-len", "2048",
        "--seed", "2"
    )
    & $Py @argv *>> "./log/r13_actscale.log"
    if ($LASTEXITCODE -ne 0) {
        throw "[actscale] 生成失败 (exit $LASTEXITCODE)，见 log/r13_actscale.log"
    }
    if (-not ((Test-Path $scales) -and (Test-Path $shifts))) {
        throw "[actscale] 未产出预期文件，请检查：$scales / $shifts"
    }
    Write-Host "[actscale] 完成"
}

# ---- 单腿执行体 ----
function Invoke-TrainBody {
    param($Wd, $PyExe, $Model, $Net, $JobTag, $Seed, $CacheDir, $Gpu, $ExtraStr, $Wbits, $Abits, $Addbit)

    if ([string]::IsNullOrWhiteSpace($ExtraStr)) { $Extra = @() }
    else { $Extra = $ExtraStr -split "\|" }

    Set-Location $Wd
    $env:CUDA_VISIBLE_DEVICES = "$Gpu"
    $dir = "./log/$JobTag"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null

    $base = @(
        "main.py", "--model", $Model, "--net", $Net,
        "--wbits", "$Wbits", "--abits", "$Abits", "--addbit", "$Addbit", "--low_p", "0.9",
        "--let", "--lwc", "--aug_loss",
        "--alpha", "0.75", "--let_lr", "1e-3", "--lwc_lr", "1e-2",
        "--nsamples", "64", "--epochs", "20", "--seed", "$Seed",
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

# ---- preflight ----
function Invoke-Preflight {
    Write-Step "第十三轮 preflight"
    $ok = $true

    Write-Host "1/8 代码状态（应为第十二轮补丁后）"
    foreach ($e in $ExpectedCode) {
        $sha = Get-Sha16 (Join-Path $Root $e.Rel)
        $good = ($sha -eq $e.Sha)
        if (-not $good) { $ok = $false }
        Write-Host ("     {0}  {1}" -f $sha, $(if ($good) { "OK  " + $e.Rel } else { "!! 与预期不符 " + $e.Rel }))
    }
    if (-not $ok) {
        Write-Host "     代码哈希与第十二轮补丁后状态不符。" -ForegroundColor Yellow
        Write-Host "     修复：把第十二轮 zip（lab_round12_quota_policy_2026-11-18.zip）整包解压到根目录，运行其 -Mode patch，再回来。" -ForegroundColor Yellow
    }

    Write-Host "2/8 Python 与依赖"
    $v = & $Py -c "import torch,transformers,sys;print(sys.version.split()[0],torch.__version__,torch.version.cuda,transformers.__version__)"
    Write-Host "     python / torch / cuda / transformers = $v"
    if ($LASTEXITCODE -ne 0) { $ok = $false }

    Write-Host "3/8 8B 模型权重"
    if (Test-Path $Model8B) { Write-Host "     8B  OK  $Model8B" }
    else { Write-Host "     8B  缺失 $Model8B" -ForegroundColor Yellow; $ok = $false }

    Write-Host "4/8 act_scales / act_shifts（--let 必需，缺失则 run 时自动生成）"
    foreach ($sub in @("act_scales", "act_shifts")) {
        $p = Join-Path $Root "$sub\$Net8B.pt"
        if (Test-Path $p) { Write-Host "     OK  $sub\$Net8B.pt" }
        else { Write-Host "     缺失 $sub\$Net8B.pt（run 时自动生成）" -ForegroundColor Yellow }
    }

    Write-Host "5/8 独立缓存目录（防词表碰撞）"
    if (Test-Path $Cache8B) { Write-Host "     OK  $Cache8B 已存在" }
    else { Write-Host "     $Cache8B 不存在，run 时自动创建（务必独立，勿复用 ./cache）" -ForegroundColor Yellow }

    Write-Host "6/8 GPU0 显存占用（应空闲）"
    $smi = & nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>$null
    Write-Host "     $smi"

    Write-Host "7/8 磁盘余量"
    $drive = (Get-Item $Root).PSDrive
    $free = [Math]::Round((Get-PSDrive $drive.Name).Free / 1GB, 1)
    Write-Host ("     {0}: 剩余 {1} GB" -f $drive.Name, $free)
    if ($free -lt 60) { Write-Host "     余量偏低，8B checkpoint 较大，建议先清理" -ForegroundColor Yellow }

    Write-Host "8/8 环境指纹"
    if (Test-Path (Join-Path $Root "env_report.py")) {
        & $Py (Join-Path $Root "env_report.py") --out (Join-Path $Root "env_round13_preflight.txt") | Out-Null
        if (Test-Path (Join-Path $Root "env_round13_preflight.txt")) { Write-Host "     OK  env_round13_preflight.txt" }
        else { Write-Host "     env_report.py 执行未产出文件" -ForegroundColor Yellow }
    } else {
        Write-Host "     env_report.py 不存在，跳过" -ForegroundColor Yellow
    }

    Write-Host ""
    if ($ok) { Write-Host "preflight 通过，可以 -Mode run（会先自动生成 8B act_scale）" -ForegroundColor Green }
    else { Write-Host "preflight 有缺失项，请先补齐再跑" -ForegroundColor Yellow }
}

# ---- 结果汇总 ----
function New-ResultFile {
    $out = Join-Path $Root "lab_round13_result.txt"
    $L = New-Object System.Collections.ArrayList
    [void]$L.Add("SpikeLLM / OSDSQ v2  第十三轮实验结果（实验室单卡 3090）")
    [void]$L.Add("生成时间: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    [void]$L.Add("项目路径: " + $Root)
    [void]$L.Add("任务: LLaMA-3-8B 跨模型验证（FP16/Orig/Stage1/AQ 四腿，nsamples=64，seed=2）")
    [void]$L.Add("")
    [void]$L.Add("---- 代码哈希（SHA-256 前 16 位）----")
    foreach ($e in $ExpectedCode) {
        $sha = Get-Sha16 (Join-Path $Root $e.Rel)
        [void]$L.Add(("  {0,-45} {1}  {2}" -f $e.Rel, $sha, $(if ($sha -eq $e.Sha) { "OK" } else { "!! 不符" })))
    }
    [void]$L.Add("")
    [void]$L.Add("---- 参照（7B, nsamples=128, 三种子均值，仅供对照量级）----")
    [void]$L.Add("  7B FP16 5.4722 / 6.9727")
    [void]$L.Add("  7B Orig 10.5806 / 13.7801")
    [void]$L.Add("  7B Stage1 Full 8.0370 / 10.7018")
    [void]$L.Add("  7B Stage1+2 AQ 7.5932 / 10.1320")
    [void]$L.Add("")

    foreach ($j in $Jobs) {
        $dir = Join-Path $Root ("log\" + $j.Tag)
        [void]$L.Add("### " + $j.Tag + "  (seed=" + $j.Seed + ")")
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

    [void]$L.Add("---- 配对读数（自动算，供三方核对；8B 本轮为单 seed，仅作量级参考）----")
    $wFp, $cFp = Get-LegPpl "r13_8b_fp16"
    $wOr, $cOr = Get-LegPpl "r13_8b_orig"
    $wFu, $cFu = Get-LegPpl "r13_8b_full"
    $wAq, $cAq = Get-LegPpl "r13_8b_aq"
    if ($wOr -ge 0 -and $wFu -ge 0) {
        [void]$L.Add(("  Stage1 vs Orig: {0} -> {1}  ({2}%)" -f $wOr, $wFu, [Math]::Round((($wFu - $wOr) / $wOr * 100), 2)))
    }
    if ($wFu -ge 0 -and $wAq -ge 0) {
        [void]$L.Add(("  Stage2(AQ) vs Stage1: {0} -> {1}  ({2}%)" -f $wFu, $wAq, [Math]::Round((($wAq - $wFu) / $wFu * 100), 2)))
    }
    if ($wOr -ge 0 -and $wAq -ge 0) {
        [void]$L.Add(("  AQ vs Orig 累计: {0} -> {1}  ({2}%)" -f $wOr, $wAq, [Math]::Round((($wAq - $wOr) / $wOr * 100), 2)))
    }
    [void]$L.Add("")
    [void]$L.Add("---- 环境指纹 ----")
    if (Test-Path (Join-Path $Root "env_round13_preflight.txt")) {
        [void]$L.Add("  见 env_round13_preflight.txt（随日志一并回传）")
    } else {
        [void]$L.Add("  未生成。本轮结果按规程【不可入账】，请补跑：python env_report.py --out env_round13_preflight.txt")
    }

    Set-Content -Path $out -Value $L -Encoding UTF8
    Write-Host "结果已写入 $out"
}

# ---- 状态 ----
function Invoke-Status {
    Write-Step "第十三轮状态"
    foreach ($j in $Jobs) {
        $dir = Join-Path $Root ("log\" + $j.Tag)
        $p = Test-Path (Join-Path $dir "done_ppl.flag")
        $d = Test-Path (Join-Path $dir "done.flag")
        $flag = if ($d) { "DONE" } elseif ($p) { "fake完成" } else { "未开始" }
        Write-Host ("  {0,-16} seed={1}  {2}" -f $j.Tag, $j.Seed, $flag)
    }
}

# ---- 主入口 ----
switch ($Mode) {

    "preflight" { Invoke-Preflight }

    "status" { Invoke-Status }

    "actscale" { Invoke-ActScale }

    "runleg" {
        $j = $Jobs | Where-Object { $_.Tag -eq $Tag }
        if (-not $j) { Fail "未知的 Tag: $Tag" }
        Invoke-TrainBody -Wd $Root -PyExe $Py -Model $j.Model -Net $j.Net -JobTag $j.Tag `
            -Seed $j.Seed -CacheDir $j.Cache -Gpu $j.Gpu -ExtraStr $j.Extra `
            -Wbits $j.Wbits -Abits $j.Abits -Addbit $j.Addbit
    }

    "run" {
        Write-Step "第十三轮：LLaMA-3-8B 跨模型（单卡 GPU0，约 24~26 小时，可断线续）"
        $start = Get-Date

        Write-Host "步骤 0：确认代码状态"
        $codeOk = $true
        foreach ($e in $ExpectedCode) {
            $sha = Get-Sha16 (Join-Path $Root $e.Rel)
            if ($sha -ne $e.Sha) { Write-Host ("  !! {0} 哈希不符（{1}）" -f $e.Rel, $sha) -ForegroundColor Red; $codeOk = $false }
        }
        if (-not $codeOk) { Fail "代码状态不对，请先 -Mode preflight 并按提示修复" }
        Write-Host "  代码状态 OK"

        Write-Host "步骤 1：生成 8B act_scales/act_shifts（若缺）"
        try { Invoke-ActScale } catch { Write-Host "[WARN] $($_.Exception.Message)" -ForegroundColor Yellow }

        Write-Host "步骤 2：四腿串行（FP16 -> Orig -> Full -> AQ）"
        foreach ($t in @("r13_8b_fp16", "r13_8b_orig", "r13_8b_full", "r13_8b_aq")) {
            $j = $Jobs | Where-Object { $_.Tag -eq $t }
            try {
                Invoke-TrainBody -Wd $Root -PyExe $Py -Model $j.Model -Net $j.Net -JobTag $j.Tag `
                    -Seed $j.Seed -CacheDir $j.Cache -Gpu 0 -ExtraStr $j.Extra `
                    -Wbits $j.Wbits -Abits $j.Abits -Addbit $j.Addbit
            } catch {
                Write-Host "[WARN] 腿 $($j.Tag) 失败：$_" -ForegroundColor Yellow
            }
            New-ResultFile
        }

        $elapsed = (Get-Date) - $start
        Write-Host ""
        Write-Host ("全部结束，用时 {0:N1} 小时" -f $elapsed.TotalHours)
        Write-Step "结果预览"
        Get-Content (Join-Path $Root "lab_round13_result.txt") -Encoding UTF8 | Select-Object -First 60
    }
}
