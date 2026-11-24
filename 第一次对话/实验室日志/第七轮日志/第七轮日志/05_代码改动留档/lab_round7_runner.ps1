<#
================================================================================
第七轮实验执行器（实验室 Windows 双 RTX 3090）· 兼本轮协作日志
生成：GLM  2026-11-02
脚本自定位：所有路径由 $PSScriptRoot 推导，不含任何开发机绝对路径。

================================================================================
【一、本轮唯一的科学问题：A 到底是不是"方法"，还是只是个实现选择】

前六轮证明了 OSDSQ v2 比原版好，但没有回答一个更基本的问题：
**必须用均值锚定吗？换成零填充（v1 的做法）会怎样？**

如果零填充与均值锚定效果接近，那么 A 只是一个等价实现，本文的核心创新点不成立，
整篇论文的骨架要重搭。这个问题必须在本轮回答，不能拖到写论文时才发现。

到目前为止，`--shield_mode zero` 的代码已经加了，但**一次都没跑过**。
本脚本就是把这块空白补上。

================================================================================
【二、A 的工作原理（一句话版，供日志留档）】

原版 SpikeLLM 把通道按显著性分两路：90% 普通通道走 4-bit，10% 离群通道走高比特脉冲。
但给 4-bit 分支算动态量程 [xmin, xmax] 时，原版把**完整张量**丢进 act_quantizer，
那 10% 离群值仍然参与 min/max，把 90% 普通通道的分辨率压扁。

A（均值锚定）= 算 4-bit 量程前，先用「低显著子集的真实均值」填充被分流的通道。
因为 min(S) <= mean(S) <= max(S)，填充后子集 [xmin,xmax] 严格不变
（等价于精确子集 min-max），但不会产生 inf/nan 梯度，且零超参数。

`--shield_mode zero` = v1 做法：用 0 填充。当子集全为正时，0 会把 min 拉到 0，
量程被虚假扩大（合成实验实测 5.16 倍）。这是我们要在真实模型上验证的缺陷。

注意：本轮的 zero 只作用于**线性层（A）**，KV 路径（B）仍保持 mean 锚定，
因此这是单变量对照，不引入混淆。

================================================================================
【三、截至第六轮的既有结果（fake_binary_simulate，128x20，alpha=0.75，let_lr=1e-3）】

  FP16                        5.4722 / 6.9727
  Original  seed2            10.5726 / 13.7738     real 10.8108 / 14.1855
  Original  seed3            10.6303 / 13.7965     real 10.8963 / 14.4792
  Original  seed4            10.5389 / 13.7700     real 10.7599 / 14.3417
  OSDSQ v2  seed2             7.9792 / 10.6792     real  8.1570 / 10.8975
  OSDSQ v2  seed3             8.0948 / 10.7126     real  8.1234 / 10.9222
  A-off     seed2            10.4181 / 12.9691     real 10.5408 / 13.5234

  配对降幅：seed2 24.53% / seed3 23.85%（wt2），22.47% / 22.35%（c4）
  Original 三种子 wt2 的 std 仅 0.44% —— 降幅比种子噪声稳得多，主结果成立。

  关键推论：A-off（10.4181）与 Original（10.5726）只差 1.46%，
  即 B+C+D 三个组件合计仅贡献 1.5%，A 独自承担 23.4 个百分点。
  → 本轮起，论文贡献收敛到 A 一件事，B/C/D 降级为工程细节。

================================================================================
【四、本轮任务（约 8 ~ 9 小时）】

  步骤 1（前置，约 10 分钟，仅当 68M 模型存在时执行）
    68M `--shield_mode zero`，seed2
    目的：花 10 分钟先看一眼方向。68M 的 mean 基线为 156.84 / 90.53（seed2）。
    若 zero 明显更差 → A 的必要性在开发机规模上先立住，7B 值得跑。
    若 zero 与 mean 接近 → 仍跑 7B（小模型结论未必外推），但请在结果里标红。

  GPU0（约 8 小时）  7b_seed2_zero   seed=2, cache=./cache, --shield_mode zero
    目的：**本轮最重要的一条腿**。与已有 mean 结果（7.9792 / 10.6792）同配置配对，
          直接回答"A 是不是必须"。fake + real 两种模式都跑。

  GPU1（约 8 小时）  7b_seed4_ours   seed=4, cache=./cache_seed4
    目的：补第三组种子，让主结果拥有 3 组配对的 mean±std（审稿人必问）。
          与已有 Original seed4（10.5389 / 13.7700）配对。

================================================================================
【五、判据（脚本会自动算，但仍请人工核对）】

  7B zero  vs  7B mean(7.9792)：
    差 >= 1.0 PPL  → A 的必要性成立，核心创新点站得住
    差 0.3 ~ 1.0   → 弱证据，需要在论文里补更多论证（如逐层量程统计）
    差 < 0.3       → **核心创新点不成立**，请立刻通知开发组，不要继续排后续实验

  7B seed4-Ours vs seed4-Original(10.5389)：
    预期降幅 23% ~ 25%。若偏离超过 3 个百分点，说明种子间波动比预期大，
    三组种子的误差棒需要重算。

================================================================================
【六、本轮不做什么】

  1. 不改任何算法代码。本轮只加参数、只跑对照。
  2. 不跑 8B。权重虽已就位（16.06 GB，4 分片 sha256 全部 MATCH），
     但 7B 的 PIQA 峰值已达 22.5/24 GB，8B 首跑必然 OOM，需单独排期并降 nsamples。
  3. 不跑零样本（PIQA/ARC 等）。单腿会从 8 小时涨到 14~17 小时，本轮时间不够。
  4. 不动 shield_mode 的默认值（仍是 mean），不改变任何现有默认数值路径。

================================================================================
【七、给实验室的三条提醒（前几轮踩过的坑）】

  1. **安全删除守卫会杀进程**：表现为"无 traceback、exit 1、日志停在某一行"。
     遇到后先别重跑，先核对已落盘文件的字节数与 sha256，能就地提交就别重下。
  2. runner 可能在最后一条腿完成瞬间就写结果 txt，**必须事后核对数字行是否齐全**。
     缺行用 -Mode run 再跑一次即可（有 done.flag 的作业会自动跳过，秒级）。
  3. 给这台机器写 .ps1 一律用 UTF-8 BOM，否则 PS 5.1 会按 ANSI 解码中文注释导致解析错位。

================================================================================
【八、使用方法】

  # 1) 环境体检（不跑训练，30 秒）
  powershell -ExecutionPolicy Bypass -File .\lab_round7_runner.ps1 -Mode preflight

  # 2) 正式执行（约 8~9 小时，可断线，重跑会自动续）
  powershell -ExecutionPolicy Bypass -File .\lab_round7_runner.ps1 -Mode run

  # 3) 查看进度
  powershell -ExecutionPolicy Bypass -File .\lab_round7_runner.ps1 -Mode status

  # 4) 只想先看 68M 的方向（10 分钟，不启动 7B）
  powershell -ExecutionPolicy Bypass -File .\lab_round7_runner.ps1 -Mode smoke

  产出：lab_round7_result.txt（纯文本，随日志一起回传）
================================================================================
#>

param(
    [ValidateSet("preflight", "run", "smoke", "status", "runleg")]
    [string]$Mode = "status",
    [string]$Tag = "",
    [string]$Py = "",
    [string]$ModelDir = "",
    [switch]$Skip68M
)

# NOTE (lab machine fix 2026-11-02): PS 5.1 turns a native command's stderr line
# into a terminating error when the preference is Stop, which kills the python child.
# Rounds 5 and 6 both run with Continue plus explicit $LASTEXITCODE checks.
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

$Net7B   = "Llama-2-7b-hf"
$Model7B = Join-Path $ModelDir $Net7B
$Net68M  = "llama-68m"
$Model68M = "JackFram/llama-68m"

# 两条 7B 腿 + 可选 68M 门禁
$Jobs = @(
    @{ Tag = "r7_68m_zero";        Model = $Model68M; Net = $Net68M; Seed = 2; Cache = "./cache";
       Extra = "--shield_mode|zero"; Gpu = 0; Need = "68m";
       Note = "68M 零填充对照（10min 门禁），对照 mean 基线 156.84 / 90.53" },
    @{ Tag = "r7_7b_seed2_zero";   Model = $Model7B; Net = $Net7B; Seed = 2; Cache = "./cache";
       Extra = "--shield_mode|zero"; Gpu = 0; Need = "7b";
       Note = "7B 零填充对照（核心），对照 mean 基线 7.9792 / 10.6792" },
    @{ Tag = "r7_7b_seed4_ours";   Model = $Model7B; Net = $Net7B; Seed = 4; Cache = "./cache_seed4";
       Extra = ""; Gpu = 1; Need = "7b";
       Note = "seed4-OSDSQv2，与 Original seed4 (10.5389/13.7700) 配第三组种子" }
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

function Get-Md5($path) {
    if (-not (Test-Path $path)) { return "MISSING" }
    return (Get-FileHash -Path $path -Algorithm MD5).Hash.Substring(0, 16).ToLower()
}

# ---------------------------------------------------------------- 单腿执行体
function Invoke-TrainBody {
    param($Wd, $PyExe, $Model, $Net, $JobTag, $Seed, $CacheDir, $Gpu, $ExtraStr)

    if ([string]::IsNullOrWhiteSpace($ExtraStr)) { $Extra = @() }
    else { $Extra = $ExtraStr -split "\|" }

    Set-Location $Wd
    $env:CUDA_VISIBLE_DEVICES = "$Gpu"
    $dir = "./log/$JobTag"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null

    $base = @(
        "main.py", "--model", $Model, "--net", $Net,
        "--wbits", "4", "--abits", "4", "--addbit", "1", "--low_p", "0.9",
        "--let", "--lwc", "--aug_loss",
        "--alpha", "0.75", "--let_lr", "1e-3", "--lwc_lr", "1e-2",
        "--nsamples", "128", "--epochs", "20", "--seed", "$Seed",
        "--cache_dir", $CacheDir
    )
    if ($Extra) { $base += $Extra }

    if (-not (Test-Path "$dir/done_ppl.flag")) {
        Write-Output "[$JobTag] fake PPL 开始（GPU$Gpu）"
        $argv = $base + @("--eval_ppl", "--output_dir", $dir)
        & $PyExe @argv *>> "$dir/run.log"
        if ($LASTEXITCODE -ne 0) {
            Move-Item "$dir/run.log" "$dir/run.log.failed" -Force -ErrorAction SilentlyContinue
            throw "[$JobTag] fake PPL 失败 (exit $LASTEXITCODE)，见 $dir/run.log.failed"
        }
        New-Item -ItemType File -Force -Path "$dir/done_ppl.flag" | Out-Null
    } else { Write-Output "[$JobTag] done_ppl 已存在，跳过" }

    if (-not (Test-Path "$dir/done_multibit.flag")) {
        Write-Output "[$JobTag] real_multibit 开始"
        $argv2 = $base + @("--epochs", "0", "--resume", "$dir/omni_parameters.pth",
                           "--sim_mode", "real_multibit_simulate", "--eval_ppl",
                           "--output_dir", "$dir/real_multibit")
        & $PyExe @argv2 *>> "$dir/run.log"
        if ($LASTEXITCODE -ne 0) { throw "[$JobTag] real_multibit 失败 (exit $LASTEXITCODE)" }
        New-Item -ItemType File -Force -Path "$dir/done_multibit.flag" | Out-Null
    } else { Write-Output "[$JobTag] done_multibit 已存在，跳过" }

    New-Item -ItemType File -Force -Path "$dir/done.flag" | Out-Null
    Write-Output "[$JobTag] 完成"
}

# ---------------------------------------------------------------- preflight
function Invoke-Preflight {
    Write-Step "第七轮 preflight"
    $ok = $true

    Write-Host "1/6 Python 与依赖"
    try {
        $v = & $Py -c "import torch,transformers,sys;print(sys.version.split()[0],torch.__version__,torch.version.cuda,transformers.__version__)"
        Write-Host "     python / torch / cuda / transformers = $v"
    } catch {
        Fail "Python 环境不可用：$Py"
    }
    if (-not (Test-Path $Py)) { Write-Host "     [WARN] 指定解释器不存在，回退到 PATH 中的 python" -ForegroundColor Yellow }

    Write-Host "2/6 代码版本（与开发机比对）"
    $want = @{
        "spike_driven_quant\spike_linear.py" = "26e84e8f1a460d43a"
        "main.py"                            = "f201ca3a429f0e168"
    }
    foreach ($k in $want.Keys) {
        $h = Get-Md5 (Join-Path $Root $k)
        if ($h -eq "MISSING") { Write-Host "     [FAIL] $k 缺失" -ForegroundColor Red; $ok = $false }
        elseif ($h.StartsWith($want[$k].Substring(0, 12))) { Write-Host "     [OK ] $k  $h" }
        else { Write-Host "     [WARN] $k 哈希 $h 与第六轮不一致（期望 $($want[$k])）" -ForegroundColor Yellow }
    }

    Write-Host "3/6 7B 模型与激活统计"
    if (-not (Test-Path (Join-Path $Model7B "config.json"))) {
        Write-Host "     [FAIL] $Model7B\config.json 不存在" -ForegroundColor Red; $ok = $false
    } else { Write-Host "     [OK ] $Model7B" }
    foreach ($f in @("act_scales\$Net7B.pt", "act_shifts\$Net7B.pt")) {
        if (Test-Path (Join-Path $Root $f)) { Write-Host "     [OK ] $f" }
        else { Write-Host "     [FAIL] $f 缺失（LET 必需）" -ForegroundColor Red; $ok = $false }
    }

    Write-Host "4/6 68M 门禁资源（缺失不影响 7B）"
    $has68 = $true
    foreach ($f in @("act_scales\$Net68M.pt", "act_shifts\$Net68M.pt")) {
        if (Test-Path (Join-Path $Root $f)) { Write-Host "     [OK ] $f" }
        else { Write-Host "     [SKIP] $f 缺失 -> 将跳过 68M 门禁" -ForegroundColor Yellow; $has68 = $false }
    }
    if ($has68) {
        Write-Host "     [OK ] 68M 模型 $Model68M 首次使用会自动下载（约 270 MB）"
        Write-Host "           若 HF 不可达，请加 -Skip68M 直接跑 7B"
    }

    Write-Host "5/6 GPU 与磁盘"
    $smi = & nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader 2>$null
    if ($smi) { $smi | ForEach-Object { Write-Host "     $_" } }
    else { Write-Host "     [WARN] nvidia-smi 不可用" -ForegroundColor Yellow }
    $d = Get-PSDrive ($Root.Substring(0, 1))
    Write-Host ("     磁盘剩余 {0:N1} GB" -f ($d.Free / 1GB))
    if ($d.Free / 1GB -lt 30) { Write-Host "     [WARN] 磁盘不足 30 GB" -ForegroundColor Yellow }

    Write-Host "6/6 环境指纹（env_report.py）"
    if (Test-Path (Join-Path $Root "env_report.py")) {
        & $Py "env_report.py" --note "round7_preflight" --out (Join-Path $Root "env_round7_preflight.txt") | Out-Null
        Write-Host "     [OK ] env_round7_preflight.txt 已生成"
    } else {
        Write-Host "     [WARN] env_report.py 不存在，跳过（可从开发机补发）" -ForegroundColor Yellow
    }

    Write-Host ""
    if ($ok) { Write-Host "preflight 通过，可以执行 -Mode run" -ForegroundColor Green }
    else { Write-Host "preflight 未通过，请先修复上面的 [FAIL]" -ForegroundColor Red }
    Write-Host ""
}

# ---------------------------------------------------------------- status
function Invoke-Status {
    Write-Step "第七轮进度"
    foreach ($j in $Jobs) {
        $dir = "./log/$($j.Tag)"
        $p = Test-Path "$dir/done_ppl.flag"
        $m = Test-Path "$dir/done_multibit.flag"
        $d = Test-Path "$dir/done.flag"
        $flag = if ($d) { "DONE" } elseif ($p) { "fake完成/real进行中" } else { "未开始" }
        Write-Host ("  {0,-20} {1}" -f $j.Tag, $flag)
    }
    Write-Host ""
    Write-Host "  已有结果（若存在）:"
    $txt = Join-Path $Root "lab_round7_result.txt"
    if (Test-Path $txt) { Get-Content $txt -Encoding UTF8 | Select-Object -Last 30 }
    else { Write-Host "    （尚未生成）" }
}

# ---------------------------------------------------------------- 结果汇总
function New-ResultFile {
    $out = Join-Path $Root "lab_round7_result.txt"
    $L = New-Object System.Collections.ArrayList
    [void]$L.Add("SpikeLLM / OSDSQ v2  第七轮实验结果（实验室双 3090）")
    [void]$L.Add("生成时间: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    [void]$L.Add("项目路径: " + $Root)
    [void]$L.Add("")
    [void]$L.Add("---- 既有 7B 参照（同协议：fake_binary, 128x20, alpha=0.75, let_lr=1e-3）----")
    [void]$L.Add("  FP16                5.4722 / 6.9727")
    [void]$L.Add("  Original  seed2    10.5726 / 13.7738    real 10.8108 / 14.1855")
    [void]$L.Add("  Original  seed3    10.6303 / 13.7965    real 10.8963 / 14.4792")
    [void]$L.Add("  Original  seed4    10.5389 / 13.7700    real 10.7599 / 14.3417")
    [void]$L.Add("  OSDSQ v2  seed2     7.9792 / 10.6792    real  8.1570 / 10.8975   <- mean 基线")
    [void]$L.Add("  OSDSQ v2  seed3     8.0948 / 10.7126    real  8.1234 / 10.9222")
    [void]$L.Add("  A-off     seed2    10.4181 / 12.9691    real 10.5408 / 13.5234")
    [void]$L.Add("")
    [void]$L.Add("---- 68M 参照（mean 基线, seed2）----")
    [void]$L.Add("  v2 mean             156.84 / 90.53")
    [void]$L.Add("")

    foreach ($j in $Jobs) {
        $dir = Join-Path $Root ("log\" + $j.Tag)
        [void]$L.Add("### " + $j.Tag + "  (GPU" + $j.Gpu + ", seed=" + $j.Seed + ")")
        [void]$L.Add("  说明: " + $j.Note)
        foreach ($pair in @(@("fake", $dir), @("real", (Join-Path $dir "real_multibit")))) {
            $label = $pair[0]
            $d = $pair[1]
            if (-not (Test-Path $d)) { continue }
            $f = Get-ChildItem -Path $d -Filter "log_rank0_*.txt" -ErrorAction SilentlyContinue |
                 Sort-Object LastWriteTime | Select-Object -Last 1
            if (-not $f) { continue }
            $w = (Select-String -Path $f.FullName -Pattern "INFO wikitext2 :" | Select-Object -Last 1)
            $c = (Select-String -Path $f.FullName -Pattern "INFO c4 :" | Select-Object -Last 1)
            $wv = if ($w) { ($w.Line -split ":")[-1].Trim() } else { "NA" }
            $cv = if ($c) { ($c.Line -split ":")[-1].Trim() } else { "NA" }
            [void]$L.Add("  [" + $label + "] wikitext2 : " + $wv)
            [void]$L.Add("  [" + $label + "] c4 : " + $cv)
        }
        $nan = 0
        $watch = Join-Path $dir "run.log"
        if (Test-Path $watch) {
            $nan = (Select-String -Path $watch -Pattern "norm:nan" -ErrorAction SilentlyContinue | Measure-Object).Count
        }
        [void]$L.Add("  norm:nan 次数: " + $nan)
        [void]$L.Add("")
    }

    [void]$L.Add("---- 判据（请人工核对）----")
    [void]$L.Add("  7B zero vs mean(7.9792):")
    [void]$L.Add("    >= 1.0  -> A 的必要性成立，核心创新点站得住")
    [void]$L.Add("    0.3~1.0 -> 弱证据，需补逐层量程统计")
    [void]$L.Add("    < 0.3   -> 核心创新点不成立，立刻通知开发组")
    [void]$L.Add("  7B seed4-Ours vs Original seed4(10.5389): 预期降幅 23%~25%")
    [void]$L.Add("  68M zero vs mean(156.84): 预期 zero 更高；若接近，说明小模型上 A 不敏感")
    [void]$L.Add("")
    [void]$L.Add("---- 环境指纹 ----")
    $envf = Join-Path $Root "env_round7_preflight.txt"
    if (Test-Path $envf) {
        [void]$L.Add("  见 env_round7_preflight.txt（随日志一并回传）")
    } else {
        [void]$L.Add("  未生成（env_report.py 可能缺失），请补跑：python env_report.py --out env_round7.txt")
    }

    Set-Content -Path $out -Value $L -Encoding UTF8
    Write-Host "结果已写入 $out"
}

# ---------------------------------------------------------------- 主流程
switch ($Mode) {

    "preflight" { Invoke-Preflight }

    "status" { Invoke-Status }

    "runleg" {
        $j = $Jobs | Where-Object { $_.Tag -eq $Tag }
        if (-not $j) { Fail "未知的 Tag: $Tag" }
        Invoke-TrainBody -Wd $Root -PyExe $Py -Model $j.Model -Net $j.Net -JobTag $j.Tag `
            -Seed $j.Seed -CacheDir $j.Cache -Gpu $j.Gpu -ExtraStr $j.Extra
    }

    "smoke" {
        Write-Step "68M 门禁（仅 68M，不启动 7B）"
        $j = $Jobs | Where-Object { $_.Tag -eq "r7_68m_zero" }
        foreach ($f in @("act_scales\$Net68M.pt", "act_shifts\$Net68M.pt")) {
            if (-not (Test-Path (Join-Path $Root $f))) {
                Fail "缺少 $f。请把开发机的 act_scales/act_shifts 下的 llama-68m.pt 复制过来（各 70 KB）"
            }
        }
        Invoke-TrainBody -Wd $Root -PyExe $Py -Model $j.Model -Net $j.Net -JobTag $j.Tag `
            -Seed $j.Seed -CacheDir $j.Cache -Gpu 0 -ExtraStr $j.Extra
        New-ResultFile
    }

    "run" {
        Write-Step "开始执行（预计 8~9 小时，可断线，重跑自动续）"
        $start = Get-Date

        # 步骤 1：68M 门禁（阻塞，约 10 分钟）
        $do68 = (-not $Skip68M) -and
                (Test-Path (Join-Path $Root "act_scales\$Net68M.pt")) -and
                (Test-Path (Join-Path $Root "act_shifts\$Net68M.pt"))
        if ($do68) {
            Write-Host "步骤 1/2：68M 门禁（约 10 分钟）"
            try {
                $j0 = $Jobs | Where-Object { $_.Tag -eq "r7_68m_zero" }
                Invoke-TrainBody -Wd $Root -PyExe $Py -Model $j0.Model -Net $j0.Net -JobTag $j0.Tag `
                    -Seed $j0.Seed -CacheDir $j0.Cache -Gpu 0 -ExtraStr $j0.Extra
            } catch {
                Write-Host "[WARN] 68M 门禁失败，不阻塞 7B：$_" -ForegroundColor Yellow
            }
            New-ResultFile
        } else {
            Write-Host "步骤 1/2：跳过 68M 门禁（模型或 act_scales 缺失，或指定了 -Skip68M）"
        }

        # 步骤 2：两条 7B 腿并行
        Write-Host ""
        Write-Host "步骤 2/2：两条 7B 腿并行（GPU0 = zero 对照，GPU1 = seed4-Ours）"
        $legs = @("r7_7b_seed2_zero", "r7_7b_seed4_ours")
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
        Write-Host ""
        Write-Host "两条腿已启动，等待完成……（可关闭窗口，用 -Mode run 重跑会续）"
        foreach ($p in $procs) { $p.WaitForExit() }

        $elapsed = (Get-Date) - $start
        Write-Host ""
        Write-Host ("全部结束，用时 {0:N1} 小时" -f $elapsed.TotalHours)
        New-ResultFile

        Write-Step "结果预览"
        Get-Content (Join-Path $Root "lab_round7_result.txt") -Encoding UTF8 |
            Select-Object -First 60
    }
}
