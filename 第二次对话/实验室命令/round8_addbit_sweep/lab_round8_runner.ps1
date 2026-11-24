<#
================================================================================
第八轮实验执行器（实验室 Windows 双 RTX 3090）· 兼本轮说明文档
生成：WorkBuddy  2026-11-04
自定位：所有路径由 $PSScriptRoot 推导，脚本放在 SpikeLLM 根目录即可运行。

--------------------------------------------------------------------------------
【一、本轮唯一要回答的问题：addbit 是不是一个廉价的替代品】

前七轮的全部结果都在同一个配置上：--addbit 1，即显著通道只多 1 bit。
而 ICLR 2026 的 phase-coding 论文（相关文件/9103_...pdf，Table 1）报告：
在不改任何方法、只把 SpikeLLM 的时间步调到 T=8 时，LLaMA-2-7B 的
WikiText-2 PPL 是 5.86（FP16 为 5.47）。我们复现的 Original 是 10.57，那是 T=2。

也就是说：我们的 24% 提升，是在一个"把显著通道压在 +1 bit"的极端便宜配置上
取得的。审稿人一定会问一句："为什么不直接把 addbit 调大？"

按我们自己的 ACE 公式（measure_ace_sops.py，按分数位宽计），
显著通道只占 10%，把它从 +1 bit 提到 +3 bit 只多花 5% 的能量（16.4 -> 17.2）。
如果这 5% 就能把 PPL 从 10.57 打到 6 附近，那么我们"零开销 24%"的整条论证
就会被一句话击穿。

这轮就是去把这句话堵上（或者提前发现堵不上）。

--------------------------------------------------------------------------------
【二、跑什么】

  门禁（约 20 分钟，先看方向）
    r8_68m_orig_a3   LLaMA-68M  --orig_spikellm --addbit 3   (T=8)
    r8_68m_ours_a3   LLaMA-68M                --addbit 3   (T=8)
    参照：68M 的 T=2 基线  Orig 172.29~175.14 / v2 156.84

  主实验（约 7~8 小时，双卡并行）
    r8_7b_orig_a3    LLaMA-2-7B  --orig_spikellm --addbit 3
    r8_7b_ours_a3    LLaMA-2-7B                --addbit 3
    参照：7B 的 T=2 基线  Orig 10.5726 / Ours 7.9792 / FP16 5.4722

  只测 fake PPL（WikiText-2 + C4），零样本全部关闭，不跑 real_multibit。
  除 --addbit 外，其余参数与第七轮逐字相同（128x20, alpha=0.75, let_lr=1e-3）。

  一句澄清（Codex 2026-11-04 复核提出，已在代码里核实）：
  addbit 同时决定两件事——act_quantizer_high 的位宽（n_bits += addbit）与 steps = 2^addbit。
  但 fake_binary_simulate 模式的前向只用到前者，steps 只在两个 multibit 分支里起作用。
  所以本轮的 fake PPL 是一次干净的"高显著分支位宽"对照；T=8 是同一件事在真实脉冲路径
  里的表现（8 个时间步 ≈ 3 个额外比特）。两者不是两个变量，是同一预算的两种记法。

--------------------------------------------------------------------------------
【三、怎么判读（回传后三方一起看，这里先写死判据，避免事后解释）】

  记 A = 7B Orig@a3 的 WikiText-2，B = 7B Ours@a3 的 WikiText-2。

  A >= 9.0        -> 原叙事成立：调大 addbit 救不回来，我们的方法有独立价值。
                     addbit 那条只需在正文放一行脚注说明"已对照，无效"。
  7.0 <= A < 9.0  -> 部分替代：必须画 T-PPL 前沿图，两条曲线比较，不能只报单点。
  A < 7.0         -> 严重：24% 只是在打一个没人用的配置，论文必须整体改成前沿叙事，
                     并重新论证"在每个等能量点上我们都更好"。

  B 与 A 的差：
    B 明显小于 A（>5%）  -> 方法在高 T 下依然有效，前沿整体外推，这是最好的结果。
    B 与 A 接近（<2%）   -> 方法只在中低 T 有效，需在正文限定适用范围。

--------------------------------------------------------------------------------
【四、怎么跑】

  1. 把本脚本放到 SpikeLLM 根目录（和 main.py 同级）。
  2. 右键"使用 PowerShell 运行"，或：
       powershell -ExecutionPolicy Bypass -File .\lab_round8_runner.ps1 -Mode preflight
       powershell -ExecutionPolicy Bypass -File .\lab_round8_runner.ps1 -Mode run
  3. 可断线。重跑 -Mode run 会自动跳过已完成（done_ppl.flag）的腿。
  4. 全部结束后，把 lab_round8_result.txt + log\r8_*\ 下的 log_rank0_*.txt
     + env_round8_preflight.txt 一起打包回传。

  注意：$ErrorActionPreference 固定为 Continue。python 往 stderr 打 warning
  是正常现象，不要因为看到红字就中断。
================================================================================
#>

param(
    [ValidateSet("preflight", "run", "gate", "status", "runleg")]
    [string]$Mode = "status",
    [string]$Tag = "",
    [string]$Py = "",
    [string]$ModelDir = "",
    [int]$AddbitOverride = 0
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
    @{ Tag = "r8_68m_orig_a3"; Model = $Model68M; Net = $Net68M; Seed = 2; Cache = "./cache"; Addbit = 3;
       Extra = "--orig_spikellm"; Gpu = 0; Group = "gate";
       Note = "68M 原版 +3bit（T=8）门禁，对照 Orig@T2 = 172.29~175.14" },
    @{ Tag = "r8_68m_ours_a3"; Model = $Model68M; Net = $Net68M; Seed = 2; Cache = "./cache"; Addbit = 3;
       Extra = ""; Gpu = 0; Group = "gate";
       Note = "68M OSDSQv2 +3bit（T=8）门禁，对照 v2@T2 = 156.84" },
    @{ Tag = "r8_7b_orig_a3"; Model = $Model7B; Net = $Net7B; Seed = 2; Cache = "./cache"; Addbit = 3;
       Extra = "--orig_spikellm"; Gpu = 0; Group = "main";
       Note = "7B 原版 +3bit（T=8）核心对照，对照 Orig@T2 = 10.5726" },
    @{ Tag = "r8_7b_ours_a3"; Model = $Model7B; Net = $Net7B; Seed = 2; Cache = "./cache"; Addbit = 3;
       Extra = ""; Gpu = 1; Group = "main";
       Note = "7B OSDSQv2 +3bit（T=8），对照 Ours@T2 = 7.9792 与 FP16 5.4722" }
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

    $ab = $Addbit
    if ($script:AddbitOverride -gt 0) { $ab = $script:AddbitOverride }

    Set-Location $Wd
    $env:CUDA_VISIBLE_DEVICES = "$Gpu"
    $dir = "./log/$JobTag"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null

    $base = @(
        "main.py", "--model", $Model, "--net", $Net,
        "--wbits", "4", "--abits", "4", "--addbit", "$ab", "--low_p", "0.9",
        "--let", "--lwc", "--aug_loss",
        "--alpha", "0.75", "--let_lr", "1e-3", "--lwc_lr", "1e-2",
        "--nsamples", "128", "--epochs", "20", "--seed", "$Seed",
        "--cache_dir", $CacheDir
    )
    if ($Extra) { $base += $Extra }

    Write-Output ("[{0}] addbit={1}  fake=高显著{2}-bit / 真实脉冲 T={3}  GPU{4}" -f $JobTag, $ab, (4 + $ab), [Math]::Pow(2, $ab), $Gpu)

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
    Write-Step "第八轮 preflight"
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

    Write-Host "4/6 缓存（7B 的 cache_dir = ./cache）"
    foreach ($f in @("dataloader_llama_wikitext2_128.cache", "testloader_llama_wikitext2_all.cache", "testloader_llama_c4_all.cache")) {
        $p = Join-Path $Root "cache\$f"
        if (Test-Path $p) { Write-Host "     OK  cache\$f" }
        else { Write-Host "     缺失 cache\$f （7B 首次运行会自动生成，但需要联网下载数据集）" -ForegroundColor Yellow }
    }
    foreach ($f in @("dataloader_llama_wikitext2_128.cache")) {
        $p = Join-Path $Root "cache\$f"
        if (-not (Test-Path $p)) { Write-Host "     [注意] 若缺 dataloader 且无外网，7B 无法启动" -ForegroundColor Yellow }
    }

    Write-Host "5/6 磁盘余量（每条 7B 腿约 14 GB checkpoint + 日志）"
    $drive = (Get-Item $Root).PSDrive
    $free = [Math]::Round((Get-PSDrive $drive.Name).Free / 1GB, 1)
    Write-Host ("     {0}: 剩余 {1} GB" -f $drive.Name, $free)
    if ($free -lt 40) { Write-Host "     余量偏低，建议先清理 log\ 下的旧 checkpoint" -ForegroundColor Yellow }

    Write-Host "6/6 环境指纹"
    if (Test-Path (Join-Path $Root "env_report.py")) {
        & $Py (Join-Path $Root "env_report.py") --out (Join-Path $Root "env_round8_preflight.txt") | Out-Null
        if (Test-Path (Join-Path $Root "env_round8_preflight.txt")) { Write-Host "     OK  env_round8_preflight.txt" }
        else { Write-Host "     env_report.py 执行未产出文件" -ForegroundColor Yellow }
    } else {
        Write-Host "     env_report.py 不存在，跳过（请在回传说明里注明）" -ForegroundColor Yellow
    }

    Write-Host ""
    if ($ok) { Write-Host "preflight 通过，可以 -Mode run" -ForegroundColor Green }
    else { Write-Host "preflight 有缺失项，请先补齐再跑（缺失项见上）" -ForegroundColor Yellow }
}

# ---------------------------------------------------------------- 结果汇总
function New-ResultFile {
    $out = Join-Path $Root "lab_round8_result.txt"
    $L = New-Object System.Collections.ArrayList
    [void]$L.Add("SpikeLLM / OSDSQ v2  第八轮实验结果（实验室双 3090）")
    [void]$L.Add("生成时间: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    [void]$L.Add("项目路径: " + $Root)
    [void]$L.Add("本轮问题: addbit(+1bit/T=2) 是否只是一个廉价的替代品")
    [void]$L.Add("")
    [void]$L.Add("---- 参照（T=2, addbit=1, fake_binary, 128x20, alpha=0.75, let_lr=1e-3）----")
    [void]$L.Add("  FP16 7B              5.4722 / 6.9727")
    [void]$L.Add("  Orig 7B  seed2      10.5726 / 13.7738")
    [void]$L.Add("  Ours 7B  seed2       7.9792 / 10.6792")
    [void]$L.Add("  Orig 68M (seed0/1)  172.29 / 175.14")
    [void]$L.Add("  Ours 68M seed2       156.84 / 90.53")
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

    [void]$L.Add("---- 判据（回传后三方核对，勿自行解释）----")
    [void]$L.Add("  A = 7B Orig@a3  wt2 ,  B = 7B Ours@a3  wt2")
    [void]$L.Add("  A >= 9.0       -> 原叙事成立，addbit 救不回来，方法有独立价值")
    [void]$L.Add("  7.0 <= A < 9.0 -> 部分替代，必须画 T-PPL 前沿，不能只报单点")
    [void]$L.Add("  A < 7.0        -> 24% 只是打了没人用的配置，论文改成前沿叙事")
    [void]$L.Add("  B 比 A 小 >5%  -> 方法在高 T 下仍有效（最好结果）")
    [void]$L.Add("  B 与 A 差 <2%  -> 方法只在中低 T 有效，正文需限定范围")
    [void]$L.Add("")
    [void]$L.Add("---- 环境指纹 ----")
    if (Test-Path (Join-Path $Root "env_round8_preflight.txt")) {
        [void]$L.Add("  见 env_round8_preflight.txt（随日志一并回传）")
    } else {
        [void]$L.Add("  未生成。本轮结果按规程【不可入账】，请补跑：python env_report.py --out env_round8_preflight.txt")
    }

    Set-Content -Path $out -Value $L -Encoding UTF8
    Write-Host "结果已写入 $out"
}

# ---------------------------------------------------------------- 状态
function Invoke-Status {
    Write-Step "第八轮状态"
    foreach ($j in $Jobs) {
        $dir = Join-Path $Root ("log\" + $j.Tag)
        $p = Test-Path (Join-Path $dir "done_ppl.flag")
        $d = Test-Path (Join-Path $dir "done.flag")
        $flag = if ($d) { "DONE" } elseif ($p) { "fake完成" } else { "未开始" }
        Write-Host ("  {0,-18} addbit={1}  {2}" -f $j.Tag, $j.Addbit, $flag)
    }
    Write-Host ""
    $txt = Join-Path $Root "lab_round8_result.txt"
    if (Test-Path $txt) { Get-Content $txt -Encoding UTF8 | Select-Object -First 40 }
    else { Write-Host "  （尚未生成结果文件）" }
}

# ---------------------------------------------------------------- 主入口
$script:AddbitOverride = $AddbitOverride

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
        Write-Step "只跑 68M 门禁（约 20 分钟）"
        foreach ($j in ($Jobs | Where-Object { $_.Group -eq "gate" })) {
            Invoke-TrainBody -Wd $Root -PyExe $Py -Model $j.Model -Net $j.Net -JobTag $j.Tag `
                -Seed $j.Seed -CacheDir $j.Cache -Gpu 0 -ExtraStr $j.Extra -Addbit $j.Addbit
        }
        New-ResultFile
    }

    "run" {
        Write-Step "开始执行（门禁 20 分钟 + 两条 7B 并行约 7~8 小时）"
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
        Write-Host "步骤 2/2：两条 7B 腿并行（GPU0 = Orig@a3，GPU1 = Ours@a3）"
        $legs = @("r8_7b_orig_a3", "r8_7b_ours_a3")
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
        Get-Content (Join-Path $Root "lab_round8_result.txt") -Encoding UTF8 | Select-Object -First 60
    }
}
