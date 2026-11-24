<#
================================================================================
第十二轮实验执行器（实验室 Windows 单卡 RTX 3090）· 兼本轮说明文档
生成：WorkBuddy  2026-11-18
自定位：所有路径由 $PSScriptRoot 推导，脚本放在 SpikeLLM 根目录即可运行。
本轮特点：
  1) 含 patch\（Antigravity 新增 --quota_policy 三策略，本地 68M 三门禁已过）
  2) 【单卡】所有 7B 腿串行执行，不占用双卡。墙钟更长，可断线续跑。
  3) 步骤含补丁 + 硬门禁 + 主实验 + 两个消融 + 零样本，一晚跑不完，分批用 -Mode 控制。

--------------------------------------------------------------------------------
【一、本轮要回答什么】

第十一轮已确认 Stage 2 在 7B 上成立（AQ 比 Full 再降 6.3%，等算力核验通过）。
本轮把证据补到"可投稿"的完整度：
  A. 7B AQ seed=4            -> 补齐 Stage 1+2 的完整三种子（与 Original/A-off/Full 对齐）
  B. 7B random_quota         -> 堵"任意随机重分配也能降 PPL"的质疑
  C. 7B heuristic_down_o     -> 堵"固定先验规则就够，不需要自适应搜索"的质疑
  D. 7B AQ 零样本六任务     -> 审稿人必问的下游任务，用已训好的 seed2 checkpoint 直测

--------------------------------------------------------------------------------
【二、腿清单（全部 fake_binary, 128x20, alpha=0.75, addbit=1，零样本全关）】

  门禁（68M，约 12 分钟，必须通过才继续）
    r12_68m_repro   seed2  无新开关
    预期：157.788040161133 / 90.831184387207（与第七轮逐位一致）

  主实验（7B，单卡串行，各约 8 小时）
    r12_7b_aq_s4    seed4  --adaptive_quota（默认 adaptive）  cache_seed4
    r12_7b_rand_s2  seed2  --adaptive_quota --quota_policy random        cache
    r12_7b_heur_s2  seed2  --adaptive_quota --quota_policy heuristic_down_o  cache

  零样本（7B，用 r12_7b_aq_s2 已训 checkpoint，约 40 分钟；见【三】）
    r12_7b_aq_s2_zs  在 r11_7b_aq_s2 的 checkpoint 上测六任务

  预期对表（仅供核对，不是判据）：
    r12_7b_aq_s4  对照 Full s4 = 8.0371 / 10.7135，预期 ~7.5x
    r12_7b_rand_s2  对照 AQ s2 = 7.5296；68M 上 random(157.80) > adaptive(154.43)
    r12_7b_heur_s2  对照 AQ s2 = 7.5296；68M 上 heuristic(157.22) 介于 random 与 adaptive 之间
    若 random/heuristic 反而比 adaptive 更低，原样回传，不要解释。

--------------------------------------------------------------------------------
【三、零样本腿说明（重要，避免误判）】

  零样本评测不需要重训，而是在已训好的 checkpoint 上 resume 并只跑 tasks。
  本轮用第十一轮的 r11_7b_aq_s2 结果（7.5296 / 10.1216）。
  若实验室保留了该轮 log\r11_7b_aq_s2\omni_parameters.pth，则脚本自动复用；
  若没有，请在回传里注明，零样本腿会跳过（不影响前面三条 PPL 腿）。

--------------------------------------------------------------------------------
【四、补丁与门禁】

  patch\ 内 2 个文件（SHA-256 前 16 位）：
    main.py                              e898ff1da6b92e91
    spike_driven_quant/spike_omniquant.py 4f7ae925a3c1bd51
  （spike_linear.py 无需打包，哈希保持 857be868ad9cc740 作基线校验）
  原文件自动备份到 backup_round12\；门禁失败自动还原并停止。
  打补丁后脚本会补跑一次 env_report.py（第十二轮结果必须带新指纹）。

--------------------------------------------------------------------------------
【五、怎么跑】

  1. 整包解压到 SpikeLLM 根目录（含 patch\，别只拷 ps1）。
  2. powershell -ExecutionPolicy Bypass -File .\lab_round12_runner.ps1 -Mode preflight
  3. 分步跑（单卡，一晚跑不完，建议按序多次执行，脚本自动续跑跳过已完成腿）：
       -Mode gate             -> 补丁 + 68M 门禁（约 13 分钟）
       -Mode run              -> 三条 7B PPL 腿 + 零样本（串行约 24 小时）
       或单独指定：
       -Mode runleg -Tag r12_7b_aq_s4      （单腿）
  4. 回传：lab_round12_result.txt + log\r12_*\log_rank0_*.txt + env_round12_preflight.txt。
================================================================================
#>

param(
    [ValidateSet("preflight", "run", "gate", "status", "runleg", "patch", "restore")]
    [string]$Mode = "status",
    [string]$Tag = "",
    [string]$Py = "",
    [string]$ModelDir = "",
    [switch]$SkipPatch
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

$PatchFiles = @(
    @{ Rel = "main.py";                                Sha = "e898ff1da6b92e91" },
    @{ Rel = "spike_driven_quant\spike_omniquant.py";  Sha = "4f7ae925a3c1bd51" }
)
$BaselineSha = @(
    @{ Rel = "spike_driven_quant\spike_linear.py"; Sha = "857be868ad9cc740" }
)

$ZSTasks = "piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande"

$Jobs = @(
    @{ Tag = "r12_68m_repro"; Model = $Model68M; Net = $Net68M; Seed = 2; Cache = "./cache"; Addbit = 1;
       Extra = ""; Gpu = 0; Group = "gate";
       Note = "补丁忠实性硬门禁：全开关关复跑，预期 157.7880/90.8312 逐位一致" },
    @{ Tag = "r12_7b_aq_s4"; Model = $Model7B; Net = $Net7B; Seed = 4; Cache = "./cache_seed4"; Addbit = 1;
       Extra = "--adaptive_quota"; Gpu = 0; Group = "main";
       Note = "7B Stage2 seed4，补齐三种子（对照 Full s4=8.0371）" },
    @{ Tag = "r12_7b_rand_s2"; Model = $Model7B; Net = $Net7B; Seed = 2; Cache = "./cache"; Addbit = 1;
       Extra = "--adaptive_quota|--quota_policy|random"; Gpu = 0; Group = "main";
       Note = "7B 等算力随机交换消融（对照 AQ s2=7.5296）" },
    @{ Tag = "r12_7b_heur_s2"; Model = $Model7B; Net = $Net7B; Seed = 2; Cache = "./cache"; Addbit = 1;
       Extra = "--adaptive_quota|--quota_policy|heuristic_down_o"; Gpu = 0; Group = "main";
       Note = "7B 固定先验规则消融（对照 AQ s2=7.5296）" }
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

# ---------------------------------------------------------------- 补丁与还原
function Invoke-Patch {
    Write-Step "打补丁（原文件备份到 backup_round12\）"
    $bak = Join-Path $Root "backup_round12"
    New-Item -ItemType Directory -Force -Path $bak | Out-Null
    foreach ($pf in $PatchFiles) {
        $src = Join-Path $Root ("patch\" + $pf.Rel)
        $dst = Join-Path $Root $pf.Rel
        if (-not (Test-Path $src)) { Fail "补丁源缺失：patch\$($pf.Rel)" }
        $bakDst = Join-Path $bak $pf.Rel
        New-Item -ItemType Directory -Force -Path (Split-Path $bakDst -Parent) | Out-Null
        if (-not (Test-Path $bakDst)) { Copy-Item $dst $bakDst -Force }
        Copy-Item $src $dst -Force
        $sha = Get-Sha16 $dst
        Write-Host ("  {0,-45} {1}  {2}" -f $pf.Rel, $sha, $(if ($sha -eq $pf.Sha) { "OK" } else { "与预期不符" }))
        if ($sha -ne $pf.Sha) { Fail "补丁哈希不符，中止：$($pf.Rel)" }
    }
    # 打补丁后立即补跑环境指纹
    if (Test-Path (Join-Path $Root "env_report.py")) {
        & $Py (Join-Path $Root "env_report.py") --out (Join-Path $Root "env_round12_preflight.txt") | Out-Null
        Write-Host "  已补跑 env_report.py -> env_round12_preflight.txt"
    }
}

function Invoke-Restore {
    Write-Step "从 backup_round12 还原原文件"
    $bak = Join-Path $Root "backup_round12"
    if (-not (Test-Path $bak)) { Fail "备份目录不存在：$bak" }
    foreach ($pf in $PatchFiles) {
        $bakSrc = Join-Path $bak $pf.Rel
        $dst = Join-Path $Root $pf.Rel
        if (Test-Path $bakSrc) { Copy-Item $bakSrc $dst -Force; Write-Host "  已还原 $($pf.Rel)" }
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

# ---------------------------------------------------------------- 零样本腿
function Invoke-Zeroshot {
    param($PyExe, $Model, $Net, $Seed, $CacheDir, $Gpu, $ExtraStr, $Addbit)

    $srcCkpt = Join-Path $Root "log\r11_7b_aq_s2\omni_parameters.pth"
    $dir = Join-Path $Root "log\r12_7b_aq_s2_zs"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null

    if (-not (Test-Path $srcCkpt)) {
        Write-Output "[zero-shot] 跳过：找不到 r11_7b_aq_s2 的 checkpoint（$srcCkpt）"
        return
    }

    if ([string]::IsNullOrWhiteSpace($ExtraStr)) { $Extra = @() }
    else { $Extra = $ExtraStr -split "\|" }

    Set-Location $Root
    $env:CUDA_VISIBLE_DEVICES = "$Gpu"

    $base = @(
        "main.py", "--model", $Model, "--net", $Net,
        "--wbits", "4", "--abits", "4", "--addbit", "$Addbit", "--low_p", "0.9",
        "--let", "--lwc", "--aug_loss",
        "--alpha", "0.75", "--let_lr", "1e-3", "--lwc_lr", "1e-2",
        "--nsamples", "128", "--epochs", "0", "--seed", "$Seed",
        "--cache_dir", $CacheDir, "--resume", $srcCkpt
    )
    if ($Extra) { $base += $Extra }

    if (-not (Test-Path "$dir/done_zs.flag")) {
        Write-Output "[zero-shot] 开始六任务评测"
        $argv = $base + @("--tasks", $script:ZSTasks, "--num_fewshot", "0", "--output_dir", $dir)
        & $PyExe @argv *>> "$dir/run.log"
        if ($LASTEXITCODE -ne 0) {
            Move-Item "$dir/run.log" "$dir/run.log.failed" -Force -ErrorAction SilentlyContinue
            throw "[zero-shot] 失败 (exit $LASTEXITCODE)，见 $dir/run.log.failed"
        }
        New-Item -ItemType File -Force -Path "$dir/done_zs.flag" | Out-Null
        Write-Output "[zero-shot] 完成"
    } else {
        Write-Output "[zero-shot] done_zs 已存在，跳过"
    }
}

# ---------------------------------------------------------------- preflight
function Invoke-Preflight {
    Write-Step "第十二轮 preflight"
    $ok = $true

    Write-Host "1/7 补丁包完整性（SHA-256 前 16 位）"
    foreach ($pf in $PatchFiles) {
        $src = Join-Path $Root ("patch\" + $pf.Rel)
        if (Test-Path $src) {
            $sha = Get-Sha16 $src
            Write-Host ("     {0}  {1}" -f $sha, $(if ($sha -eq $pf.Sha) { "OK  patch\" + $pf.Rel } else { "与预期不符 patch\" + $pf.Rel }))
        } else { Write-Host "     缺失 patch\$($pf.Rel)" -ForegroundColor Yellow; $ok = $false }
    }
    foreach ($b in $BaselineSha) {
        $sha = Get-Sha16 (Join-Path $Root $b.Rel)
        Write-Host ("     {0}  spike_linear.py 基线校验 {1}" -f $sha, $(if ($sha -eq $b.Sha) { "OK" } else { "不符" }))
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
    foreach ($sd in @("cache_seed4")) {
        if (Test-Path (Join-Path $Root $sd)) { Write-Host "     OK  $sd 已存在" }
        else { Write-Host "     $sd 不存在，run 时自动创建" }
    }

    Write-Host "6/7 零样本 checkpoint（第十一轮 r11_7b_aq_s2）"
    $ck = Join-Path $Root "log\r11_7b_aq_s2\omni_parameters.pth"
    if (Test-Path $ck) { Write-Host "     OK  有 checkpoint，零样本腿将执行" }
    else { Write-Host "     缺失 $ck —— 零样本腿会跳过（不影响 PPL 腿）" -ForegroundColor Yellow }

    Write-Host "7/7 磁盘余量"
    $drive = (Get-Item $Root).PSDrive
    $free = [Math]::Round((Get-PSDrive $drive.Name).Free / 1GB, 1)
    Write-Host ("     {0}: 剩余 {1} GB" -f $drive.Name, $free)
    if ($free -lt 40) { Write-Host "     余量偏低，建议先清理 log\ 下的旧 checkpoint" -ForegroundColor Yellow }

    Write-Host ""
    if ($ok) { Write-Host "preflight 通过。单卡建议：先 -Mode gate，再 -Mode run" -ForegroundColor Green }
    else { Write-Host "preflight 有缺失项，请先补齐再跑" -ForegroundColor Yellow }
}

# ---------------------------------------------------------------- 结果汇总
function New-ResultFile {
    $out = Join-Path $Root "lab_round12_result.txt"
    $L = New-Object System.Collections.ArrayList
    [void]$L.Add("SpikeLLM / OSDSQ v2  第十二轮实验结果（实验室单卡 3090）")
    [void]$L.Add("生成时间: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    [void]$L.Add("项目路径: " + $Root)
    [void]$L.Add("任务: 7B AQ seed4 补全 + random/heuristic 消融 + AQ 零样本")
    [void]$L.Add("")
    [void]$L.Add("---- 补丁哈希（SHA-256 前 16 位）----")
    foreach ($pf in $PatchFiles) {
        $sha = Get-Sha16 (Join-Path $Root $pf.Rel)
        [void]$L.Add(("  {0,-45} {1}  {2}" -f $pf.Rel, $sha, $(if ($sha -eq $pf.Sha) { "OK" } else { "!! 不符" })))
    }
    [void]$L.Add("")
    [void]$L.Add("---- 参照 ----")
    [void]$L.Add("  7B Full  s2 7.9792 / 10.6792   s3 8.0948 / 10.7126   s4 8.0371 / 10.7135")
    [void]$L.Add("  7B AQ    s2 7.5296 / 10.1216   s3 7.5272 / 10.1423")
    [void]$L.Add("  7B Orig 10.5806 ± 0.0462    A-off 10.2818 ± 0.1353")
    [void]$L.Add("  68M 门禁参照 157.7880 / 90.8312")
    [void]$L.Add("")

    foreach ($j in $Jobs) {
        $dir = Join-Path $Root ("log\" + $j.Tag)
        [void]$L.Add("### " + $j.Tag + "  (seed=" + $j.Seed + ")")
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

    [void]$L.Add("---- 零样本（若执行）----")
    $zdir = Join-Path $Root "log\r12_7b_aq_s2_zs"
    if (Test-Path (Join-Path $zdir "done_zs.flag")) {
        $zf = Get-ChildItem -Path $zdir -Filter "log_rank0_*.txt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
        if ($zf) { [void]$L.Add("  见 " + $zf.FullName) }
    } else {
        [void]$L.Add("  未执行（无 checkpoint 或未跑到）")
    }
    [void]$L.Add("")
    [void]$L.Add("---- 门禁一判定（自动）----")
    $dir = Join-Path $Root "log\r12_68m_repro"
    $f = Get-ChildItem -Path $dir -Filter "log_rank0_*.txt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
    if ($f) {
        $w = (Select-String -Path $f.FullName -Pattern "INFO wikitext2 :" | Select-Object -Last 1)
        $wv = if ($w) { [double](($w.Line -split ":")[-1].Trim()) } else { -1 }
        if ($wv -ge 0) {
            $dw = [Math]::Abs($wv - 157.788040161133)
            if ($dw -lt 0.0001) { [void]$L.Add("  PASS：门禁一复现逐位一致（差 $dw）") }
            else { [void]$L.Add("  FAIL：门禁一差 $dw —— 补丁不忠实，主实验不可比") }
        }
    }
    [void]$L.Add("")
    [void]$L.Add("---- 环境指纹 ----")
    if (Test-Path (Join-Path $Root "env_round12_preflight.txt")) {
        [void]$L.Add("  见 env_round12_preflight.txt（补丁后补跑，随日志一并回传）")
    } else {
        [void]$L.Add("  未生成。本轮结果按规程【不可入账】，请补跑：python env_report.py --out env_round12_preflight.txt")
    }

    Set-Content -Path $out -Value $L -Encoding UTF8
    Write-Host "结果已写入 $out"
}

# ---------------------------------------------------------------- 状态
function Invoke-Status {
    Write-Step "第十二轮状态"
    foreach ($j in $Jobs) {
        $dir = Join-Path $Root ("log\" + $j.Tag)
        $p = Test-Path (Join-Path $dir "done_ppl.flag")
        $d = Test-Path (Join-Path $dir "done.flag")
        $flag = if ($d) { "DONE" } elseif ($p) { "fake完成" } else { "未开始" }
        Write-Host ("  {0,-18} seed={1}  {2}" -f $j.Tag, $j.Seed, $flag)
    }
    $zdir = Join-Path $Root "log\r12_7b_aq_s2_zs"
    Write-Host ("  {0,-18}        {1}" -f "r12_7b_aq_s2_zs", $(if (Test-Path (Join-Path $zdir "done_zs.flag")) { "DONE" } else { "未开始" }))
}

# ---------------------------------------------------------------- 主入口
$script:ZSTasks = $ZSTasks

switch ($Mode) {

    "preflight" { Invoke-Preflight }

    "status" { Invoke-Status }

    "patch" { Invoke-Patch }

    "restore" { Invoke-Restore }

    "runleg" {
        $j = $Jobs | Where-Object { $_.Tag -eq $Tag }
        if (-not $j) { Fail "未知的 Tag: $Tag" }
        Invoke-TrainBody -Wd $Root -PyExe $Py -Model $j.Model -Net $j.Net -JobTag $j.Tag `
            -Seed $j.Seed -CacheDir $j.Cache -Gpu $j.Gpu -ExtraStr $j.Extra -Addbit $j.Addbit
    }

    "gate" {
        Write-Step "补丁 + 68M 门禁（约 13 分钟）"
        if ($SkipPatch) { Write-Host "[SkipPatch] 跳过补丁" } else { Invoke-Patch }
        $j0 = $Jobs | Where-Object { $_.Tag -eq "r12_68m_repro" }
        Invoke-TrainBody -Wd $Root -PyExe $Py -Model $j0.Model -Net $j0.Net -JobTag $j0.Tag `
            -Seed $j0.Seed -CacheDir $j0.Cache -Gpu 0 -ExtraStr $j0.Extra -Addbit $j0.Addbit
        $dir = Join-Path $Root "log\r12_68m_repro"
        $f = Get-ChildItem -Path $dir -Filter "log_rank0_*.txt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
        if ($f) {
            $w = (Select-String -Path $f.FullName -Pattern "INFO wikitext2 :" | Select-Object -Last 1)
            $wv = if ($w) { [double](($w.Line -split ":")[-1].Trim()) } else { -1 }
            if (($wv -ge 0) -and ([Math]::Abs($wv - 157.788040161133) -lt 0.0001)) {
                Write-Host "门禁一 PASS（wt2 = $wv）" -ForegroundColor Green
            } else {
                Write-Host "门禁一 FAIL：wt2 = $wv，预期 157.788040161133" -ForegroundColor Red
                Invoke-Restore
                Fail "补丁不忠实，已还原原文件。请回传结果后告知开发组。"
            }
        }
        New-ResultFile
    }

    "run" {
        Write-Step "开始执行（单卡串行：3 条 7B PPL 腿 + 零样本）"
        $start = Get-Date

        Prepare-Cache (Join-Path $Root "cache_seed4")
        Write-Host "[OK ] cache_seed4 就绪"

        if (-not $SkipPatch) {
            $patchDone = Test-Path (Join-Path $Root "backup_round12")
            if (-not $patchDone) { Invoke-Patch }
            else { Write-Host "[提示] 检测到 backup_round12 已存在，跳过补丁（已打）。如代码被还原请先 -Mode patch" }
        }

        Write-Host "步骤 1：68M 门禁（如未跑过）"
        $j0 = $Jobs | Where-Object { $_.Tag -eq "r12_68m_repro" }
        if (-not (Test-Path (Join-Path $Root "log\r12_68m_repro\done.flag"))) {
            Invoke-TrainBody -Wd $Root -PyExe $Py -Model $j0.Model -Net $j0.Net -JobTag $j0.Tag `
                -Seed $j0.Seed -CacheDir $j0.Cache -Gpu 0 -ExtraStr $j0.Extra -Addbit $j0.Addbit
        } else { Write-Host "门禁已跑过，跳过" }

        Write-Host "步骤 2：三条 7B PPL 腿（单卡串行）"
        foreach ($t in @("r12_7b_aq_s4", "r12_7b_rand_s2", "r12_7b_heur_s2")) {
            $j = $Jobs | Where-Object { $_.Tag -eq $t }
            try {
                Invoke-TrainBody -Wd $Root -PyExe $Py -Model $j.Model -Net $j.Net -JobTag $j.Tag `
                    -Seed $j.Seed -CacheDir $j.Cache -Gpu 0 -ExtraStr $j.Extra -Addbit $j.Addbit
            } catch {
                Write-Host "[WARN] 腿 $($j.Tag) 失败：$_" -ForegroundColor Yellow
            }
            New-ResultFile
        }

        Write-Host "步骤 3：零样本六任务（用 r11_7b_aq_s2 checkpoint）"
        $jzs = $Jobs | Where-Object { $_.Tag -eq "r12_7b_aq_s2" }
        $jseed = 2
        try {
            Invoke-Zeroshot -PyExe $Py -Model $Model7B -Net $Net7B -Seed 2 -CacheDir "./cache" -Gpu 0 `
                -ExtraStr "--adaptive_quota" -Addbit 1
        } catch {
            Write-Host "[WARN] 零样本腿失败：$_" -ForegroundColor Yellow
        }
        New-ResultFile

        $elapsed = (Get-Date) - $start
        Write-Host ""
        Write-Host ("全部结束，用时 {0:N1} 小时" -f $elapsed.TotalHours)
        Write-Step "结果预览"
        Get-Content (Join-Path $Root "lab_round12_result.txt") -Encoding UTF8 | Select-Object -First 70
    }
}
