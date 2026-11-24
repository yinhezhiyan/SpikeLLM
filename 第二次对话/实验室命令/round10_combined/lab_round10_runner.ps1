<#
================================================================================
第十轮实验执行器（实验室 Windows 双 RTX 3090）· 兼本轮说明文档
生成：WorkBuddy  2026-11-08（合体版：补丁 + 双门禁 + A-off 种子）
自定位：所有路径由 $PSScriptRoot 推导，脚本放在 SpikeLLM 根目录即可运行。
       本包比以往多一个 patch\ 目录，先打补丁再跑，流程见【三】。

--------------------------------------------------------------------------------
【一、本轮三件事（一晚做完，互相咬合）】

  A. 更新实验室代码（Antigravity 2026-11-07 落地的 Stage 2 开关，3 个文件）
  B. 门禁一（硬门禁）：补丁后复跑一条已知数，验证"开关关闭=逐位一致"
  C. 主体：7B A-off 补 seed3/seed4 —— 把论文因果链上最大的洞（单种子）补掉

  为什么 A-off 是主体：
  "A 单独贡献 23.4 个百分点"是全文最重要的一句话，现在只有 seed2 一个数，
  而 Full 和 Original 都是三种子。本轮补齐。

  顺带（门禁二，约 36 分钟）：68M 上 --adaptive_quota 的 seed2/3/4 三种子稳定性。
  开发机 seed2 实测：154.4269 / 88.3023（比 Stage 1 的 159.28 / 91.74 再降 3%），
  高显著 MAC 严格不变（946,176 -> 946,176）。本机数字可能与开发机不同，只记录不判。

--------------------------------------------------------------------------------
【二、腿清单与预期（预期仅供对表，不是判据）】

  门禁一（约 12 分钟，GPU0，必须通过才继续）
    r10_68m_repro      68M  全开关关闭  seed=2
    预期：wikitext2 = 157.7880401611328，c4 = 90.83118438720703
    （这是第七轮 r7_68m_mean_check 在本机的实测值。差超过 0.0001 即判失败，
      脚本会自动还原补丁并停止，不会浪费 8 小时主实验。）

  门禁二（约 36 分钟，GPU0，逐条跑，失败不阻塞主体）
    r10_68m_aq_s2/s3/s4   68M  --adaptive_quota  seed=2/3/4
    预期（开发机参照）：s2 ≈ 154.43 / 88.30；s3/s4 未知，只记录。

  主实验（约 8 小时，双卡并行）
    r10_7b_aoff_seed3    7B   --ablate_no_linear_shield  seed=3  (GPU0, cache_seed3)
    r10_7b_aoff_seed4    7B   --ablate_no_linear_shield  seed=4  (GPU1, cache_seed4)
    预期：wt2 落在 10.3~10.6（参照 seed2=10.4181、Orig 三种子 10.5806±0.0462）。

  全程 fake_binary_simulate，只测 PPL，零样本关闭，不跑 real_multibit。

--------------------------------------------------------------------------------
【三、补丁与还原（重要）】

  patch\ 内 3 个文件（SHA-256 前 16 位，打完补丁脚本会自验）：
    main.py                              49485bc7ad4e9e0b
    spike_driven_quant/spike_linear.py   857be868ad9cc740
    spike_driven_quant/spike_omniquant.py 2b17e4fcfea74fce
  原 3 个文件会先备份到 backup_round10\ 下。门禁一失败时脚本自动从备份还原。
  手动还原：把 backup_round10 下的文件复制回原位即可。

  与既有结果的可比性：Antigravity 已在开发机 68M 上验证"开关关闭时
  checkpoint MD5 逐位相同"。门禁一是同一件事在本机的复验，通过后，
  本轮 A-off 两条腿与第七轮的 Full/Original 三种子属同一数值路径。

--------------------------------------------------------------------------------
【四、怎么跑】

  1. 把 zip 整包解压到 SpikeLLM 根目录（和 main.py 同级），应看到
     lab_round10_runner.ps1 与 patch\ 两个东西。
  2. powershell -ExecutionPolicy Bypass -File .\lab_round10_runner.ps1 -Mode preflight
     powershell -ExecutionPolicy Bypass -File .\lab_round10_runner.ps1 -Mode run
  3. 可断线，重跑 -Mode run 自动跳过已完成的腿。
  4. 回传：lab_round10_result.txt + log\r10_*\log_rank0_*.txt + env_round10_preflight.txt。

  【兜底】门禁一失败时，脚本会自动还原原文件并停止。此时不需要第二个包，
  加 -NoPatch 重跑即可在当前（原版）代码上直接完成两条 A-off 主腿。
================================================================================
#>

param(
    [ValidateSet("preflight", "run", "gate", "status", "runleg", "patch", "restore")]
    [string]$Mode = "status",
    [string]$Tag = "",
    [string]$Py = "",
    [string]$ModelDir = "",
    [switch]$NoPatch
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
    @{ Rel = "main.py";                              Sha = "49485bc7ad4e9e0b" },
    @{ Rel = "spike_driven_quant\spike_linear.py";   Sha = "857be868ad9cc740" },
    @{ Rel = "spike_driven_quant\spike_omniquant.py"; Sha = "2b17e4fcfea74fce" }
)

$Jobs = @(
    @{ Tag = "r10_68m_repro"; Model = $Model68M; Net = $Net68M; Seed = 2; Cache = "./cache"; Addbit = 1;
       Extra = ""; Gpu = 0; Group = "gate1";
       Note = "补丁忠实性硬门禁：全开关关闭复跑第七轮 mean_check，预期 157.7880/90.8312 逐位一致" },
    @{ Tag = "r10_68m_aq_s2"; Model = $Model68M; Net = $Net68M; Seed = 2; Cache = "./cache"; Addbit = 1;
       Extra = "--adaptive_quota"; Gpu = 0; Group = "gate2";
       Note = "68M Stage2 seed2（开发机参照 154.4269/88.3023）" },
    @{ Tag = "r10_68m_aq_s3"; Model = $Model68M; Net = $Net68M; Seed = 3; Cache = "./cache"; Addbit = 1;
       Extra = "--adaptive_quota"; Gpu = 0; Group = "gate2";
       Note = "68M Stage2 seed3 稳定性" },
    @{ Tag = "r10_68m_aq_s4"; Model = $Model68M; Net = $Net68M; Seed = 4; Cache = "./cache"; Addbit = 1;
       Extra = "--adaptive_quota"; Gpu = 0; Group = "gate2";
       Note = "68M Stage2 seed4 稳定性" },
    @{ Tag = "r10_7b_aoff_seed3"; Model = $Model7B; Net = $Net7B; Seed = 3; Cache = "./cache_seed3"; Addbit = 1;
       Extra = "--ablate_no_linear_shield"; Gpu = 0; Group = "main";
       Note = "7B A-off seed3，与 Orig seed3(10.6303) / Ours seed3(8.0948) 配对" },
    @{ Tag = "r10_7b_aoff_seed4"; Model = $Model7B; Net = $Net7B; Seed = 4; Cache = "./cache_seed4"; Addbit = 1;
       Extra = "--ablate_no_linear_shield"; Gpu = 1; Group = "main";
       Note = "7B A-off seed4，与 Orig seed4(10.5389) / Ours seed4(8.0371) 配对" }
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
function Get-Sha16($path) {
    if (-not (Test-Path $path)) { return "MISSING" }
    return (Get-FileHash -Path $path -Algorithm SHA256).Hash.Substring(0, 16).ToLower()
}

function Invoke-Patch {
    Write-Step "打补丁（原文件备份到 backup_round10\）"
    $bak = Join-Path $Root "backup_round10"
    New-Item -ItemType Directory -Force -Path $bak | Out-Null
    foreach ($pf in $PatchFiles) {
        $src = Join-Path $Root ("patch\" + $pf.Rel)
        $dst = Join-Path $Root $pf.Rel
        if (-not (Test-Path $src)) { Fail "补丁源缺失：patch\$($pf.Rel)（请确认 zip 整包解压）" }
        $dstDir = Split-Path $dst -Parent
        $bakDst = Join-Path $bak $pf.Rel
        New-Item -ItemType Directory -Force -Path (Split-Path $bakDst -Parent) | Out-Null
        if (-not (Test-Path $bakDst)) { Copy-Item $dst $bakDst -Force }
        Copy-Item $src $dst -Force
        $sha = Get-Sha16 $dst
        Write-Host ("  {0,-45} {1}  {2}" -f $pf.Rel, $sha, $(if ($sha -eq $pf.Sha) { "OK" } else { "与预期不符" }))
    }
}

function Invoke-Restore {
    Write-Step "从 backup_round10 还原原文件"
    $bak = Join-Path $Root "backup_round10"
    if (-not (Test-Path $bak)) { Fail "备份目录不存在：$bak" }
    foreach ($pf in $PatchFiles) {
        $bakSrc = Join-Path $bak $pf.Rel
        $dst = Join-Path $Root $pf.Rel
        if (Test-Path $bakSrc) { Copy-Item $bakSrc $dst -Force; Write-Host "  已还原 $($pf.Rel)" }
        else { Write-Host "  备份缺失 $($pf.Rel)，跳过" -ForegroundColor Yellow }
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
    Write-Step "第十轮 preflight"
    $ok = $true

    Write-Host "1/7 Python 与依赖"
    $v = & $Py -c "import torch,transformers,sys;print(sys.version.split()[0],torch.__version__,torch.version.cuda,transformers.__version__)"
    Write-Host "     python / torch / cuda / transformers = $v"
    if ($LASTEXITCODE -ne 0) { $ok = $false }

    Write-Host "2/7 补丁包完整性"
    foreach ($pf in $PatchFiles) {
        $src = Join-Path $Root ("patch\" + $pf.Rel)
        if (Test-Path $src) {
            $sha = Get-Sha16 $src
            Write-Host ("     {0}  {1}" -f $sha, $(if ($sha -eq $pf.Sha) { "OK  patch\" + $pf.Rel } else { "与预期不符 patch\" + $pf.Rel }))
        } else { Write-Host "     缺失 patch\$($pf.Rel)" -ForegroundColor Yellow; $ok = $false }
    }

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
    foreach ($sd in @("cache_seed3", "cache_seed4")) {
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
        & $Py (Join-Path $Root "env_report.py") --out (Join-Path $Root "env_round10_preflight.txt") | Out-Null
        if (Test-Path (Join-Path $Root "env_round10_preflight.txt")) { Write-Host "     OK  env_round10_preflight.txt" }
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
    $out = Join-Path $Root "lab_round10_result.txt"
    $L = New-Object System.Collections.ArrayList
    [void]$L.Add("SpikeLLM / OSDSQ v2  第十轮实验结果（实验室双 3090，合体版）")
    [void]$L.Add("生成时间: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    [void]$L.Add("项目路径: " + $Root)
    [void]$L.Add("任务: A-off 补 seed3/4（主体）+ 补丁忠实性门禁 + 68M Stage2 三种子")
    [void]$L.Add("")
    [void]$L.Add("---- 补丁哈希（应与预期一致）----")
    foreach ($pf in $PatchFiles) {
        $sha = Get-Sha16 (Join-Path $Root $pf.Rel)
        [void]$L.Add(("  {0,-45} {1}  {2}" -f $pf.Rel, $sha, $(if ($sha -eq $pf.Sha) { "OK" } else { "!! 不符" })))
    }
    [void]$L.Add("")
    [void]$L.Add("---- 参照 ----")
    [void]$L.Add("  7B  Ours  8.0370 ± 0.0578   Orig 10.5806 ± 0.0462   A-off seed2 10.4181")
    [void]$L.Add("  7B  逐种子  Ours: 7.9792 / 8.0948 / 8.0371")
    [void]$L.Add("              Orig: 10.5726 / 10.6303 / 10.5389")
    [void]$L.Add("  68M mean_check（第七轮本机）157.7880 / 90.8312")
    [void]$L.Add("  68M Stage2（开发机 seed2 参照）154.4269 / 88.3023")
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

    [void]$L.Add("---- 门禁一判定（自动）----")
    $wv, $cv = Get-LegPpl "r10_68m_repro"
    if ($wv -ge 0) {
        $dw = [Math]::Abs($wv - 157.7880401611328)
        $dc = [Math]::Abs($cv - 90.83118438720703)
        if (($dw -lt 0.0001) -and ($dc -lt 0.0001)) {
            [void]$L.Add("  PASS：与第七轮 mean_check 逐位一致（wt2 差 $dw，c4 差 $dc）")
        } else {
            [void]$L.Add("  FAIL：wt2 差 $dw，c4 差 $dc —— 补丁不忠实，已自动还原，主实验未启动")
        }
    } else {
        [void]$L.Add("  未知：repro 腿未找到结果")
    }
    [void]$L.Add("")
    [void]$L.Add("---- 环境指纹 ----")
    if (Test-Path (Join-Path $Root "env_round10_preflight.txt")) {
        [void]$L.Add("  见 env_round10_preflight.txt（随日志一并回传）")
    } else {
        [void]$L.Add("  未生成。本轮结果按规程【不可入账】，请补跑：python env_report.py --out env_round10_preflight.txt")
    }

    Set-Content -Path $out -Value $L -Encoding UTF8
    Write-Host "结果已写入 $out"
}

# ---------------------------------------------------------------- 状态
function Invoke-Status {
    Write-Step "第十轮状态"
    foreach ($j in $Jobs) {
        $dir = Join-Path $Root ("log\" + $j.Tag)
        $p = Test-Path (Join-Path $dir "done_ppl.flag")
        $d = Test-Path (Join-Path $dir "done.flag")
        $flag = if ($d) { "DONE" } elseif ($p) { "fake完成" } else { "未开始" }
        Write-Host ("  {0,-22} seed={1}  {2}" -f $j.Tag, $j.Seed, $flag)
    }
}

# ---------------------------------------------------------------- 主入口
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
        Write-Step "只跑门禁（约 48 分钟，不含主实验）"
        Invoke-Patch
        $j0 = $Jobs | Where-Object { $_.Tag -eq "r10_68m_repro" }
        Invoke-TrainBody -Wd $Root -PyExe $Py -Model $j0.Model -Net $j0.Net -JobTag $j0.Tag `
            -Seed $j0.Seed -CacheDir $j0.Cache -Gpu 0 -ExtraStr $j0.Extra -Addbit $j0.Addbit
        $wv, $cv = Get-LegPpl "r10_68m_repro"
        if (($wv -ge 0) -and ([Math]::Abs($wv - 157.7880401611328) -lt 0.0001)) {
            Write-Host "门禁一 PASS，继续门禁二" -ForegroundColor Green
            foreach ($j in ($Jobs | Where-Object { $_.Group -eq "gate2" })) {
                Invoke-TrainBody -Wd $Root -PyExe $Py -Model $j.Model -Net $j.Net -JobTag $j.Tag `
                    -Seed $j.Seed -CacheDir $j.Cache -Gpu 0 -ExtraStr $j.Extra -Addbit $j.Addbit
            }
        } else {
            Write-Host "门禁一 FAIL：wt2 = $wv，预期 157.7880401611328" -ForegroundColor Red
            Invoke-Restore
            Fail "补丁不忠实，已还原原文件。请回传 lab_round10_result.txt，然后加 -NoPatch 重跑完成 A-off 主腿。"
        }
        New-ResultFile
    }

    "run" {
        Write-Step "开始执行（补丁 1 分钟 + 门禁 48 分钟 + 两条 7B 并行约 8 小时）"
        $start = Get-Date

        foreach ($sd in @("cache_seed3", "cache_seed4")) {
            Prepare-Cache (Join-Path $Root $sd)
            Write-Host "[OK ] $sd 就绪"
        }

        if ($NoPatch) {
            Write-Host "[NoPatch] 跳过补丁与两道门禁，直接在当前代码上跑两条 A-off 主腿" -ForegroundColor Yellow
        } else {
            Write-Host "步骤 1/3：打补丁"
            Invoke-Patch

            Write-Host "步骤 2/3：门禁一（硬门禁，失败自动还原并停止）"
            $j0 = $Jobs | Where-Object { $_.Tag -eq "r10_68m_repro" }
            Invoke-TrainBody -Wd $Root -PyExe $Py -Model $j0.Model -Net $j0.Net -JobTag $j0.Tag `
                -Seed $j0.Seed -CacheDir $j0.Cache -Gpu 0 -ExtraStr $j0.Extra -Addbit $j0.Addbit
            $wv, $cv = Get-LegPpl "r10_68m_repro"
            if (($wv -lt 0) -or ([Math]::Abs($wv - 157.7880401611328) -ge 0.0001)) {
                New-ResultFile
                Invoke-Restore
                Fail "门禁一 FAIL：wt2 = $wv，预期 157.7880401611328。已还原原文件。请回传 lab_round10_result.txt，然后加 -NoPatch 重跑完成 A-off 主腿。"
            }
            Write-Host "门禁一 PASS（wt2 = $wv）" -ForegroundColor Green

            Write-Host "步骤 3/3a：门禁二（68M Stage2 三种子，失败不阻塞主体）"
            foreach ($j in ($Jobs | Where-Object { $_.Group -eq "gate2" })) {
                try {
                    Invoke-TrainBody -Wd $Root -PyExe $Py -Model $j.Model -Net $j.Net -JobTag $j.Tag `
                        -Seed $j.Seed -CacheDir $j.Cache -Gpu 0 -ExtraStr $j.Extra -Addbit $j.Addbit
                } catch {
                    Write-Host "[WARN] 门禁二腿 $($j.Tag) 失败：$_" -ForegroundColor Yellow
                }
            }
            New-ResultFile
        }

        Write-Host ""
        Write-Host "步骤 3/3b：两条 7B A-off 腿并行（GPU0 = seed3，GPU1 = seed4）"
        $legs = @("r10_7b_aoff_seed3", "r10_7b_aoff_seed4")
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
        Get-Content (Join-Path $Root "lab_round10_result.txt") -Encoding UTF8 | Select-Object -First 70
    }
}
