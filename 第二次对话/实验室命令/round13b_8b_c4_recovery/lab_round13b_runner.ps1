<#
================================================================================
第十三轮（补）实验执行器 · 8B C4 低成本恢复（实验室 Windows 单卡 RTX 3090）
生成：WorkBuddy  2026-11-23
自定位：所有路径由 $PSScriptRoot 推导，脚本放在 SpikeLLM 根目录即可运行。
本轮性质：【第十三轮收口】上一轮 8B 四条腿 wikitext2 已出、c4 因 datautils.py
          一个 off-by-one 边界 bug 全崩。本包只补 c4，约 20 分钟，免重训。

--------------------------------------------------------------------------------
【零、背景一句话（不用细读）】

上一轮 LLaMA-3-8B 四腿（FP16/Orig/Full/AQ）的 wikitext2 都正常出来了，
但评估 c4 时崩在 datautils.py 的一个边界 bug（样本长度恰好等于 2048 时，
random.randint(0, -1) 空区间崩溃）。这个 bug 只在 8B（128K 词表）现场生成
c4 缓存时触发，7B 一直走预生成缓存所以从没踩过。
本包修好 datautils.py 后，利用上一轮已经落盘的检查点（omni_parameters.pth），
用 --resume --epochs 0 免重训直接把三条量化腿的 c4 补出来，约 20 分钟。

--------------------------------------------------------------------------------
【一、本包改了什么、没改什么（重要）】

  改：只覆盖 datautils.py 一个文件（旧哈希 54bbaae1c48888e2 -> 新哈希 f0786e8472931dfe）
      修的是第 103/118/167 三处 C4 采样边界：random.randint(0, ...) ->
      random.randint(0, max(0, ...))。对任何长度 > 2048 的样本零改动（随机流一致），
      只在恰好 == 2048 时避免崩溃并取整段。

  不改：main.py（e898ff1da6b92e91）、spike_omniquant.py（4f7ae925a3c1bd51）等
        一律不动。请保持实验室代码与上一轮完全同版，只换 datautils.py 这一个文件。

--------------------------------------------------------------------------------
【二、四步流程（-Mode run 一次跑完，可断线续）】

  步骤 0  打 datautils.py 补丁（原文件备份到 backup_round13b\，哈希不符自动还原并停）
  步骤 1  FP16 腿先跑通，生成 8B 的 c4 缓存（记录 shape + SHA-256）
          -> 落盘 ./cache_llama3_8b/testloader_Llama_c4_all.cache
          预期 shape (1, 524288) = 256 段 x 2048
  步骤 2  校验 3 份检查点完整（orig/full/aq 的 omni_parameters.pth 是否含 32 层 + mask_low_mse）
  步骤 3  三条量化腿 --resume <检查点> --epochs 0 --eval_ppl 补 c4
          -> 每条先验「恢复后的 wikitext2 与原值一致」再采 c4：
              orig 应复现 61.5950，full 应复现 16.5609，aq 应复现 14.7334
          -> 若 WT2 不一致，说明检查点不完整/漂移，该腿标记「需重训」，不要采 c4

--------------------------------------------------------------------------------
【三、怎么跑】

  1. 本包解压到 SpikeLLM 根目录（含 patch_datautils.py + 本 runner，别只拷 ps1）。
  2. 先预检：
       powershell -ExecutionPolicy Bypass -File .\lab_round13b_runner.ps1 -Mode preflight
  3. 开跑（约 20 分钟）：
       powershell -ExecutionPolicy Bypass -File .\lab_round13b_runner.ps1 -Mode run
     或分步：
       -Mode patch        -> 只打 datautils 补丁
       -Mode gencache     -> 只跑 FP16 生成 c4 缓存
       -Mode recover      -> 只跑三条量化腿的 c4 恢复
  4. 回传：lab_round13b_result.txt + log\r13b_*\log_rank0_*.txt + env_round13b_preflight.txt
          + c4 缓存的 shape 与 SHA-256（脚本会自动打印进 result）。

--------------------------------------------------------------------------------
【四、回传清单】

  1. lab_round13b_result.txt            （本脚本自动生成，含四腿 c4 与 WT2 先验对照）
  2. env_round13b_preflight.txt         （环境指纹）
  3. log\r13b_*\log_rank0_*.txt         （四条腿原始日志）
  4. 请在回传时一并说明：三条量化腿恢复后的 WT2 是否与原值（61.5950/16.5609/14.7334）
     逐位一致；若有哪条不一致，该腿的 c4 不要入账，开发组会单独处理。
================================================================================
#>

param(
    [ValidateSet("preflight", "run", "status", "patch", "gencache", "recover", "restore")]
    [string]$Mode = "status",
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

# ---- 常量 ----
$Net8B    = "Llama-3-8B"
$Model8B  = Join-Path $ModelDir $Net8B
$Cache8B  = "./cache_llama3_8b"

# datautils.py 补丁哈希
$PatchSha = "f0786e8472931dfe"
$OldSha   = "54bbaae1c48888e2"

# 三条量化腿：Tag -> 原 wikitext2 值（先验一致性用）
$Legs = @(
    @{ Tag = "r13_8b_orig"; Ref = 61.595027923584; Note = "8B Original SpikeLLM，补 c4" },
    @{ Tag = "r13_8b_full"; Ref = 16.560863494873; Note = "8B Stage 1 Full，补 c4" },
    @{ Tag = "r13_8b_aq";   Ref = 14.7334117889404; Note = "8B Stage 1+2 AQ，补 c4" }
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

# ---- 补丁 ----
function Invoke-Patch {
    Write-Step "打 datautils.py 补丁（原文件备份到 backup_round13b\）"
    $src = Join-Path $Root "patch_datautils.py"
    $dst = Join-Path $Root "datautils.py"
    if (-not (Test-Path $src)) { Fail "补丁源缺失：patch_datautils.py" }

    $bak = Join-Path $Root "backup_round13b"
    New-Item -ItemType Directory -Force -Path $bak | Out-Null

    $cur = Get-Sha16 $dst
    if ($cur -eq $PatchSha) {
        Write-Host "  datautils.py 已是新哈希（$PatchSha），跳过打补丁"
        return
    }
    if ($cur -ne $OldSha) {
        Fail "datautils.py 当前哈希 $cur 既不是旧版也不是新版，请回传 env 指纹给开发组，勿擅自覆盖"
    }

    # 备份
    Copy-Item $dst (Join-Path $bak "datautils.py.bak") -Force
    Copy-Item $src $dst -Force
    $sha = Get-Sha16 $dst
    Write-Host ("  datautils.py  {0}  {1}" -f $sha, $(if ($sha -eq $PatchSha) { "OK" } else { "不符" }))
    if ($sha -ne $PatchSha) { Fail "补丁哈希不符，中止" }
}

function Invoke-Restore {
    Write-Step "从 backup_round13b 还原 datautils.py"
    $bak = Join-Path $Root "backup_round13b\datautils.py.bak"
    if (-not (Test-Path $bak)) { Fail "备份不存在：$bak" }
    Copy-Item $bak (Join-Path $Root "datautils.py") -Force
    Write-Host "  已还原"
}

# ---- FP16 生成 c4 缓存 ----
function Invoke-GenCache {
    Write-Step "FP16 腿先跑通，生成 8B c4 缓存（约 4 分钟）"
    $dir = "./log/r13b_8b_fp16"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $env:CUDA_VISIBLE_DEVICES = "0"
    Set-Location $Root

    $base = @(
        "main.py", "--model", $Model8B, "--net", $Net8B,
        "--wbits", "16", "--abits", "16", "--addbit", "1", "--low_p", "0.9",
        "--let", "--lwc", "--aug_loss",
        "--alpha", "0.75", "--let_lr", "1e-3", "--lwc_lr", "1e-2",
        "--nsamples", "64", "--epochs", "0", "--seed", "2",
        "--cache_dir", $Cache8B
    )

    if (-not (Test-Path "$dir/done_ppl.flag")) {
        Write-Output "[r13b_8b_fp16] fake PPL 开始（首跑会现场生成 c4 缓存）"
        $argv = $base + @("--eval_ppl", "--output_dir", $dir)
        & $Py @argv *>> "$dir/run.log"
        if ($LASTEXITCODE -ne 0) {
            Move-Item "$dir/run.log" "$dir/run.log.failed" -Force -ErrorAction SilentlyContinue
            throw "[r13b_8b_fp16] 失败 (exit $LASTEXITCODE)，见 run.log.failed"
        }
        New-Item -ItemType File -Force -Path "$dir/done_ppl.flag" | Out-Null
        Write-Output "[r13b_8b_fp16] 完成"
    } else {
        Write-Output "[r13b_8b_fp16] done_ppl 已存在，跳过"
    }

    # 打印 c4 缓存 shape + SHA-256
    $cacheFile = Join-Path $Root "$Cache8B\testloader_Llama_c4_all.cache"
    if (Test-Path $cacheFile) {
        $sha = (Get-FileHash -Path $cacheFile -Algorithm SHA256).Hash.Substring(0, 16).ToLower()
        $shape = & $Py -c "import torch; t=torch.load(r'$cacheFile', weights_only=False); print(tuple(t.shape))" 2>$null
        Write-Host ("  c4 缓存 SHA-256(前16) = {0}" -f $sha)
        Write-Host ("  c4 缓存 shape        = {0}  (预期 (1, 524288))" -f $shape)
        # 存进标记文件供 result 汇总读
        Set-Content -Path "$dir/cache_info.txt" -Value ("sha=$sha`nshape=$shape") -Encoding UTF8
    } else {
        Write-Host "  [WARN] c4 缓存未生成：$cacheFile" -ForegroundColor Yellow
    }
}

# ---- 检查点完整性校验 ----
function Test-Checkpoint([string]$tag) {
    $ck = Join-Path $Root "log\$tag\omni_parameters.pth"
    if (-not (Test-Path $ck)) { return $false }
    # 用 python 校验 32 层键 + mask_low_mse 存在
    $script = @"
import torch, sys
ck = r'$ck'
try:
    d = torch.load(ck, map_location='cpu', weights_only=True)
except Exception as e:
    print('LOAD_FAIL', e); sys.exit(0)
keys = list(d.keys())
has_mask = any('mask_low_mse' in k for layer in d.values() for k in (layer.keys() if isinstance(layer, dict) else []))
print('LAYERS', len(keys), 'MASK', has_mask)
"@
    $out = & $Py -c $script 2>$null
    Write-Host ("  {0} 检查点: {1}" -f $tag, $out)
    return ($out -match "LAYERS 32")
}

# ---- 单腿 c4 恢复 ----
function Invoke-RecoverLeg($tag, $ref) {
    $srcCkpt = Join-Path $Root "log\$tag\omni_parameters.pth"
    $dir = Join-Path $Root "log\r13b_$tag"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null

    if (-not (Test-Path $srcCkpt)) {
        Write-Host "[$tag] 跳过：检查点缺失 $srcCkpt（需重训，开发组单独处理）" -ForegroundColor Yellow
        return "NOCKPT"
    }

    $env:CUDA_VISIBLE_DEVICES = "0"
    Set-Location $Root

    # 先验：恢复后 WT2 应与原值一致
    $base = @(
        "main.py", "--model", $Model8B, "--net", $Net8B,
        "--wbits", "4", "--abits", "4", "--addbit", "1", "--low_p", "0.9",
        "--let", "--lwc", "--aug_loss",
        "--alpha", "0.75", "--let_lr", "1e-3", "--lwc_lr", "1e-2",
        "--nsamples", "64", "--epochs", "0", "--seed", "2",
        "--cache_dir", $Cache8B, "--resume", $srcCkpt
    )

    if (-not (Test-Path "$dir/done_ppl.flag")) {
        Write-Output "[$tag] 恢复 + 补 c4 开始（--resume --epochs 0 --eval_ppl）"
        $argv = $base + @("--eval_ppl", "--output_dir", $dir)
        & $Py @argv *>> "$dir/run.log"
        if ($LASTEXITCODE -ne 0) {
            Move-Item "$dir/run.log" "$dir/run.log.failed" -Force -ErrorAction SilentlyContinue
            throw "[$tag] 恢复失败 (exit $LASTEXITCODE)，见 run.log.failed"
        }
        New-Item -ItemType File -Force -Path "$dir/done_ppl.flag" | Out-Null
        Write-Output "[$tag] 完成"
    } else {
        Write-Output "[$tag] done_ppl 已存在，跳过"
    }

    # 读取恢复后 WT2，先验与原值一致
    $f = Get-ChildItem -Path $dir -Filter "log_rank0_*.txt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
    if ($f) {
        $w = (Select-String -Path $f.FullName -Pattern "INFO wikitext2 :" | Select-Object -Last 1)
        $wv = if ($w) { [double](($w.Line -split ":")[-1].Trim()) } else { -1 }
        $diff = [Math]::Abs($wv - $ref)
        if ($diff -lt 0.01) {
            Write-Host ("  [$tag] WT2 先验一致：{0}（原值 {1}，差 {2}）-> c4 可入账" -f $wv, $ref, $diff) -ForegroundColor Green
            return "OK"
        } else {
            Write-Host ("  [$tag] WT2 漂移：{0} vs 原值 {1}（差 {2}）-> c4 不可入账，标记需重训" -f $wv, $ref, $diff) -ForegroundColor Red
            return "DRIFT"
        }
    }
    return "NORESULT"
}

# ---- preflight ----
function Invoke-Preflight {
    Write-Step "第十三轮（补）preflight"
    $ok = $true

    Write-Host "1/7 补丁文件与目标哈希"
    $src = Get-Sha16 (Join-Path $Root "patch_datautils.py")
    Write-Host ("     patch_datautils.py  {0}  {1}" -f $src, $(if ($src -eq $PatchSha) { "OK" } else { "不符" }))
    $cur = Get-Sha16 (Join-Path $Root "datautils.py")
    Write-Host ("     datautils.py 当前   {0}  ({1})" -f $cur, $(if ($cur -eq $PatchSha) { "已是新版" } elseif ($cur -eq $OldSha) { "旧版，run 时会打补丁" } else { "!! 异常，需开发组" }))
    if ($src -ne $PatchSha) { $ok = $false }

    Write-Host "2/7 代码基线（应保持第 13 轮同版，只换 datautils）"
    $main = Get-Sha16 (Join-Path $Root "main.py")
    $omni = Get-Sha16 (Join-Path $Root "spike_driven_quant\spike_omniquant.py")
    Write-Host ("     main.py               {0}  {1}" -f $main, $(if ($main -eq "e898ff1da6b92e91") { "OK" } else { "!! 与第13轮不同版" }))
    Write-Host ("     spike_omniquant.py    {0}  {1}" -f $omni, $(if ($omni -eq "4f7ae925a3c1bd51") { "OK" } else { "!! 与第13轮不同版" }))

    Write-Host "3/7 8B 模型"
    if (Test-Path $Model8B) { Write-Host "     8B  OK  $Model8B" }
    else { Write-Host "     8B  缺失 $Model8B" -ForegroundColor Yellow; $ok = $false }

    Write-Host "4/7 三条量化腿的检查点（omni_parameters.pth）"
    foreach ($l in $Legs) {
        $ck = Join-Path $Root ("log\" + $l.Tag + "\omni_parameters.pth")
        if (Test-Path $ck) {
            $sz = [Math]::Round((Get-Item $ck).Length / 1MB, 1)
            Write-Host ("     OK  {0}  ({1} MB)" -f $l.Tag, $sz)
        } else {
            Write-Host ("     缺失 {0}\omni_parameters.pth（该腿需重训）" -f $l.Tag) -ForegroundColor Yellow
        }
    }

    Write-Host "5/7 GPU0 显存"
    $smi = & nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>$null
    Write-Host "     $smi"

    Write-Host "6/7 磁盘余量"
    $drive = (Get-Item $Root).PSDrive
    $free = [Math]::Round((Get-PSDrive $drive.Name).Free / 1GB, 1)
    Write-Host ("     {0}: 剩余 {1} GB" -f $drive.Name, $free)

    Write-Host "7/7 环境指纹"
    if (Test-Path (Join-Path $Root "env_report.py")) {
        & $Py (Join-Path $Root "env_report.py") --out (Join-Path $Root "env_round13b_preflight.txt") | Out-Null
        if (Test-Path (Join-Path $Root "env_round13b_preflight.txt")) { Write-Host "     OK  env_round13b_preflight.txt" }
    }

    Write-Host ""
    if ($ok) { Write-Host "preflight 通过，可以 -Mode run" -ForegroundColor Green }
    else { Write-Host "preflight 有缺失项，请先补齐" -ForegroundColor Yellow }
}

# ---- 结果汇总 ----
function New-ResultFile {
    $out = Join-Path $Root "lab_round13b_result.txt"
    $L = New-Object System.Collections.ArrayList
    [void]$L.Add("SpikeLLM / OSDSQ v2  第十三轮（补）8B C4 恢复结果（实验室单卡 3090）")
    [void]$L.Add("生成时间: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    [void]$L.Add("项目路径: " + $Root)
    [void]$L.Add("任务: 修 datautils off-by-one 后，免重训补出 8B 三条量化腿的 c4")
    [void]$L.Add("")
    [void]$L.Add("---- 补丁与基线哈希 ----")
    [void]$L.Add("  datautils.py  " + (Get-Sha16 (Join-Path $Root "datautils.py")) + "  (应 = f0786e8472931dfe)")
    [void]$L.Add("  main.py       " + (Get-Sha16 (Join-Path $Root "main.py")) + "  (应 = e898ff1da6b92e91)")
    [void]$L.Add("  spike_omniquant.py " + (Get-Sha16 (Join-Path $Root "spike_driven_quant\spike_omniquant.py")) + "  (应 = 4f7ae925a3c1bd51)")
    [void]$L.Add("")
    [void]$L.Add("---- FP16 腿 + c4 缓存 ----")
    $fpDir = Join-Path $Root "log\r13b_8b_fp16"
    $ci = Join-Path $fpDir "cache_info.txt"
    if (Test-Path $ci) { Get-Content $ci -Encoding UTF8 | ForEach-Object { [void]$L.Add("  " + $_) } }
    $fpf = Get-ChildItem -Path $fpDir -Filter "log_rank0_*.txt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
    if ($fpf) {
        $w = (Select-String -Path $fpf.FullName -Pattern "INFO wikitext2 :" | Select-Object -Last 1)
        $c = (Select-String -Path $fpf.FullName -Pattern "INFO c4 :" | Select-Object -Last 1)
        [void]$L.Add("  FP16 wikitext2 : " + $(if ($w) { ($w.Line -split ":")[-1].Trim() } else { "NA" }))
        [void]$L.Add("  FP16 c4        : " + $(if ($c) { ($c.Line -split ":")[-1].Trim() } else { "NA" }))
    }
    [void]$L.Add("")
    [void]$L.Add("---- 三条量化腿 c4 恢复（先验 WT2 一致才可入账）----")
    foreach ($l in $Legs) {
        $dir = Join-Path $Root ("log\r13b_" + $l.Tag)
        [void]$L.Add("### " + $l.Tag)
        [void]$L.Add("  说明: " + $l.Note)
        $f = Get-ChildItem -Path $dir -Filter "log_rank0_*.txt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
        if ($f) {
            $w = (Select-String -Path $f.FullName -Pattern "INFO wikitext2 :" | Select-Object -Last 1)
            $c = (Select-String -Path $f.FullName -Pattern "INFO c4 :" | Select-Object -Last 1)
            $wv = $(if ($w) { ($w.Line -split ":")[-1].Trim() } else { "NA" })
            $cv = $(if ($c) { ($c.Line -split ":")[-1].Trim() } else { "NA" })
            [void]$L.Add("  [fake] wikitext2 : " + $wv + "   (原值 " + $l.Ref + ")")
            [void]$L.Add("  [fake] c4        : " + $cv)
        } else {
            [void]$L.Add("  [fake] 未找到 log_rank0_*.txt（该腿可能未恢复）")
        }
        [void]$L.Add("")
    }
    [void]$L.Add("---- 环境指纹 ----")
    if (Test-Path (Join-Path $Root "env_round13b_preflight.txt")) {
        [void]$L.Add("  见 env_round13b_preflight.txt")
    } else {
        [void]$L.Add("  未生成。请补跑：python env_report.py --out env_round13b_preflight.txt")
    }
    Set-Content -Path $out -Value $L -Encoding UTF8
    Write-Host "结果已写入 $out"
}

# ---- 状态 ----
function Invoke-Status {
    Write-Step "第十三轮（补）状态"
    Write-Host ("  datautils.py        " + (Get-Sha16 (Join-Path $Root "datautils.py")))
    Write-Host ("  FP16 c4 缓存        " + $(if (Test-Path (Join-Path $Root "$Cache8B\testloader_Llama_c4_all.cache")) { "已生成" } else { "未生成" }))
    foreach ($l in $Legs) {
        $dir = Join-Path $Root ("log\r13b_" + $l.Tag)
        $p = Test-Path (Join-Path $dir "done_ppl.flag")
        Write-Host ("  {0,-16} {1}" -f $l.Tag, $(if ($p) { "DONE" } else { "未开始" }))
    }
}

# ---- 主入口 ----
switch ($Mode) {

    "preflight" { Invoke-Preflight }

    "status" { Invoke-Status }

    "patch" { Invoke-Patch }

    "restore" { Invoke-Restore }

    "gencache" { Invoke-GenCache }

    "recover" {
        foreach ($l in $Legs) {
            try { Invoke-RecoverLeg $l.Tag $l.Ref | Out-Null }
            catch { Write-Host "[WARN] 腿 $($l.Tag) 失败：$_" -ForegroundColor Yellow }
        }
        New-ResultFile
    }

    "run" {
        Write-Step "第十三轮（补）：8B c4 恢复（约 20 分钟，可断线续）"
        $start = Get-Date

        Write-Host "步骤 0：打 datautils 补丁"
        try { Invoke-Patch } catch { Fail $_ }

        Write-Host "步骤 1：FP16 生成 c4 缓存"
        try { Invoke-GenCache } catch { Write-Host "[WARN] $_" -ForegroundColor Yellow }

        Write-Host "步骤 2：校验三条量化腿检查点"
        foreach ($l in $Legs) { Test-Checkpoint $l.Tag }

        Write-Host "步骤 3：三条量化腿免重训补 c4"
        foreach ($l in $Legs) {
            try { Invoke-RecoverLeg $l.Tag $l.Ref | Out-Null }
            catch { Write-Host "[WARN] 腿 $($l.Tag) 失败：$_" -ForegroundColor Yellow }
        }

        New-ResultFile
        $elapsed = (Get-Date) - $start
        Write-Host ("全部结束，用时 {0:N1} 分钟" -f $elapsed.TotalMinutes)
        Write-Step "结果预览"
        Get-Content (Join-Path $Root "lab_round13b_result.txt") -Encoding UTF8 | Select-Object -First 60
    }
}
