<#
第十二轮单卡实验室命令。请先结束旧执行器，解压整包到 SpikeLLM 根目录。
在该目录运行 powershell -ExecutionPolicy Bypass -File .\lab_round12_single.ps1 -Mode run
默认沿用当前 CUDA_VISIBLE_DEVICES 中第一张卡，未指定时用 0。指定物理卡时加 -Gpu 1。
preflight 只预检，gate 安装补丁并检查 68M 的 WT2 和 C4。run 包含两者，再串行执行
7B adaptive seed4、random seed2、heuristic_down_o seed2，最后用第十一轮 AQ seed2 做六任务零样本。
校验通过的 PPL 原始记录支持续跑。门禁失败停止，回滚本次新装的补丁。
中断后重新运行同一命令。模型和缓存沿用实验室配置，环境变量沿用第十一轮。
完成或失败后回传项目根目录 lab_round12_return_codex.zip，内含纯文本结果、错误及原始日志。
#>
param([ValidateSet("run","gate","preflight")][string]$Mode="run",
      [string]$Gpu="", [string]$Py="", [string]$ProjectRoot="")
$ErrorActionPreference="Stop"
if (!$ProjectRoot) {
    $ProjectRoot=$PSScriptRoot
    if (!(Test-Path (Join-Path $ProjectRoot "main.py"))) { $ProjectRoot=Split-Path $ProjectRoot -Parent }
}
if (!(Test-Path (Join-Path $ProjectRoot "main.py"))) { throw "Set -ProjectRoot to the SpikeLLM folder" }
if (!$Gpu) {
    if (Test-Path Env:CUDA_VISIBLE_DEVICES) {
        $Gpu=($env:CUDA_VISIBLE_DEVICES -split ",")[0].Trim()
    } else { $Gpu="0" }
}
if ($Gpu -notmatch '^(\d+|GPU-[0-9a-fA-F-]+)$') { throw "Select one CUDA device" }
if (!$Py) {
    $Py="python"
    foreach ($p in @("$ProjectRoot\.venv\Scripts\python.exe","$ProjectRoot\..\.venv\Scripts\python.exe","$ProjectRoot\venv\Scripts\python.exe")) {
        if (Test-Path $p) { $Py=$p; break }
    }
}
$env:CUDA_VISIBLE_DEVICES=$Gpu
Push-Location -LiteralPath $ProjectRoot
try {
    & $Py -B (Join-Path $PSScriptRoot "lab_round12_dispatch.py") $Mode
    $rc=$LASTEXITCODE
} finally { Pop-Location }
exit $rc
