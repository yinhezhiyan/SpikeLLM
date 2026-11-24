================================================================================
第七轮实验包  2026-11-02
================================================================================

【怎么用】

1. 把本压缩包解压到 SpikeLLM 根目录（与 main.py 同一层），选择"合并/覆盖"。
   解压后应当看到：
       SpikeLLM\lab_round7_runner.ps1        <- 执行器（头部注释即本轮日志）
       SpikeLLM\env_report.py                <- 环境指纹脚本
       SpikeLLM\act_scales\llama-68m.pt      <- 68M 门禁用（70 KB）
       SpikeLLM\act_shifts\llama-68m.pt
       SpikeLLM\README_第七轮.txt            <- 本文件

2. 先体检（约 30 秒，不跑训练）：
       powershell -ExecutionPolicy Bypass -File .\lab_round7_runner.ps1 -Mode preflight

3. 正式执行（约 8~9 小时）：
       powershell -ExecutionPolicy Bypass -File .\lab_round7_runner.ps1 -Mode run

   如果 HF 连不上、不想等 68M 那 270 MB 下载，直接加 -Skip68M：
       powershell -ExecutionPolicy Bypass -File .\lab_round7_runner.ps1 -Mode run -Skip68M

4. 中途看进度：
       powershell -ExecutionPolicy Bypass -File .\lab_round7_runner.ps1 -Mode status

5. 断线/关机后重跑同一条 -Mode run 命令即可，已完成的作业会自动跳过。


【本轮跑什么】

  步骤 1（约 10 分钟，可选）
      68M  --shield_mode zero  seed2
      对照：68M mean 基线 156.84 / 90.53

  GPU0（约 8 小时）
      7b_seed2_zero   seed=2  --shield_mode zero
      对照：已有 mean 结果 7.9792 / 10.6792

  GPU1（约 8 小时）
      7b_seed4_ours   seed=4
      对照：已有 Original seed4 = 10.5389 / 13.7700

  本轮**不改任何算法代码**，只加参数、只跑对照。


【本轮要回答的问题】

  必须用均值锚定吗？换成零填充（v1 做法）会不会差不多？

  如果 7B 上 zero 与 mean 的差距 < 0.3 PPL，说明 A 只是个等价实现，
  核心创新点不成立，请立刻通知开发组，不要继续排后续实验。
  如果差距 >= 1.0 PPL，说明 A 的必要性成立，核心创新点站得住。


【判据（脚本也会写进结果 txt）】

  7B zero vs 7.9792：
      >= 1.0   -> A 的必要性成立
      0.3~1.0  -> 弱证据，需补逐层量程统计
      < 0.3    -> 核心创新点不成立

  7B seed4-Ours vs 10.5389（Original seed4）：预期降幅 23% ~ 25%


【需要回传的文件】

  1. lab_round7_result.txt        <- 脚本自动生成，纯文本
  2. env_round7_preflight.txt     <- 环境指纹
  3. log\r7_*\ 下的 log_rank0_*.txt（如果结果数字有缺失或异常）
  4. 若某条腿失败：对应的 run.log.failed


【三条提醒（前几轮踩过的坑）】

  1. 安全删除守卫会杀进程：表现为"无 traceback、exit 1、日志停在某一行"。
     遇到后先别重跑，先核对已落盘文件的字节数与 sha256，能就地提交就别重下。

  2. runner 可能在最后一条腿完成瞬间就写结果 txt，必须事后核对数字行是否齐全。
     缺行就再跑一次 -Mode run（有 done.flag 的作业自动跳过，秒级）。

  3. 本机写 .ps1 必须 UTF-8 BOM，否则 PS 5.1 按 ANSI 解码中文注释会解析错位。
     本包内脚本已带 BOM，无需改动。


【本轮不做】

  - 不跑 8B（权重已就位 16.06 GB，但 7B 的 PIQA 峰值已达 22.5/24 GB，8B 必 OOM，需单独排期）
  - 不跑零样本（单腿会从 8 小时涨到 14~17 小时）
  - 不改 shield_mode 默认值（仍是 mean，不改变任何现有默认数值路径）

================================================================================
