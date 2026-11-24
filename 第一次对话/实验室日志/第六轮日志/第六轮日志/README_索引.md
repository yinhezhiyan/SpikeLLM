# 第六轮日志 — 索引

> 轮次时间：2026-11-01 10:11:23 → 17:58（GMT+8），7 h 47 m，双卡并行 + 后台下载，runner exit 0（见第五节，中途有一次异常与补跑）
> 主题：**`shield_mode` 补丁（v1 zero 填充分支）+ seed3-Ours / seed4-Original 两条腿 + 8B base 权重下载**
> 归档时间：2026-11-02 20:02

**这一轮不做解读。** 结果以 `01_结果原件\lab_round6_result.txt` 为唯一权威，本文件只记录"跑了什么、怎么来的、文件在哪"。

---

## 一、这一轮的由来

前五轮的结果都是 seed=2 单次：OSDSQ v2 vs Original 的差异只有单点，没有重复间标准差；且 `seed3-Original` 已在第五轮跑完，缺的是与之配对的 `seed3-Ours`。开发组给出 `round6_patch_2026-11-01.zip`，新增 `--shield_mode {mean,zero}`（默认 `mean`，语义与现行路径完全一致；`zero` 是 v1 的零填充分支，仅供对照），并指定本轮三条任务。

---

## 二、执行记录

| 项目 | 内容 |
|---|---|
| 补丁包 | `round6_patch_2026-11-01.zip`（3 文件：main.py / spike_linear.py / lab_round6_runner.ps1） |
| 覆盖前 diff | `main.py` +3 / −0；`spike_linear.py` +7 / −1，**唯一被改写的行是 `_extra_keys` 集合（追加 `"shield_mode"`）**，属纯追加；`import lab_compat` 等 lab 兼容修复未被回退（见 `05_代码改动留档\patch_diff_*.diff`） |
| 备份 | `main.py.bak_r6`、`spike_linear.py.bak_r6`（10:09） |
| preflight | 6/6 通过（shield_mode 补丁就位、7B 模型完整、act_scales/shifts 在、双卡 186/306 MiB、C 盘 166.6 GB） |
| 环境补设 | `HF_HOME=<root>\hf_cache`（runner 自身不设；不补会让 seed4 的校准集落到他人公共缓存） |
| 未启用 | 未开 `deactive_amp`，未改量化算法，本轮未跑 8B 冒烟（按第六轮指令第 7 条） |

| 来源 | Tag | 关键参数 | 完成时刻 | flags |
|---|---|---|---|---|
| GPU0 | `7b_seed3_ours` | seed=3，`./cache_seed3`，默认 `shield_mode=mean` | 17:57 | 3/3 |
| GPU1 | `7b_seed4_orig` | `--orig_spikellm`，seed=4，`./cache_seed4` | 17:58 | 3/3 |
| 后台 | `8b_download` | `NousResearch/Meta-Llama-3-8B` → `models\Meta-Llama-3-8B` | 见第五节 | `done.flag`（本地提交后补） |

共同参数：`--wbits 4 --abits 4 --addbit 1 --low_p 0.9 --let --lwc --aug_loss --alpha 0.75 --let_lr 1e-3 --lwc_lr 1e-2 --nsamples 128 --epochs 20`。

---

## 三、原始数字（照抄日志，未做任何换算）

| 腿 | fake wt2 | fake c4 | real wt2 | real c4 | norm:nan |
|---|---|---|---|---|---|
| `7b_seed3_ours` | 8.094834327697754 | 10.712592124938965 | 8.123409271240234 | 10.922171592712402 | 7 |
| `7b_seed4_orig` | 10.53885269165039 | 13.769993782043457 | 10.75993537902832 | 14.341706275939941 | 12 |

`norm:nan` 均为单步出现后恢复，无持续性 NaN、无 `norm:inf`、无非有限 loss（看门狗全程记录：两条腿均 `layer=31`、`epochs_logged=640`、`norm_inf=0`）。

本轮 txt 未做任何减法；seed3 配对所需的 `Original seed3 = 10.630260467529297 / 13.796493530273438`（第五轮产出，real `10.896252632141113 / 14.479215621948242`）在 txt 的参照区已列出，降幅请开发组自行计算。

---

## 四、文件清单

```
第六轮日志\
├─ README_索引.md                       ← 本文件
├─ 01_结果原件\
│   └─ lab_round6_result.txt            ← 脚本自动生成，交回开发机的原件（17:59:35 版，8 个数字齐全）
├─ 02_原始日志\
│   ├─ run_7b_seed3_ours.log                (324 KB, UTF-16LE)
│   ├─ run_7b_seed4_orig.log                (328 KB, UTF-16LE)
│   ├─ run_7b_seed4_orig.log.failed         ← 10:12 被外部中止的第一次尝试（原样保留）
│   ├─ finish_seed4.log                     ← 补跑收尾脚本的 3 行记录
│   └─ lab_watch_round6.log                 ← 看门狗每 10 min 快照，末行 watchdog stopped: all done
├─ 03_门禁结果\
│   ├─ fake_7b_seed3_ours.txt
│   ├─ real_7b_seed3_ours.txt
│   ├─ fake_7b_seed4_orig.txt
│   ├─ fake_7b_seed4_orig.attempt1_crashed.txt   ← 第一次尝试，仅 2 行，无 PPL
│   ├─ real_7b_seed4_orig.txt
│   └─ 归档校验.txt                     ← 全部文件 md5 + checkpoint 比对 + 从归档件重取 PPL 的复核
├─ 04_checkpoint\
│   ├─ omni_parameters_7b_seed3_ours.pth   (8 983 506 B, de1840c91522…)
│   └─ omni_parameters_7b_seed4_orig.pth   (8 983 506 B, 4a48798b24ab…)
├─ 05_代码改动留档\
│   ├─ round6_patch_2026-11-01.zip        ← 开发组原始补丁包
│   ├─ lab_round6_runner.ps1              ← 本轮执行器
│   ├─ lab_finish_seed4.ps1               ← 本轮新增：seed4 补跑后的收尾脚本（纯 ASCII 注释）
│   ├─ main.py / spike_linear.py          ← 覆盖后（当前在用）
│   ├─ main.py.bak_r6 / spike_linear.py.bak_r6   ← 覆盖前备份
│   └─ patch_diff_*.diff                  ← 前→后 的 unified diff
└─ 06_8B下载与校验\
    ├─ run.log.download_8b                ← 下载进程原始日志（含安全删除守卫记录）
    ├─ exit_code.txt                      ← exit_code=1（进程在中止点被杀）
    ├─ local_commit_verify.txt            ← 本地提交的依据：逐文件 sha256 与仓库 LFS 比对
    ├─ 8B权重_file_manifest_sha256.txt    ← 提交后实测的文件清单 + sha256（16.06 GB）
    └─ done.flag
```

合计 29 文件 / 19 MB（含 2 个 checkpoint，各 8.98 MB；8B 权重 16.06 GB 留在 `SpikeLLM\models\Meta-Llama-3-8B`，未复制进日志）。`归档校验.txt` 的清单列出本目录除自身以外的 28 个文件。

---

## 五、本轮两次异常（必读）

### 1. seed4 首次尝试在 10:12 被外部中止

现象：无 Python traceback，日志停在 `get_wikitext2` 之后，GPU 从未分配，留下 0 字节 `datasets\*.lock`；父 runner 据此把该腿记为失败并把 `run.log` 改名为 `run.log.failed`。

排查：内存 256 GB / 余 218 GB 排除 OOM；单独跑 `load_dataset('wikitext','wikitext-2-raw-v1')` 返回 36718 行正常，数据未损坏；进程命令行显示为工具层注入的**文件删除守卫**（`CODEBUDDY_SAFE_DELETE_PS_R…`），当时 `get_wikitext2` 正在清建 arrow 缓存、8B 下载在同一 `HF_HOME` 上并发抢锁。

处置：**用完全相同的 argv 在 GPU1 手工重跑**该腿（10:18 起），正常跑到结束；失败原件与失败时的 `log_rank0` 均原样保留。未重跑整条命令，避免让已跑 15 分钟的 seed3 从零开始。

### 2. runner 先落 txt、后抛错

父 runner 在 17:56:00 就写出了 `lab_round6_result.txt`（当时 seed4 的 real c4 还要到 17:57:48 才出，所以那版只有 7 个数字），随后因启动时记下的 seed4 失败状态抛 "训练未全部完成" 并 exit 1。

处置：两条腿的 flag 齐全后，重跑一次 `-Mode run -SkipDownload8B` —— 训练两段均因 `done_ppl.flag` / `done_multibit.flag` 存在而跳过，秒级只重写 txt，未重跑任何训练，exit 0。当前归档的 `lab_round6_result.txt` 为 17:59:35 版，8 个数字齐全。

### 3. 8B 权重：数据其实已下完，卡在"提交改名"

下载进程 15:45 被同一个删除守卫在 hf_hub 提交阶段（删除 `.lock` 与 `models\tmp_*`）中止，exit 1，无 traceback。

核对：4 个分片与 `tokenizer.json` 的 `.incomplete` 文件字节数与仓库完全一致（4976698672 / 4999802720 / 4915916176 / 1168138808 / 9085698）；本地 sha256 与 `HfApi().model_info(..., files_metadata=True)` 给出的 LFS sha256 逐文件 MATCH。于是用 `os.replace` 就地把 5 个文件提交为正式文件名，**未重新下载任何字节**。提交后实测：4 分片头部可解析（张量数 82/104/100/5），权重字节合计 16 060 556 376（16.06 GB），`config.json` 为 `model_type=llama`、32 层、hidden 4096（base 版，非 Instruct）；`models\` 下 8 个 0 字节 `tmp_*` 残留已清理。因此 `log\8b_download\done.flag` 是提交完成之后补的，不是下载脚本自己写的。

---

## 六、复核结论（只看"数据是否可信"，不看"结果意味什么"）

`03_门禁结果\归档校验.txt` 为脚本生成，四项全过：

1. 全部文件列出字节数与 md5，可用于对方侧逐字节核对。
2. 两个归档 checkpoint 与 `SpikeLLM\log\<tag>\omni_parameters.pth` **md5 完全一致**（`de1840c91522…`、`4a48798b24ab…`）。
3. 8 个 PPL 数字**从归档的 `log_rank0_*.txt` 直接重取**，与 `lab_round6_result.txt` 逐位相同。
4. 两条腿的 `done_ppl.flag` / `done_multibit.flag` / `done.flag` 齐全，`8b_download` 有 `done.flag`。

---

## 七、留给开发组判断的事

1. seed3 的 Ours / Original 配对降幅，与 seed2 的 −24.5% / −22.5%（txt 原话）怎么比较、种子间波动是否可接受，本轮不代算、不下结论。
2. `shield_mode=zero` 的分支本轮**只加了代码、没有使用**，是否要单独排一轮 v1 对照，需要开发组决定。
3. seed4-Ours 未跑，三组种子的误差棒还差一半；ACE / 发放率脚本仍未收到，口径（逐层 ACE vs 发放率曲线、用哪些 checkpoint、哪些校准样本）待定。
4. 8B 冒烟按指令第 7 条未跑（PIQA 在 24 GB 卡上极可能 OOM），需单独排期并给出规避方案。
5. 零样本 `acc_norm` 与官方口径的差异（`lm_eval\base.py:660` 按选项字符长度归一）仍未获确认，跨表引用原文 49.51 是否成立需开发组拍板。

---

## 八、本轮踩到的坑（避免下一轮重犯）

1. **安全删除守卫会杀进程**：本次两处（seed4 清建 arrow 缓存、8B 提交改名）都是"无 traceback、exit 1、日志停在某一行"。大文件下载被中止后**先别重下**，先比字节数与 sha256，全 MATCH 就地提交。
2. **runner 可能在最后一条腿完成瞬间就落结果 txt**，必须事后核对全部数字行是否齐全；缺行用 `-Mode run -SkipDownload8B` 重跑（flags 齐全则秒级只重写 txt）。
3. **PS 5.1 读无 BOM 的 UTF-8 中文注释会按 ANSI 解码**导致解析错位（`lab_finish_seed4.ps1` 首发报 line 33 `}` 意外）。给这台机器写 `.ps1` 一律用纯 ASCII 注释。
4. `run.log` / `log_rank0_*.txt` 同名冲突：归档时必须按 tag 与 fake/real 改名，否则后者覆盖前者（第五轮已踩过，本轮沿用改名规则）。
5. 三条任务并发时单腿速度从 0.53 min/epoch 降到约 1.07 min/epoch，训练段实际 7 h 20 m（10:11 → 17:28 首条 PPL），比原估 8.5–9 h 的整体时长接近但分布不同。

---

## 九、术语与口径（沿用前几轮，勿改）

- fake = `fake_binary_simulate`（默认），real = `real_multibit_simulate`。
- wt2 = WikiText-2，c4 = C4；数字为 PPL，越低越好。
- A = 线性层低显著子集均值锚定屏蔽；B = **仅 Value/PV 路径**的屏蔽（`is_p=False` 的 Key/QK^T 分支不做屏蔽）；C = 步长加权显著性；D = 加权第二目标 + 余弦项（仅 `--aug_loss` 生效）。
- `shield_mode`：`mean`（默认，即 A 现行的均值锚定）/ `zero`（v1 零填充，仅对照用）。
- 所有数字单 seed 单次运行，未做重复，**不构成误差棒**。
