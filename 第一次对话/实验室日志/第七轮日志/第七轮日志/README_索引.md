# 第七轮日志 — 索引

> 轮次时间：2026-11-02 20:18（preflight）→ 2026-11-03 04:24，训练段 20:37:11 → 04:24，7 h 47 m，双卡并行，runner exit 0
> 主题：**`shield_mode=zero` 零填充对照（68M 门禁 + 7B 核心）+ seed4-Ours（第三组种子配对）**
> 归档时间：2026-11-03 09:57

**这一轮不做解读。** 结果以 `01_结果原件\lab_round7_result.txt` 为唯一权威，本文件只记录"跑了什么、怎么来的、文件在哪、核查了什么"。

---

## 一、这一轮的由来

第六轮补丁加了 `--shield_mode {mean,zero}`，但 `zero` 分支只加了代码、没跑过。开发组给 `round7_patch_2026-11-02.zip`，指定本轮三条任务：

1. 68M 的 `zero` 门禁（10 分钟级），对照 mean 基线判断小模型上 A 是否敏感；
2. 7B 的 `zero` 对照（核心），判据见 txt：差值 ≥1.0 认为 A 必要、0.3~1.0 弱证据、<0.3 认为核心创新点不成立；
3. `seed4-Ours`，与第六轮 `seed4-Original` 配成第三组种子，形成三组误差棒。

补丁本身**只新增文件、不覆盖任何现有代码**（`env_report.py`、`lab_round7_runner.ps1`、`README_第七轮.txt`、`act_scales|act_shifts\llama-68m.pt` 各 70 048 B）。

---

## 二、执行记录

| 项目 | 内容 |
|---|---|
| 补丁包 | `round7_patch_2026-11-02.zip`（5 文件，纯新增） |
| 解压方式 | zip 用**反斜杠当路径分隔符**，`unzip` 与 `Expand-Archive` 都会报错/散架；改用 `zipfile` + `name.replace('\\','/')` 落盘 |
| 对 runner 的唯一改动 | `$ErrorActionPreference` 由 `"Stop"` 改为 `"Continue"`（留 `NOTE (lab machine fix 2026-11-02)` 注释），见 `05_代码改动留档\patch_diff_lab_round7_runner.ps1.diff`（+4 / −1） |
| 备份 | `lab_round7_runner.ps1.bak_r7`（补丁原件） |
| 解释器 | 必须显式传 `-Py "C:\Users\admin\.conda\envs\spikellm\python.exe"`；脚本默认找 `.venv\Scripts\python.exe`，本机无 `.venv`，回退到 PATH 上的 anaconda 3.12（无 torch） |
| preflight | 6/6 通过：代码 md5 `main.py f201ca3a429f0e16` / `spike_linear 26e84e8f1a460d43` 与开发机一致；双卡各剩 24 GB；C 盘 151 GB；见 `06_环境与预检\env_round7_preflight.txt` |
| 环境补设 | `HF_HOME=<root>\hf_cache`、`HF_ENDPOINT=https://hf-mirror.com`、`HF_DATASETS_TRUST_REMOTE_CODE=1` |
| 未启用 | 未开 `deactive_amp`，未改量化算法，未跑 8B 冒烟 |

| 顺序 | Tag | 关键参数 | 起止 | flags |
|---|---|---|---|---|
| ① GPU0 | `r7_68m_zero` | `JackFram/llama-68m`，seed=2，`./cache`，`--shield_mode zero` | 20:22 → 20:36 | 3/3 |
| ② GPU0 | `r7_7b_seed2_zero` | `Llama-2-7b-hf`，seed=2，`./cache`，`--shield_mode zero` | 20:37 → 03:50(real 起) | 3/3 |
| ③ GPU1 | `r7_7b_seed4_ours` | `Llama-2-7b-hf`，seed=4，`./cache_seed4`，默认 `mean` | 20:37 → 04:24 | 3/3 |
| ④ GPU0 | `r7_68m_mean_check` | 与 ① 完全同参，**仅省略 `--shield_mode`**（=mean），只跑 fake | 04:27 → 04:42 | 无（诊断腿） |

共同参数：`--wbits 4 --abits 4 --addbit 1 --low_p 0.9 --let --lwc --aug_loss --alpha 0.75 --let_lr 1e-3 --lwc_lr 1e-2 --nsamples 128 --epochs 20`。

`llama-68m` 首次下载 261 MB（约 1 分钟），落在 `hf_cache\hub`，来源与 sha256 见 `06_环境与预检\68M模型与act_stats来源.txt`。

---

## 三、原始数字（照抄日志，未做任何换算）

| 腿 | fake wt2 | fake c4 | real wt2 | real c4 | norm:nan |
|---|---|---|---|---|---|
| `r7_68m_zero` | 157.7880401611328 | 90.83118438720703 | 154.45602416992188 | 89.72161865234375 | 3 |
| `r7_68m_mean_check`（诊断） | 157.7880401611328 | 90.83118438720703 | — | — | 3 |
| `r7_7b_seed2_zero` | 7.979187488555908 | 10.679176330566406 | 8.157027244567871 | 10.89747142791748 | 5 |
| `r7_7b_seed4_ours` | 8.037086486816406 | 10.713491439819336 | 8.084365844726562 | 10.926255226135254 | 5 |

`norm:nan` 均为单步出现后恢复，无持续性 NaN、无 `norm:inf`（看门狗记录：两条 7B 腿均 `layer=31`、`epochs_logged=640`）。

本文件不做减法。三组种子的 Ours / Original 配对降幅请开发组自行计算；既有参照值已在 txt 的参照区列出（Original seed2/3/4、Ours seed2/seed3、A-off seed2）。

---

## 四、文件清单

```
第七轮日志\
├─ README_索引.md                       ← 本文件
├─ 01_结果原件\
│   └─ lab_round7_result.txt            ← 脚本生成，交回开发机的原件（12 个 PPL 行齐全）
├─ 02_原始日志\
│   ├─ run_r7_68m_zero.log                    ( 65 KB, UTF-16LE)
│   ├─ run_r7_7b_seed2_zero.log               (324 KB, UTF-16LE)
│   ├─ run_r7_7b_seed4_ours.log               (324 KB, UTF-16LE)
│   ├─ run_r7_68m_mean_check.log              ( 36 KB, 诊断腿，本机 mean 对照)
│   └─ lab_watch_round7.log                   ← 看门狗每 10 min 快照，末行 watchdog stopped: all done
├─ 03_门禁结果\
│   ├─ fake_/real_ × 三条正式腿 (6 份 log_rank0)
│   ├─ fake_r7_68m_mean_check.txt       ← 诊断腿的评测输出
│   ├─ zero_mean_等价性核查.txt          ← 脚本生成：5 项独立核查（见第五节）
│   └─ 归档校验.txt                     ← 脚本生成：全部文件 md5 + checkpoint 比对 + 从归档件重取 PPL
├─ 04_checkpoint\
│   ├─ omni_parameters_r7_68m_zero.pth          (139 170 B, a9bdb276…)
│   ├─ omni_parameters_r7_68m_mean_check.pth    (139 170 B, a9bdb276… 与零填充完全相同)
│   ├─ omni_parameters_r7_7b_seed2_zero.pth     (8 983 506 B, 45c1b5df… 与 mean 基线完全相同)
│   └─ omni_parameters_r7_7b_seed4_ours.pth     (8 983 506 B, 768622d1…)
├─ 05_代码改动留档\
│   ├─ round7_patch_2026-11-02.zip      ← 开发组原始补丁包
│   ├─ lab_round7_runner.ps1            ← 本轮执行器（已改 ErrorActionPreference）
│   ├─ lab_round7_runner.ps1.bak_r7     ← 补丁原件（Stop 版）
│   ├─ patch_diff_lab_round7_runner.ps1.diff
│   ├─ env_report.py                    ← 补丁自带的环境指纹脚本
│   └─ README_第七轮.txt                ← 补丁自带说明
└─ 06_环境与预检\
    ├─ env_round7_preflight.txt
    ├─ 68M模型与act_stats来源.txt        ← 模型 revision + 四个文件 sha256，act_stats sha256
    ├─ act_scales_llama-68m.pt          (70 048 B)
    └─ act_shifts_llama-68m.pt          (70 048 B)
```

合计 29 文件 / 19 MB（含 4 个 checkpoint）。68M 权重 261 MB 与 7B 权重 12.56 GB 留在原处，未复制进日志。

---

## 五、本轮最重要的核查：`zero` 与 `mean` 逐位等价

**现象**：`r7_7b_seed2_zero` 的四个 PPL 与既有 mean 基线（`log\7b_osdsq_v2_with_let`）**完全相同**，差值 0.000。按 txt 判据会落进「<0.3 → 核心创新点不成立」。但三条腿的实参、flag、日志都正常，因此先做核查再报。

`03_门禁结果\zero_mean_等价性核查.txt` 为脚本现场生成，五项：

1. **checkpoint 逐位比对**：7B zero 的 md5 `45c1b5df88acaefee0fa41fac513f17b` 与 mean 基线**完全相同**；68M zero 与 68M 本机 mean 亦相同（`a9bdb276…`）。
2. **训练轨迹逐行比对**：7B seed2 的 635 条 epoch 记录（含 NaN 计数 5）与 mean 基线**逐位相同**，首条/末条 loss 与 norm 完全一致。
3. **合成实验定位分歧条件**：低显著子集正负混合时 `max|mean−zero| = 0`（逐位相同）；子集全为正时才差 0.268。
4. **接线核查**：直接构造 `SpikeQuantLinear` 并传 `shield_mode`，`self.shield_mode` 分别得到 `mean` / `zero`，参数确实被消费；运行日志中的 Namespace 也是 `shield_mode='zero'`。
5. **同机同参 68M 对照**：本机补跑 mean（`log\r7_68m_mean_check`，15 min，未改算法、只省略 `--shield_mode`）得 **157.7880401611328 / 90.83118438720703**，与 zero 腿逐位相同、NaN 计数同为 3。

**原因**：两个分支只改 `mask_low` 之外的通道取值，而输出还要再乘 `mask_low`，故结果只取决于 `act_quantizer` 的 min/max；`mean_low` 与 `0` 都落在低显著子集的 `[xmin, xmax]` 内时不改变量程 → 输出逐位相同。只有子集全正或全负时，`0` 才把区间拉到 0，此时才分歧。

**因此**：`zero` 不是有效的 A 对照（它并没有削弱 A 的机制，只是换了一个同样落在区间内的填充值）。A 的必要性证据仍为第五轮的 A-off：**10.4181 / 12.9691**（real 10.5408 / 13.5234）vs mean 7.9792 / 10.6792。

**另一个要请开发组定的数**：runner 写的 68M mean 基线 `156.84 / 90.53` 在本机**不可复现**（本机 mean = 157.7880 / 90.8312）；仓库 `generate_paper_figures.py` 里还有第三个值 `156.09 / 90.92`。三值并存，68M 只能做**同机同版**对照。

---

## 六、复核结论（只看"数据是否可信"，不看"结果意味什么"）

`03_门禁结果\归档校验.txt` 为脚本生成：

1. 全部 28 个文件列出字节数与 md5，可用于对方逐字节核对。
2. 4 个归档 checkpoint 与 `SpikeLLM\log\<tag>\omni_parameters.pth` **md5 全部 IDENTICAL**。
3. 7 组 PPL 数字**从归档的 `log_rank0_*.txt` 直接重取**，与 `lab_round7_result.txt` 逐位一致。
4. 三条正式腿的 `done_ppl.flag` / `done_multibit.flag` / `done.flag` 齐全；`r7_68m_mean_check` 是诊断腿，无 flag（不入 runner 账）。
5. runner 相对补丁原件只有 1 处改写（`ErrorActionPreference`），+4 / −1 行。

---

## 七、留给开发组判断的事

1. `shield_mode=zero` 在本轮两个模型上都与 `mean` 逐位等价，**是否还要把它作为 v1 基线写进论文**；若要用它做对照，需要构造低显著子集单符号的场景（例如按通道取绝对值后再判显著性），否则这个对照恒等于 mean。
2. 68M 的三个 mean 基线（156.84 / 156.09 / 157.79）该以哪个为准；本机同机同版实测为 157.79。
3. 三组种子已齐（Ours seed2/3/4 与 Original seed2/3/4 全配对），误差棒可直接计算；是否需要再补 seed=5。
4. 8B 冒烟仍未排（PIQA 在 24 GB 卡上极可能 OOM），8B 权重已就位（第六轮）。
5. ACE / 发放率脚本仍未收到，口径（逐层 ACE vs 发放率曲线、用哪些 checkpoint、哪些校准样本）待定。
6. 零样本 `acc_norm` 与官方口径的差异（`lm_eval\base.py:660` 按选项字符长度归一）仍未确认，跨表引用原文 49.51 是否成立需拍板。

---

## 八、本轮踩到的坑（避免下一轮重犯）

1. **zip 用反斜杠当路径分隔符**：`round7_patch_2026-11-02.zip` 的条目名是 `SpikeLLM\...` 形式，`unzip` 报错、`Expand-Archive` 会散架。用 `zipfile` + `name.replace('\\','/')` 解压。
2. **补丁里的 runner 把 `$ErrorActionPreference` 设回 `"Stop"`**：PS 5.1 下 python 往 stderr 打一句 `urllib3 ... doesn't match` 的 warning 就会被当成致命错误杀掉子进程（第五、六轮都是 `Continue`）。每轮覆盖补丁后**必须先查这一行**。
3. **runner 默认找 `.venv\Scripts\python.exe`**：本机无 `.venv`，会静默回退到 PATH 上的 anaconda 3.12（无 torch）。必须显式传 `-Py`。
4. **68M 与 7B 的缓存文件名在 Windows 上其实是同一个**（`llama` vs `Llama`，文件名不区分大小写）。已实测两个分词器逐 token 等价（同一句 26 tokens、id 逐一相同），故共用 `./cache` 不污染数字；换别的 68M 版本时必须重测。
5. 68M 门禁与 7B 串行在同一张卡上（20:22 门禁 → 20:37 起 7B），未出现显存残留竞争；两条 7B 腿速度约 0.55 min/epoch，与第五轮单腿持平。

---

## 九、术语与口径（沿用前几轮，勿改）

- fake = `fake_binary_simulate`（默认），real = `real_multibit_simulate`。
- wt2 = WikiText-2，c4 = C4；数字为 PPL，越低越好。
- A = 线性层低显著子集均值锚定屏蔽；B = **仅 Value/PV 路径**的屏蔽（`is_p=False` 的 Key/QK^T 分支不做屏蔽）；C = 步长加权显著性；D = 加权第二目标 + 余弦项（仅 `--aug_loss` 生效）。
- `shield_mode`：`mean`（默认）/ `zero`（v1 零填充，仅对照用）——本轮实测二者在混合符号激活上逐位等价。
- 所有数字单 seed 单次运行；seed2/3/4 三组为**独立 `--cache_dir`**（`./cache`、`./cache_seed3`、`./cache_seed4`）。
