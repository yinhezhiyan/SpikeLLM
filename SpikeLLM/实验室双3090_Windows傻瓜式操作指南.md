# 实验室双 RTX 3090 (Windows + ToDesk) 傻瓜式一键部署指南

> **无需 SSH、无需申请 Meta 官方 Token（已内置免审核官方同源权重镜像与国内 `hf-mirror.com` 自动加速）**

---

## 第一步：通过 ToDesk 传代码到实验室电脑
1. 在 ToDesk 顶部菜单点击 **【文件传输】**。
2. 将本机的 `d:\spikellm\SpikeLLM` 文件夹（体积很小，代码核心文件仅几 MB；**不需要传本机 `models/` 或 `cache/` 大文件**）复制到实验室电脑的纯英文路径下，例如：
   - `D:\spikellm\SpikeLLM`
3. *(推荐偷懒方式)*：看截图您实验室桌面上装有 **WorkBuddy / VS Code**！您也可以直接在实验室电脑上用 WorkBuddy 打开 `D:\spikellm\SpikeLLM`，然后对它说：
   > **“请阅读 `实验室双3090_Windows傻瓜式操作指南.md`，依次帮我运行 `-Mode check`、`-Mode download_7b` 和 `-Mode smoke_7b`”**

---

## 第二步：四条命令依次执行（在 PowerShell 中复制粘贴）

在实验室电脑上打开 **PowerShell**，进入 `D:\spikellm\SpikeLLM`：

```powershell
cd D:\spikellm\SpikeLLM
```

*(注：如果实验室电脑的虚拟环境不在默认位置，只需在下面命令末尾加上 `-PythonExe "你的环境路径\python.exe"` 即可)*

### 1️⃣ 环境体检（10 秒）
检查两张 RTX 3090 (24GB) 是否识别、依赖包是否齐全、硬盘空间与网络镜像是否正常：
```powershell
powershell -ExecutionPolicy Bypass -File .\run_lab_dual3090_7b_8b.ps1 -Mode check
```

### 2️⃣ 自动下载 LLaMA-2-7B 与校准集（支持断点续传 + 自动国内镜像加速）
默认从免 Token 的官方同源仓库 `NousResearch/Llama-2-7b-hf` 下载（无需排队申请 Meta 权限），并自动预缓存 WikiText-2 和 C4 数据集：
```powershell
powershell -ExecutionPolicy Bypass -File .\run_lab_dual3090_7b_8b.ps1 -Mode download_7b
```

### 3️⃣ 跑 LLaMA-2-7B 冒烟门禁（Smoke Gate，`nsamples=8, epochs=1`）
验证全模型 32 层量化校准 + 评测端到端不报错、不爆显存：
```powershell
powershell -ExecutionPolicy Bypass -File .\run_lab_dual3090_7b_8b.ps1 -Mode smoke_7b
```
- 跑完看到绿字 `SMOKE GATE PASSED for Llama-2-7b-hf!` 后，即可进入第 4 步正式实验！

### 4️⃣ 启动双 3090 满血并行正式实验（LLaMA-2-7B W4A4）
**0 号 3090** 自动跑 `Original SpikeLLM (+LET)`，**1 号 3090** 同时并行跑 `Ours: OSDSQ v2 (+LET)`，并自动串联：
- Phase A：`fake_binary_simulate` 20 epochs 校准 + WikiText-2 / C4 PPL 评测（生成 `done_ppl.flag`）
- Phase B：`real_multibit_simulate` 真实多步脉冲仿真 PPL 评测（生成 `done_multibit.flag`）
- Phase C：6 个 Zero-Shot 常识推理任务评测（生成 `done.flag`）

```powershell
powershell -ExecutionPolicy Bypass -File .\run_lab_dual3090_7b_8b.ps1 -Mode 7b_parallel
```

---

## 第三步：（可选进阶）LLaMA-3-8B 双卡并行验证
当 LLaMA-2-7B 跑完后，如需继续冲击 **LLaMA-3-8B**（32 层 GQA 架构），同样只需三条命令：
```powershell
powershell -ExecutionPolicy Bypass -File .\run_lab_dual3090_7b_8b.ps1 -Mode download_8b
powershell -ExecutionPolicy Bypass -File .\run_lab_dual3090_7b_8b.ps1 -Mode smoke_8b
powershell -ExecutionPolicy Bypass -File .\run_lab_dual3090_7b_8b.ps1 -Mode 8b_parallel
```
