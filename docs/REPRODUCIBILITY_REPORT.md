# REPRODUCIBILITY_REPORT — 实验复现状态、数值差异与运行证据

> 归档验证环境：Python 3.11.15，torch 2.13.0+cu126（单卡 NVIDIA GPU），
> numpy 2.4.4，pandas 3.0.2，scipy 1.17.1。所有"重跑"指：**发布版代码 +
> 归档冻结权重 +（仅链接到原项目的）DKASC 数据**，在
> `/tmp/DMU-NACI-verify` 沙箱中执行；发布树本身不含数据。
> 日期：2026-10-27。状态标记：PASS / FAIL / NOT RUN / MISSING。

## 0. 结论速览

| 范围 | 状态 |
|---|---|
| 论文 Table 2/3/4/5 全部单元格（来自归档证据CSV） | **PASS**（177/177 项断言） |
| s7 全量重跑（5 seeds×8池×A0–A3，Table 3/4 来源） | **PASS（逐位一致，max\|Δ\|=0）** |
| Site 1B p8+9 全量重跑（Table 5 Site 1B 行） | **PASS（逐位一致，max\|Δ\|=0）** |
| s9 CQR 复现门控 + CQR-ACI（Table 5 Site 18 CQR 行） | **PASS（门控 1.1e-16；逐位一致）** |
| 单种子 s7 重放（seed 42，8个方法×2 span） | **PASS（max\|Δ\|=1.9e-8）** |
| s2 训练路径（数据/划分/训练器冒烟） | **PASS**（54,877/6,331 与 Table 1 一致；3-epoch ES RMSE 0.287→冻结全程最优 0.274） |
| 论文图再生成（最终稿编号） | **PASS：Fig.3（输入表征）/4（残差池）/6（困难状态）/7（跨系统）与论文 PNG 逐字节相同（MD5 相同）；Fig.1/2 为绘图工具资产；Fig.5/S1 输入缺失见 MISSING_FILES** |
| 单元/权重/配置/烟雾测试 | **PASS（125 项）** |
| s7 内配对日块 bootstrap | **PASS（逐位一致，80/80 行 max\|Δ\|=0）** |
| s10 两层 bootstrap 重跑 | **NOT RUN**（证据与运行日志已归档） |
| 完整 s2–s5 训练重跑（100 epoch×5 seeds×多模型） | **NOT RUN**（GPU-小时级；冻结权重即论文产出权重） |
| Fig.5（消融）/Fig.S1（代表日）再生成、Table S1/S2 与 Fig.3 诊断生成器 | **MISSING 输入 / legacy**（详见 MISSING_FILES.md §3–4） |
| 2024/2025 预开宇宙从原始数据重建 | **MISSING**（legacy 依赖闭合未随库分发） |

\* 见 §3.1 的 bootstrap 完成记录。

## 1. 静态与单元验证（无需数据）

执行命令与结果（发布树内运行）：

```
$ python -m pytest tests/ -q
125 passed in 51.14s
```

覆盖：共形有限样本分位数规则；(n+1) 规则边界；ACI 三步成熟、FIFO 顺序、
α 方向（miss→α↓→区间加宽）与 clip[0.001,0.5]；IS 定义；RCE/Streak；
γ 选择规则；TCN 参数量 36,705（论文 Table S1）与 GRU 3,777（论文 §2.2）；
CQR 头形状；configs/*.yaml 与代码常量一致；**全部 135 个 .pt（Site 18 100 + Site 1B 35）加载进各自
架构**（含 site18 40 + site1b 35 个逐文件断言、FREEZE_MANIFEST SHA-256
对账、scaler 指纹与 checkpoint 内记录一致）；两处移植的原始单元测试
（因果窗口 40 项、历史熟悉度 16 项）；合成数据端到端冒烟
（清洗→窗口→TCN→残差→GRU→共形→延迟ACI，48 s）。

## 2. 论文数值 ← 归档证据核对（静态，无重计算）

```
$ python scripts/verify_paper_numbers.py
… [OK] ×177
All manuscript values reproduced from the archived evidence CSVs.
```

- Table 3：7个残差池 × 8列（Std(z)/Spearman/PICP/PINAW/IS/IS-D/PICP-D/PICP-H）；
- Table 4：A0–A3 × 10列（含 streak、HIGH_N、HARD_DF、Daylight、Width-H、IS-H）；
- Table 5：两站 × 6方法（PICP/PINAW/IS/NIS/RCE1D/RCE3D/streak）；
- Table 2：两站 RMSE/MAE/nRMSE/nMAE（Site 18 来自冻结数值字典的
  verified-replay 记录，Site 1B 来自 eval CSV）。
- 差异：**0**（全部在 3 位小数舍入容差 5e-4 / 2 位小数容差 5e-3 内，
  172 项完全相等）。

## 3. 真实数据重执行（动态验证）

沙箱构建：发布树副本 + `data/{raw,processed}`、`outputs/predictions`
符号链接至原项目数据区 + 权重放回管线期望位置（符号链接）。

### 3.1 Site 18 主线

| 命令 | 结果 |
|---|---|
| `python scripts/replay_s7_seed.py --seed 42` | PASS：8 方法×2 span，53 列指标 max\|Δ\|=1.9e-8（GPU 浮点噪声级） |
| `python experiments/rigorous_v2/s7_final_eval.py`（全量5 seeds） | **PASS：metrics_per_seed.csv 140行×53浮点列 max\|Δ\|=0（逐位一致）**；run_manifest：n_calib=4,601、n_same=25,668、n_cross=98,239 —— 与论文 Table 1 一致；winner=T3p 与冻结清单一致 |
| 同上，配对日块 bootstrap 段 | **PASS：bootstrap_paired.csv 80/80 行（4对×5种子×4指标）max\|Δ\|=0（逐位一致，每 seed 固定 RNG）** |
| `python experiments/cqr_aci/s9_cqr_aci_site18.py` | PASS：门控1 "fixed-CQR reproduction max abs diff: 1.11e-16"；CQR-ACI per-seed 与归档证据 max\|Δ\|=0；cross 中位数 0.900/0.093/0.726/0.140/0.034/0.020 与论文 Table 5 CQR+ACI 行一致；2000次两层bootstrap复执行完成（20/20 行） |

### 3.2 Site 1B

| 命令 | 结果 |
|---|---|
| `python experiments/site1b_validation/p8_eval.py`（全量5 seeds×5方法×2 span） | **PASS：site1b_eval_per_seed.csv 50行×12浮点列 max\|Δ\|=0（逐位一致）**；cross 中位数与论文 Table 5 Site 1B 六行全部一致（A0 0.950/4.983、A1 0.918/3.097、A2 0.855/2.574、A3 0.898/2.530、CQR 0.903/2.683、CQR-ACI 0.900/2.607）；逐 seed 显著性结论与 PHASE10_REPORT 一致（如 A3-vs-CQR RCE1D 5/5） |

### 3.3 训练路径（s2 冒烟）

真实 2024 DEV 数据上以 `common_v2.train_tcn(seed=42, max_epochs=3)` 触发
完整数据链：54,877 梯度行 / 6,331 早停行（**与论文 Table 1 完全一致**），
scaler 拟合、AdamW/512/58003 训练循环、kW 域早停——3 epoch 内 ES RMSE
0.2997→0.2871，收敛方向与冻结 checkpoint 的全程最优 0.2744（epoch 15）一致。
完整 100-epoch×5-seed 重训未执行（NOT RUN；属 GPU-小时级），归档权重
（SHA-256 已登记于 FREEZE_MANIFEST）即论文产出权重。

### 3.4 图表再生成（发布树内）

| 图 | 命令 | 与论文 PNG 对比 |
|---|---|---|
| Fig.3（输入表征；资产名 figS3_representation.png） | `python figures/latex_v4_repair/fig5_representation_repair.py` | **MD5 相同**（3830038d…） |
| Fig.4（残差池；资产名 fig3_provenance.png） | `python figures/latex_v4_repair/fig6_provenance_pools_repair.py` | **MD5 相同**（f80c83ad…） |
| Fig.6（困难状态；资产名 fig5_harddf.png） | `python figures/latex_v5_repair/fig5_harddf_conditional_v5.py` | **MD5 相同**（eda45479…） |
| Fig.7（跨系统；资产名 fig6_crosssystem.png） | `python figures/latex_v5_repair/fig6_crosssystem_v5.py` | **MD5 相同**（5357f8ae…） |
| Fig.1/Fig.2 | 绘图工具资产（jpg），非代码生成 | 稿件资产 |
| Fig.5（消融）/Fig.S1（代表日） | `fig8_ablation_dynamic_repair.py` / `fig7_interval_examples_repair.py` | 脚本在库；输入为逐样本文件（数据政策未分发）→ MISSING 输入 |
| （终稿未引用）TCN示意/滚动覆盖/数据概览 | `fig2_3_schematics_repair.py` / `figS5_rolling_v5.py` / `fig1_data_overview_repair.py` | 辅助资产生成器（TCN示意已验证与早期草稿资产逐字节一致 ce33bd85…，终稿未引用） |

## 4. 未核实 / 部分核实内容（如实声明）

1. **跨 GPU 型号的训练位级可重复性**未验证。评估路径（推理+校准+ACI+指标）
   在本机为逐位一致；训练路径仅做了 3-epoch 冒烟与收敛方向核对。
2. **s10 两层 bootstrap** 未重跑（NOT RUN）。其输入为逐样本区间文件
   （数据政策未分发）；归档证据 `results/rigorous_v2/s10_two_level_bootstrap/
   two_level_bootstrap.csv` 与运行日志 `s10_run.log` 已随库，其中记录了
   对冻结指标的 1e-9/1e-6 回放断言。s9 内的对应两层 bootstrap（同算法）
   已在 §3.1 重执行并逐位一致，可作为该 bootstrap 实现正确性的直接证据。
3. **s1（2021/2022 从原始CSV构建）**未在沙箱重跑（需要 ~400 MB 原始数据
   下载）；其窗口机制已由合成冒烟测试与 s2 数据加载间接覆盖。归档的
   `data/2021_windows_audit.json`、`2022_windows_audit.json` 为原始运行记录。
4. **s8（报告汇编器）** 未重跑（其全部输入表格已在 §3 重生成并可对比）。
5. **2024/2025 预开宇宙重建**：生成器脚本已归档但依赖 legacy 闭合
   （MISSING_FILES.md §3）；本验证直接链接了原项目的冻结产物。
6. 论文 Section 3.2 的阈值常数（N_Q75=0.5554、D_Q75=0.4873）的拟合过程
   位于 legacy 工具（16/20号脚本），本发布只包含按阈值打标的消费端。

## 5. 运行环境与命令完整记录

- 沙箱：`/tmp/DMU-NACI-verify`（发布树副本；数据经符号链接）。
- GPU：单卡（CUDA 12.6 runtime）；三次全量重跑（s7/s9/p8）时长分别约
  50/12/12 分钟。
- 本报告引用的全部命令均为发布树内的真实路径；日志：
  `/tmp/s7_full_rerun.log`、`/tmp/s9_rerun.log`、`/tmp/p8_rerun.log`
  （归档环境执行时产生，不随库分发；关键结论已摘录于本报告）。
