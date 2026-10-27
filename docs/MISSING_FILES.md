# MISSING_FILES — 缺失模块、权重与未完成实验清单

本文件如实列出**不随本发布分发**或**无法在本发布内独立运行**的内容。
凡列为缺失者，均未以任何替代实现冒充；相应实验状态标记遵循
`REPRODUCIBILITY_REPORT.md` 的 PASS / FAIL / NOT RUN / MISSING 约定。

## 1. 数据（有意排除；政策见 `data/README.md`）

| 缺失内容 | 原始位置 | 影响 | 恢复途径 |
|---|---|---|---|
| DKASC 原始 CSV 2017–2025（约1.8 GB） | `data/raw/Alice_Springs_{year}.csv` | s1（2021/2022窗口）、s4/s5（回放/vintage输入）、p2（Site 1B）无法运行 | 官方下载 https://dkasolarcentre.com.au ，文件名见 `data/README.md` |
| 2024/2025 预开宇宙（窗口+状态标签） | `data/processed/external_{2024,2025}_preopen/`（约463 MB）与 `outputs/predictions/stage37r_{2024,2025}_preopen/` | **s2–s7、s9、s10 无法运行**（它们是2024 DEV/SELECT/CALIB/SAME与2025 CROSS的唯一窗口来源） | 用 `legacy/preopen_builders/37rh/37rj` 重建（见§3 legacy 依赖） |
| Site 1B 清洗数据 | `data/processed/site1b_5var_5min.csv`（38 MB） | p34–p9 无法运行 | `p2_clean.py` 从原始CSV重建（自包含） |
| 2023 外部窗口包 | `data/processed/external_2023/`（187 MB） | s4（E23回放）需要 | 原项目生成器（legacy 链）；或按 s1 的 2021/2022 机制重建（代码在 `common_v2.build_year_windows`，但2023当年的已归档产物由 legacy 工具生成） |
| 逐样本残差/区间文件 | `results/rigorous_v2/s{3,4,5}/*.parquet`、`s7_final_eval/{mu_spans.npz, per_sample_*.parquet}`、`s9_cqr_aci/cqr_aci_intervals_seed*.csv`、Site 1B `E2*_site1b.csv`、`cqr_aci/cqr_aci_intervals_*.csv` | 图 Fig.5/Fig.S1 与 s10 bootstrap 的直接输入缺失 | 各自管线重跑即再生成（s7→s0→Fig.5；s9→滚动覆盖辅助图；p34/p567→site1b池） |

随库保留的**小型模型派生文件**（供重放验证）：Site 1B `site1b_mean_cache.npz`
（μ缓存，1.7 MB）、两站 CQR CALIB 分数 CSV（每seed约90 KB）。

## 2. 模型权重（无缺失）

`checkpoints/` 已包含论文全部冻结权重（135个 .pt = Site 18 100个 + Site 1B 35个，全部通过加载与 SHA-256
对账，见 REPRODUCIBILITY_REPORT §1）：

- Site 18：final_mean ×5、OOF折 4×5、vintage 3×5、GRU 8变体×5(+equpd 3×5)、CQR ×5；
- Site 1B：mean ×5、GRU_T3p ×5、OOF折 4×5、CQR ×5；
- 两站 scaler（site18_scaler.json / site1b_mean_scaler.json）。

**无任何伪造权重**。缺失清单此节为空。

## 3. legacy 依赖（引用但未随库分发；权属原项目）

`legacy/preopen_builders/37rh_2024_preopen.py`、`37rj_2025_preopen.py` 为
2024/2025 预开宇宙的**真实生成器**，逐字保留但**在本发布内不可运行**。
缺失的依赖闭合：

| 类别 | 缺失文件（原项目路径） |
|---|---|
| legacy 模型/工具 | `src/models/bscn.py`（BSCN及其依赖）、`src/eval/compact_final_reliability.py`、`src/features/scaler.py` 使用方、`experiments/diagnose_hetero_representation.py`（Stage-D1，1005行）、`experiments/diagnose_uncertainty_encoders.py`、`experiments/diagnose_multiyear_uncertainty.py`、`experiments/stage8_common.py`、`experiments/pre_submission_correction.py`、`experiments/stage6_backbone_audit.py`、`experiments/stage8_step13_tables.py`、`scripts/37rq3_point_baseline.py` 及 dlinear/patchtst/timemixer 模型 |
| 冻结协议/锁文件 | `stage37r_h/i` 协议 JSON、`POSTEXTERNAL_EXTENSION_MASTER_LOCK.json`、`stage37r_g_decision.json`、`stage37r_official_calibration_quantiles.json`、`dynamic_index_config.json`、`historical_familiarity_thresholds.json`、`train_derived_thresholds.json`、`FINAL_2023_EXTERNAL_LOCK.json` |
| 冻结权重/记忆 | `outputs/checkpoints/causal_clean_2022/multiscale_tcn_causal_clean_best_cuda.pt`、`input_scaler_causal_clean.json`、`outputs/checkpoints/bscn_baseline_config.json`、`data/processed/history/site18_2021_memory.csv` |

影响与途径：预开宇宙重建、Table S1/S2 与 Fig.3 表征诊断的**从原始数据再生成**
需取得上述 legacy 闭合（原项目保留完整副本）。DMU-NACI 方法管线本身
（s2–s10、Site 1B、CQR、A0–A3）**不依赖**这些 legacy 模块——其依赖仅为
§1 的数据条目。

## 4. 实验/图表状态汇总（详见 REPRODUCIBILITY_REPORT.md）

| 项 | 状态 | 说明 |
|---|---|---|
| s7 全量5-seed重跑（Table 3/4全部变体） | 见 REPRODUCIBILITY_REPORT（后台执行） | 需 §1 的 2024/2025 预开宇宙（沙箱内已链接） |
| Site 1B p8+9 全量重跑（Table 5 Site 1B） | **PASS（逐位一致）** | 修复 p8 CQR 加载缺陷后 |
| s9 CQR门控+CQR-ACI 重跑（Table 5 Site 18 CQR行） | **PASS（门控1e-16；逐位一致）** | |
| s2 训练路径（数据/划分/训练器） | **PASS（3-epoch 冒烟，54,877/6,331 与 Table 1 一致）** | 完整100-epoch重训为 GPU-小时级，未在本归档执行 |
| s10 两层bootstrap重跑 | **NOT RUN**（证据已归档：`s10_two_level_bootstrap/two_level_bootstrap.csv` 与运行日志 `s10_run.log`） | 依赖逐样本区间文件；重跑需先重跑 s7/s9 再执行 |
| Fig.5（消融）与 Fig.S1（代表日区间） | **MISSING 输入**（逐样本文件未分发） | 脚本完整；重跑管线后可再生成。Fig.1/Fig.2 为绘图工具资产；早期草稿的数据概览/TCN示意/滚动覆盖图未纳入终稿 |
| Table S1/S2 与 Fig.3 的**诊断生成器** | legacy（§3） | 证据 CSV 已归档；Fig.3 已验证逐字节一致 |
| 论文 Fig.1（时间线）/Fig.2（框架图） | 绘图工具资产（`figs/*.jpg`），非代码生成 | 属稿件资产，不在代码归档范围 |

## 5. 其他未核实事项

- 训练在**不同 GPU 型号**上的位级可重复性未验证（同型号 CUDA 环境下
  评估路径已验证逐位一致；训练为固定种子但未做跨设备对照）。
- `results/rigorous_v2/s10_run.log` 为归档运行日志，本次未重新执行该步。
- 原项目 `results/` 其余大量 legacy 诊断目录（stage6/7/8/9 系列、
  final_paper_package 等）与论文最终数值链无关，未归档（清单见原项目
  `results/stage9_6_provenance_freeze/01_SUPERSEDED_RESULT_MAP.csv`）。
