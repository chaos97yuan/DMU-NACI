# PAPER_CODE_MAPPING — 论文章节 ↔ 真实代码对应表

> Paper: *Provenance-Controlled Residual Learning with Adaptive Conformal
> Calibration for Cross-Year PV Prediction Intervals* (Shi, Jin, Zhang, Xiao;
> manuscript `latex_v8`, 2026-10-26). All paths are relative to this repository.
> "原项目路径" refers to the original research project `PV_BSCN_Project`;
> unless a change is flagged in `CHANGELOG.md`, the file here is a verbatim copy.

## 1. 方法章节

| 论文位置 | 内容 | 实现代码 | 产出证据 |
|---|---|---|---|
| §2.1 (Multi-scale TCN) | 72×5输入、5/15/60 min三分支、32通道、64维融合、线性头 | `src/models/multiscale_tcn.py`、`src/models/tcn_blocks.py`；训练 `experiments/rigorous_v2/common_v2.py::train_tcn` | Table 2（RMSE 0.286/0.867）、Table S1（36,705参数） |
| §2.2 (误差尺度GRU) | GRU(5→32)+Linear(32→1)+softplus、高斯NLL | `src/models/uncertainty_encoders.py::ScaleModel`；训练 `common_v2.py::train_gru`（3×50重启日程） | Table S2(b)（3,777参数）、Table 3 的 Std(z)/Spearman 列 |
| §2.3 (残差来源控制) | 溯源元组、E24_OOF、E22/E23_replay、V1/V2/V3、T1in、T3match | `common_v2.py`（OOF_FOLDS/VINTAGE/ALLOWED_PROVENANCE/assert族）、`s3_e24_oof.py`、`s4_replay.py`、`s5_vintage.py`；池组装与预注册选择 `s6_gru_select.py` | `results/rigorous_v2/s*/residual_manifest.json`、`s6_gru_select/selection_decision.json`（T3p胜出）、`s6_gru_select/pool_manifest.csv`（池样本数 28,418/127,789/230,359…） |
| §2.3 (T3-match对照) | 按月×昼光分层抽样28,418 | `s6_gru_select.py::build_pools`（T3match 分层子采样） | Table 3 第4行 |
| §2.4.1 (归一化共形) | s=|e|/(σ+1e-6)、(n+1)规则分位数 | `common_v2.py::conformal_q`；s7 内 `cal_norm_sorted` | 单元测试 `tests/test_calibration_semantics.py` |
| §2.4.2 (三步延迟ACI) | FIFO队列、成熟反馈、clip[1e-3,0.5]、γ按seed在SELECT选择 | `common_v2.py::run_aci`/`select_gamma`（ACI_DELAY_MIN=15） | `s6_gru_select/gamma_selection.json`（γ_A3=0.001×4, γ_A1∈{0.02,0.05}） |
| §2.5 + Algorithm 1 | 完整DMU-NACI离线+在线流程 | `s2→s3→s4→s6→s7` 整体；在线部分 `run_aci` | `results/rigorous_v2/FREEZE_MANIFEST.json`（冻结时间2026-10-01T20:42） |

## 2. 数据与实验设置（第三章）

| 论文位置 | 内容 | 实现代码 | 产出证据 |
|---|---|---|---|
| §3.1 (数据读取/清洗) | DKASC原始CSV→清洗→5 min全网格 | `src/data/schema.py`（列名映射）、`scripts/02_clean_data.py`（冻结清洗函数）、`common_v2.py::build_year_core` | `results/rigorous_v2/data/*_windows_audit.json` |
| §3.1 (防泄漏窗口) | 72步因果窗、目标不插值、未来插值端点依赖剔除 | `src/eval/historical_interpolation_causality_audit.py`、`common_v2.py::build_year_windows`；2024/2025冻结宇宙由 `legacy/preopen_builders/37r{h,j}_*.py` 构建 | 窗口审计JSON（候选/保留数） |
| Table 1 (划分) | DEV 54,877 / ES 6,331 / SELECT 11,138 / CALIB 4,601 / SAME 25,668 / CROSS 98,239(18B: 87,839) | `common_v2.py` 角色边界常量；`s2_train_final_mean.py`/`s7_final_eval.py`；Site 1B `site1b_validation/common_1b.py` | `s2_final_mean/split_manifest.json`（54,877/6,331 已在归档环境重新验证）；`s7_final_eval/run_manifest.json`（n_calib=4,601, n_same=25,669, n_cross=98,239）；Site 1B `site1b_val/phase2_clean_manifest.json` |
| §3.2 (HIGH_N/HARD_DF) | 12维上下文、2021记忆、N_t/D_t、阈值0.5554/0.4873 | `src/features/historical_familiarity.py`、`src/features/dynamic_index.py`；标签写入 preopen context（37rh/37rj，legacy） | `results/rigorous_v2/s7_final_eval/conditional_correct_per_seed.csv`（论文口径）；阈值记录于原项目配置 |
| §3.3 (A0–A3消融) | 2×2：绝对/归一化 × 固定/延迟ACI | `s7_final_eval.py`（A0/A1/A3由winner+CALIB分数生成）；Site 1B `p8_eval.py` | Table 4、Table 5 |
| §3.3 (CQR/CQR+ACI) | TCN主干+pinball(0.05/0.5/0.95)+CALIB共形；延迟ACI同A3 | Site 18：`experiments/site1b_validation/p8b_site18_cqr.py`（训练）+ `experiments/cqr_aci/s9_cqr_aci_site18.py`（复现门控+CQR-ACI）；Site 1B：`common_1b.py::train_tcn_cqr`+`cqr_aci/s9_cqr_aci_site1b.py` | `s7_final_eval/site18_cqr_per_seed.csv`、`s9_cqr_aci/`、`site1b_val/cqr_aci/` |
| §3.4 (指标) | RMSE/MAE/PICP/PINAW/IS/NIS/RCE1D/RCE3D/Streak/Std(z)/Spearman/昼光GHI>20 | `common_v2.py::interval_metrics/scale_metrics/interval_score`、`common_1b.py::interval_metrics_1b` | 单元测试覆盖 IS/RCE/Streak 语义 |

## 3. 结果（第四章）与统计

> 图号按**最终稿**（`史超远初稿_v3.pdf`，正文7图）标注；图片资产文件名为
> 早期草稿遗留标签，对应关系以 `figures/README.md` 为准。

| 论文位置 | 内容 | 证据文件（results/） | 复现命令 |
|---|---|---|---|
| §4.1 Table 2 | 点预测 0.286/0.867 kW | `supplementary/FINAL_PAPER_NUMBER_DICTIONARY_V4.csv`（S18，verified replay）、`site1b_val/site1b_eval_per_seed.csv`（S1B A0行） | `s7`/`p8`（重推理）；`scripts/verify_paper_numbers.py` |
| §4.1 Fig.3 + Table S2(a) | 输入表征对比（R0/R1/R2） | `stage8_final_architecture/05_stage1_confirmatory/metrics_per_seed.csv` | 图 `figures/latex_v4_repair/fig5_representation_repair.py`（已验证与论文PNG逐字节一致）；诊断生成器为legacy |
| §4.2 Table 3 + Fig.4 | 残差池对比 | `rigorous_v2/s7_final_eval/metrics_per_seed.csv`（T1p/T2p/T3p/T3match/T1in/V1/V3）、`monthly_cross_year.csv` | `s7`；图 `figures/latex_v4_repair/fig6_provenance_pools_repair.py`（逐字节一致） |
| §4.3 Table 4 + Fig.5 | 校准消融 | `metrics_per_seed.csv`（T3p:A0/A1/A3行）+ `figures/figure_source_data/Fig9_HARD_DF_tradeoff_source.csv`（Width-H/IS-H） | `s7`；图需逐样本parquet（见MISSING_FILES） |
| §4.4 Fig.6 | HARD_DF可靠性 | `figure_source_data/Fig9_*.csv`、`s7_final_eval/conditional_correct_per_seed.csv` | `figures/latex_v5_repair/fig5_harddf_conditional_v5.py`（逐字节一致） |
| §4.5 Table 5 + Fig.7 | 跨系统A0–A3+CQR | `rigorous_v2/s9_cqr_aci/cqr_aci_per_seed.csv`、`site1b_val/cqr_aci/cqr_aci_per_seed.csv`、`site1b_val/site1b_eval_per_seed.csv`、`s7_final_eval/site18_cqr_per_seed.csv` | `s9_*`/`p8`；图 `figures/latex_v5_repair/fig6_crosssystem_v5.py`（逐字节一致） |
| §3.4 + §4.2 配对CI | 2000次两层bootstrap（种子×日块） | `rigorous_v2/s7_final_eval/bootstrap_paired.csv`（逐seed日块）、`s10_two_level_bootstrap/two_level_bootstrap.csv`（论文引用的两层CI）、`s9_cqr_aci/cqr_aci_bootstrap.csv`、`site1b_val/site1b_bootstrap.csv` | `s10_two_level_bootstrap.py`（RNG 2026；逐样本区间冻结回放断言1e-9/1e-6） |

## 4. 补充材料（最终稿：Fig. S1 + Tables S1/S2）

| 论文位置 | 内容 | 证据/代码 | 状态 |
|---|---|---|---|
| Fig. S1 | 代表日预测区间（clear/variable/HARD_DF × A0/A3/CQR） | `figures/latex_v4_repair/fig7_interval_examples_repair.py` + `figure_source_data/Fig7_case_days.csv` | 需逐样本区间parquet（数据政策未分发） |
| Table S1 | 点预测骨干对比 | 证据 `results/supplementary/table2_mean_backbone.csv`；生成器原项目 `experiments/stage6_backbone_audit.py`（legacy） | 证据已归档；生成器未随库分发 |
| Table S2 | (a)表征诊断 + (b)uncertainty encoder对比 | 证据 `results/stage8_final_architecture/05_stage1_confirmatory/metrics_per_seed.csv`、`06_stage2_confirmatory/metrics_per_seed.csv`、`results/supplementary/table3_uncertainty_encoder.csv`；生成器原项目 `experiments/diagnose_hetero_representation.py`、`experiments/diagnose_uncertainty_encoders.py`（legacy） | 证据已归档（Fig.3亦由此生成且逐字节一致）；再生成需legacy |
| （终稿未引用的早期草稿图） | 数据概览/TCN示意/滚动覆盖 | `figures/latex_v4_repair/fig1_data_overview_repair.py`、`fig2_3_schematics_repair.py`、`latex_v5_repair/figS5_rolling_v5.py` | 辅助资产生成器，最终PDF未引用 |

## 5. 关键发现（论文口径 vs 代码实现）

1. **HIGH_N 的两种口径**：`common_v2.state_split` 中 `HIGH_N = ¬HARD_DF ∧ 非NORMAL`
   （诊断用互斥口径）；论文 Table 4 的 HIGH_N 列（如 A0=0.740）来自
   `s7_final_eval/conditional_correct_per_seed.csv`，其口径为论文式(14)的
   **包含HARD_DF** 的 HIGH_N。两者数值不同但各自内部一致；论文使用后者。
   HARD_DF 与 Daylight 列两种口径一致。
2. **s2 的 ES 修正**：`common_v2.train_tcn` 的早停RMSE在kW域计算（原Stage-6辅助
   函数曾比较标准化预测与kW目标）；代码docstring明确记录了该修正，且v2管线所有
   模型均重新训练，冻结checkpoint即本代码产物。
3. **2024/2025窗口宇宙**由legacy预开构建器生成（`legacy/preopen_builders/`），
   其`mu`列在DMU-NACI管线中仅用于行对齐、不参与任何计算（`preopen.py` docstring）。
4. **p8_eval.py 的 CQR checkpoint 加载死代码缺陷**已在发布版修复（原项目该分支
   从未执行，不影响任何论文数值）——详见 `CHANGELOG.md`。
