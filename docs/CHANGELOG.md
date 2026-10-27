# CHANGELOG — 原始项目 → DMU-NACI 开源归档

本文件记录从原始科研项目 `PV_BSCN_Project` 到本发布仓库的**每一处代码修改**。
除下列条目外，所有代码文件均为逐字复制。核心算法逻辑（模型结构、损失、训练
策略、实验参数、评估协议）**零改动**。验证方式见 `REPRODUCIBILITY_REPORT.md`。

## 0. 版本 1.0.2（2026-10-27，归档复审修订）

1. **`torch.load` 安全加固（已回归验证的代码修改）**：发布版全部 31 处
   `torch.load(..., weights_only=False)` 改为 `weights_only=True`（公共分发的
   最小反序列化面）。切换前对全部 135 个冻结权重做了双路加载对比：
   `weights_only=True/False` 张量级完全一致（135/135）；切换后回归：
   `tests/test_checkpoint_loading.py` 71 项全过，沙箱内
   `scripts/replay_s7_seed.py --seed 42` 对真实数据重放与归档指标偏差仍为
   1e-8 量级（GPU 噪声级），结论不变。
2. **元数据一致性更正**：CHANGELOG v1.0.1 中"300 个发布文件"更正为
   "302 个"；CHANGELOG §4 与 `CODE_AUDIT_REPORT.md` §3.3 关于 CITATION
   许可证的表述同步为"已移除 license 键（1.0.0 曾为 NOASSERTION），待权利人
   选定后设为 SPDX MIT/Apache-2.0"。
3. **图表复现口径统一**：`CODE_AUDIT_REPORT.md` 权威口径改为 4 张正文图
   （Fig.3/4/6/7，本归档环境独立重执行、与论文 PNG 逐字节一致）；TCN 示意
   （终稿未引用）作为辅助资产另行注明。
4. **CITATION.cff**：移除 `date-released` 可选字段（内部构建日期保留于本
   CHANGELOG；待公开 Release 时填写真实发布日期）；作者列表与论文题页的
   5 位作者及顺序对齐（Shi, Xiao, Jin, Zhang, Wang）；`LICENSE` 与
   `docs/license_candidates/MIT.txt` 的版权行同步为同一作者列表。
5. **README 措辞**：区间公式更正为 `C = [μ − q_tσ, μ + q_tσ]`；"the exact
   pipeline that produced every number" 改为 "the principal experimental
   pipeline and the archived evidence supporting the reported results"，与
   首页"非完整端到端可复现"的边界声明一致。
6. **data/README.md**：精确区分清洗层（含功率在内的短缺口有界、逐格标记的
   双侧线性插值）与窗口保留规则（目标必须直接观测、历史无缺失核心格、
   禁止未来端点依赖插值），并补充 Site 1B 完全不插值的差异；此为论文
   "Missing values are not interpolated..." 一句的代码级语义。
7. **打包方式**：`DMU-NACI.zip` 改由 `git archive`（跟踪文件清单）生成，
   保证不含 .git 对象、pytest/Matplotlib 缓存或任何未核对文件。

## 0.1. 版本 1.0.1（2026-10-27，归档评审修订）

本轮仅修订文档/打包/仓库治理，**无代码与数值变化**：

1. **发布状态明示**（README 首页横幅）：明确本仓库为
   "pipeline and frozen-artifact release — not fully end-to-end
   reproducible from raw"，2024/2025 冻结窗口宇宙及其 legacy 依赖闭包不在
   分发范围内（详见 `MISSING_FILES.md`）。
2. **图表编号按最终稿重映射**：正文7图编号以 `史超远初稿_v3.pdf`（latex_v8）
   出现顺序为准——Fig.3=输入表征、Fig.4=残差池、Fig.5=消融、Fig.6=困难状态、
   Fig.7=跨系统、补充材料 Fig.S1=代表日区间、Tables S1/S2。图片**资产文件名**
   （figS3_representation.png 等）为早期草稿遗留标签；`figures/README.md`
   建立"终稿图号↔资产文件名↔生成脚本"权威映射，README §7、
   `PAPER_CODE_MAPPING.md`、`REPRODUCIBILITY_REPORT.md`、`MISSING_FILES.md`、
   `configs/experiments.yaml` 同步更正。终稿未引用的早期草稿图
   （数据概览/TCN示意/滚动覆盖）标注为辅助资产。
3. **权重数量更正**：冻结权重为 **135 个 .pt**（Site 18 100 个：5 mean +
   20 OOF + 15 vintage + 55 GRU + 5 CQR；Site 1B 35 个：5 mean + 5 GRU +
   20 OOF + 5 CQR）。首轮文档误写"136"，经与原项目产物目录及冻结清单交叉
   核对确认 135 为完整集合、无遗漏，全部文档已更正。
4. **`.gitignore` 细粒度化**：`results/rigorous_v2/data/` 整目录排除改为
   "目录内排除 + `!*_audit.json` 白名单"，避免吞掉已发布的审计 JSON
   （2021/2022_windows_audit.json、s1_data_audit.json）。
   已 `git init` 并以 `git status --ignored` 实际验收：302 个发布文件全部
   被跟踪、无发布文件被忽略、模拟再生成的 core/flags/X/parquet/npz 正确进入
   忽略列表（本地仓库，无 remote、未推送）。
5. **许可证状态强化**：`LICENSE` 占位声明保持 all-rights-reserved 并新增
   STATUS 段；候选标准文本 MIT / Apache-2.0 逐字存入
   `docs/license_candidates/` 供权利人择一替换；README §10 与首页横幅措辞
   改为"not yet open-source"。
6. **CITATION.cff 修正**：删除非法的空 `ORCID: ""` 字段；移除 `license` 键
   （以注释说明待权利人选定后设为 SPDX `MIT`/`Apache-2.0`）；摘要与 README
   的投稿状态按事实改为 "Manuscript in preparation for submission to
   Solar Energy"（原"under review"表述删除）；版本号 1.0.0→1.0.1，
   date-released 为本代码归档构建日期。
7. 复现报告 §0/§3.4 的图号与计数行同步更正（验证结论本身不变）。

## 1. 版本 1.0.0（2026-10-27，初始归档）

- 代码冻结基线：原项目 `experiments/rigorous_v2/`（FREEZE_MANIFEST 2026-10-01T20:42）
  + Site 1B 管线 + CQR 基线 + 图表脚本（2026-10-20 前后状态）。

## 2. 代码修改（完整清单）

### 2.1 依赖闭合（新文件，内容为逐字提取）

| 文件 | 修改 |
|---|---|
| `experiments/rigorous_v2/preopen.py` | **新增**。`load_year_preopen` 从原项目 `experiments/diagnose_multiyear_uncertainty.py`（756行诊断脚本）逐字提取，仅模块级路径常量随迁。目的：该诊断脚本的 import 级联会拖入本发布不含的 legacy Stage-D1 工具链。 |
| `experiments/rigorous_v2/common_v2.py` | **1行替换**：`M3 = load_module("stage_m", "experiments/diagnose_multiyear_uncertainty.py")` → `from experiments.rigorous_v2 import preopen as M3`。`M3.load_year_preopen(year, SEEDS)` 调用签名与语义不变。原文件中该行上方已加注释说明。 |

### 2.2 绝对路径 → 仓库相对路径（仅替换路径表达式，不改逻辑）

原项目大量脚本以 `Path("/home/chao1020/PV_BSCN_Project")` 硬编码仓库根目录，
发布版统一改为 `Path(__file__).resolve().parents[2]`（文件位于
`experiments/<子目录>/` 或 `figures/<子目录>/`，`parents[2]` 即仓库根）：

- `experiments/site1b_validation/common_1b.py`（PROJECT_ROOT）
- `experiments/site1b_validation/p8b_site18_cqr.py`、`p9_report.py`、`phase01_audit.py`（ROOT）
- `experiments/cqr_aci/s9_cqr_aci_site18.py`、`s9_cqr_aci_site1b.py`（ROOT）
- `figures/figure_scripts/*.py`（11个脚本，ROOT）
- `figures/latex_v4_repair/*.py`（9个脚本，ROOT）
- `figures/latex_v5_repair/*.py`（3个脚本，ROOT）

### 2.3 图表脚本输出/输入目录适配（目录重排所致）

| 文件 | 修改 |
|---|---|
| `figures/figure_scripts/figstyle.py` | `FIG_DIR` 由 `parents[1]/"figures"` → `parents[1]/"output_stage95"`（发布版输出目录） |
| `figures/latex_v4_repair/figstyle.py` | `FIG_DIR` 由 `parents[1]/"figs"` → `parents[1]/"output_v4"` |
| `figures/figure_scripts/*.py` | 引用 `ROOT/"results"/"stage9_5_v4_visual_completion"/{figure_scripts,figure_source_data,figures}` 的行改为 `ROOT/"figures"/{figure_scripts,figure_source_data,output_stage95}` |
| `figures/latex_v4_repair/*.py`、`latex_v5_repair/*.py` | 引用 `ROOT/"manuscript"/"latex_vX"/{figure_repair,figs}` 的行改为发布版 `ROOT/"figures"/{latex_v4_repair,output_v4,output_v5}` |

上述目录均为图表**输出/汇总数据**位置；无算法内容。

### 2.4 兼容性缺陷修复（行为保持；不影响任何论文数值）

| 文件 | 缺陷 | 修复 |
|---|---|---|
| `experiments/site1b_validation/p8_eval.py`（原69–72行） | CQR checkpoint 已存在时，代码用裸 `MultiScaleTCN` 加载 `CQRNet`（trunk.*/head.* 键）的 state_dict，必然抛 `RuntimeError`。原项目归档运行时 checkpoint 不存在、走的是"现场训练"分支，故从未触发；任何带 checkpoint 的重跑都会崩溃。 | 该分支改为构造与 `common_1b.train_tcn_cqr` 完全一致的 `_CQRNet` 并加载（代码内注释说明）。**论文数值由"现场训练"分支产生，不受影响**；修复使重跑路径可用。归档环境验证：修复后 p8 全量重跑与归档证据逐位一致（max\|Δ\|=0）。 |
| `tests/test_historical_interpolation_causality_audit.py` | 移植的原测试中 `test_target_guard_still_blocks` 依赖本发布不含的 legacy 模块 `src/eval/n_grouped_external_validation.py` | 移除该单测（原文件位置留注释），其余39项测试原样通过。 |

### 2.5 新增文件（非研究代码，均为辅助/验证工具）

| 文件 | 用途 |
|---|---|
| `README.md`、`LICENSE`（占位，见§4）、`CITATION.cff`、`requirements.txt`、`.gitignore` | 发布基础设施 |
| `configs/site18.yaml`、`configs/site1b.yaml`、`configs/experiments.yaml` | 冻结协议常量的人类可读镜像（**管线不读取**；`tests/test_configs_match_code.py` 断言与代码常量一致，防止漂移） |
| `data/README.md` | DKASC 获取/使用/政策说明 |
| `scripts/make_synthetic_example.py` + `data/synthetic_example.csv` | 合成 DKASC 格式数据（冒烟测试/格式示例，无真实观测） |
| `scripts/verify_paper_numbers.py` | 从归档证据 CSV 重算论文 Table 2/3/4/5 全部单元格并与稿面值比对（177项断言） |
| `scripts/replay_s7_seed.py` | 单种子重放 s7 评估并与冻结证据比对（全量 s7 需>1小时，重放数分钟） |
| `tests/`（6个新测试文件 + 2个移植测试） | 语义/架构/配置一致性/权重加载/合成冒烟测试 |
| `figures/README.md`、`docs/*.md`、`legacy/preopen_builders/README.md` | 文档与溯源说明 |

## 3. 内容裁剪（未修改任何代码；详见 MISSING_FILES.md）

**不随库分发**（许可/数据政策）：

1. DKASC 原始 CSV（`data/raw/`，2017–2025，约1.8 GB）；
2. 全部数据衍生数组/表：`X_*.npy`、`*_core.csv`、`*_flags.csv`、
   `site1b_5var_5min.csv`、preopen 目录、逐样本残差 parquet/CSV
   （s3/s4/s5 残差、s7 per_sample parquet、mu_spans.npz、Site 1B
   E22/E23/OOF CSV、CQR 逐样本区间 CSV）；
3. 大型 legacy 依赖闭合（`diagnose_multiyear_uncertainty.py` 及其 Stage-D1
   级联、`stage6_backbone_audit.py` 闭合、`bscn.py` 链、preopen 构建器的
   协议锁 JSON/TCN checkpoint 等）——其中 preopen 构建器两个入口脚本
   **已**逐字保留在 `legacy/preopen_builders/` 供溯源，但注明不可在本库运行。

**随库分发**：全部管线代码（s1–s10、site1b、CQR）、核心 src 模块、
5种子冻结权重（Site 18 + Site 1B + OOF折 + vintage + CQR，共135个 .pt：Site 18 100个 + Site 1B 35个，
SHA-256 与 FREEZE_MANIFEST 对账通过）、聚合结果证据（论文表格数值来源CSV、
manifest、冻结清单、图表源数据、Table S1/S2 证据、论文数值字典）、
可运行的图表脚本。

## 4. 许可证状态

作者/权属尚未最终确认开源许可：`LICENSE` 目前为 **All rights reserved 占位**，
不授予任何再分发/修改权利；`CITATION.cff` 已移除 license 键（1.0.0 版曾为
NOASSERTION），待权利人选定后设为 SPDX `MIT` 或 `Apache-2.0`。
获得权利人确认后应替换为 MIT 或 Apache-2.0 并更新 CITATION.cff。

## 5. 布局说明

按任务要求评估过 `src/{data,models,provenance,calibration,baselines,evaluation}`
的目录方案，**未采用**：真实代码按实验管线组织（`experiments/rigorous_v2`
的 s1–s10 单一协议核心 + Site 1B 适配层），`common_v2.py` 内聚了数据、校准、
评估、训练等职责；强行按功能切分会把该协议核心拆散、大幅增加改动面并提高
行为漂移风险。发布版保留原布局（`src/` 放模型/特征/数据/因果审计等可复用
模块，`experiments/` 放管线），并将权重集中到 `checkpoints/`（管线重跑时
写入 `results/`，两者分离）。此偏离已在 README §2 声明。
