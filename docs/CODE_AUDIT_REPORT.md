# CODE_AUDIT_REPORT — 代码功能、依赖及安全检查报告

> 审计对象：`DMU-NACI/`（本仓库）。对照物：原始科研项目 `PV_BSCN_Project`。
> 审计日期：2026-10-27。方法：**已执行验证**（标注 ✔，附命令/输出）、
> **静态检查**（标注 ◆）、**未核实**（标注 ✖）。配套文件：
> `PAPER_CODE_MAPPING.md`、`REPRODUCIBILITY_REPORT.md`、`CHANGELOG.md`、
> `MISSING_FILES.md`。

## 1. 仓库内容清单与功能判定

| 目录 | 内容 | 功能判定 |
|---|---|---|
| `src/data/` | `schema.py`（DKASC列名/特征序/窗口常数）、`preopen.py`（2024/2025冻结宇宙加载） | ✔ 导入+加载验证（s2 冒烟读取 54,877 行） |
| `src/models/` | `multiscale_tcn.py`（36,705参数）、`tcn_blocks.py`（因果TCN，带weight-norm）、`uncertainty_encoders.py`（GRU/MLP/TCN/PatchTST编码器+ScaleModel） | ✔ 71项权重加载测试逐一通过；参数量与论文一致 |
| `src/features/` | `dynamic_index.py`（D_t 动态指数，权重0.35/0.30/0.20/0.15）、`historical_familiarity.py`（12维上下文/2021记忆/N_t）、`scaler.py` | ✔ 16项熟悉度单测通过；动态指数斜率差取绝对值与论文 eq.(13) 一致（◆ 代码走查） |
| `src/eval/` | `historical_interpolation_causality_audit.py`（未来插值端点依赖审计=防泄漏窗口机制） | ✔ 39项移植单测通过 |
| `experiments/rigorous_v2/` | `common_v2.py`（协议核心：角色边界断言、清洗调用、scaler、训练器、指标、ACI、bootstrap）+ `s1`–`s8`、`s10`、`preopen.py` | ✔ s7 全量重跑逐位一致；◆ s1/s3/s4/s5/s6/s8/s10 逐行走查（导入✔，执行需数据） |
| `experiments/site1b_validation/` | `common_1b.py`（Site 1B 适配层，复用 common_v2 算法）+ `phase01/p2/p34/p567/p8/p8b/p9` | ✔ p8 全量重跑逐位一致；◆ p2/p34/p567/p9 走查 |
| `experiments/cqr_aci/` | `s9_cqr_aci_site18.py`、`s9_cqr_aci_site1b.py`（CQR复现门控+CQR-ACI） | ✔ site18 全量重跑逐位一致 |
| `figures/` | 3层图表脚本 + 9个可发布源数据CSV + README（逐图溯源） | ✔ 5张论文图逐字节复现 |
| `legacy/preopen_builders/` | 37rh/37rj 预开宇宙构建器（逐字保留） | ◆ 静态走查；✖ 本库内不可运行（依赖闭合缺失，已声明） |
| `scripts/` | 合成数据生成、论文数值核验、单种子重放 | ✔ 全部执行通过 |
| `tests/` | 8个测试文件，125项 | ✔ 125/125 通过 |
| `checkpoints/` | 135个冻结 .pt（Site 18：5 mean + 20 OOF + 15 vintage + 55 GRU + 5 CQR；Site 1B：5 mean + 5 GRU + 20 OOF + 5 CQR）+ 2个 scaler + μ缓存 | ✔ SHA-256 与 FREEZE_MANIFEST 对账；全部可加载 |
| `results/` | 聚合证据 CSV/JSON/MD（表格数值来源、manifest、冻结清单、图表源数据、Table S1/S2、论文数值字典） | ✔ verify_paper_numbers 177项比对通过 |
| `configs/` | 协议常量YAML镜像 | ✔ 一致性测试通过（管线不读取，已声明） |

无占位/无功能文件；未创建虚假脚本填充目录。

## 2. 依赖审计 ◆→✔

- 直接依赖（`requirements.txt`）：numpy、pandas、scipy、torch、pyarrow、
  matplotlib、pyyaml、pytest。在发布树内 `import` 全量通过（29个模块+18个脚本）。
- 传递依赖：仅上述包的标准生态（无 git/URL 依赖、无本地 wheel）。
- 第三方许可：BSD/Apache/MIT 系，均允许学术使用与再分发；
  **本仓库未再分发任何第三方包**，LICENSE 中已列明归属。
- `requirements.txt` 未硬锁版本（记录了论文环境精确版本于注释）。
  ✖ 未在其他 Python/torch 版本组合上测试兼容性。

## 3. 数据与发布安全审计

### 3.1 数据再分发 ✔

- 原始 DKASC CSV（1.8 GB）**未**包含于仓库；`data/README.md` 给出官方来源
  （https://dkasolarcentre.com.au）、所需年份、精确文件名与列 schema。
- 数据衍生数组/表（X_*.npy、core/flags、preopen、Site 1B 清洗表、逐样本
  parquet/CSV）**未**包含；清单见 `MISSING_FILES.md` §1。
- 随库保留的模型派生小文件（μ缓存 npz、CQR CALIB 分数 CSV、聚合指标）
  均为模型输出/统计量，不构成观测数据再分发。

### 3.2 敏感信息 ✔（扫描命令与结果）

```
$ grep -rn "/home/chao1020" --include="*.py" --include="*.yaml" ... .
（无匹配 —— 全部硬编码路径已改为仓库相对路径，见 CHANGELOG §2.2）

$ grep -rnEi "api_key|secret|passw|token|ghp_|AKIA" --include="*.py" ...
（仅3处为普通变量名：文件名 token 检查、禁用列名清单、PatchTST 词元；
 无任何凭据/令牌）

$ find . -name "*Zone.Identifier*" -o -name "__pycache__" -o -name "*.pyc"
（发布前已清理；.gitignore 持续防护）
```

- 无个人账户、密钥、令牌、私人服务器路径；代码内无作者邮箱
  （唯一邮箱在 `CITATION.cff` 通讯作者字段，属论文公开元数据）。
- 证据 JSON/CSV 中无绝对路径（`results/`、`checkpoints/` 全量扫描通过）。
- `.gitignore` 覆盖：原始/衍生数据、pycache、编辑器文件、Zone.Identifier、
  管线再生成产物；同时注明 checkpoints/ 与聚合证据**有意**纳入版本管理。

### 3.3 许可证 ◆（待作者确认）

- **未擅自授予开源许可**：`LICENSE` 为 All-rights-reserved 占位，
  声明仅限论文复现用途；`CITATION.cff` license=NOASSERTION。
- 建议路径：确认全体作者权属后采用 MIT 或 Apache-2.0，并替换该文件。
- 代码中无第三方代码片段拷贝（◆：核心模块均始于本项目的开发历史；
  tcn_blocks 为项目自实现因果TCN）。

## 4. 代码规范整理 ◆

- 保留了原有的算法解释性 docstring（含协议断言的语义说明）与冻结常量注释；
  未发现聊天式注释/模板占位文本需要清理（原管线代码即工程化风格）。
- 命名、导入方式、日志输出保持原样（未改变任何实验行为）；`print` 为
  原脚本的进度输出方式，未引入新日志框架以免改变行为。
- 图表脚本中的 docstring 提及原项目目录（manuscript/latex_v5 等）之处
  保留原文并已在 `figures/README.md` 解释对应关系——不掩盖来源。
- 未添加任何"独立编写/原创性"声明。

## 5. 已识别的问题与处置（完整记录见 CHANGELOG）

| # | 发现 | 处置 | 对论文数值影响 |
|---|---|---|---|
| 1 | 28个文件硬编码绝对路径 | 改为仓库相对路径 | 无 |
| 2 | `common_v2` 依赖 legacy 诊断模块（仅为一个函数） | 逐字提取为 `preopen.py` | 无（重跑逐位一致） |
| 3 | `p8_eval.py` CQR checkpoint 加载分支用错模型类（原项目死代码，重跑必崩） | 加载至与训练一致的 CQRNet（内联同类） | 无（原结果由现场训练分支产生；修复后重跑逐位一致） |
| 4 | 移植测试引用未随库的 legacy 模块 | 移除该单测并注释 | 无 |
| 5 | s7 内 `state_split` 的 HIGH_N 为互斥口径，论文 Table 4 用包含口径（`conditional_correct_per_seed.csv`） | **保持原样**，双口径均在 `PAPER_CODE_MAPPING.md` §5 如实记录 | 无（论文数值来源文件随库） |
| 6 | pandas 3.0 下若干 numpy 除零 RuntimeWarning（如空月度分组） | 保持原样（原项目同环境同警告） | 无 |

## 6. 复核检查单（验收对照）

- [x] 仓库结构完整（README §2；目录为管线真实结构，偏离已说明）
- [x] 核心实验逻辑与论文一致（s7/p8/s9 重跑逐位一致 + 177项表值核对）
- [x] 代码和配置可追溯（逐文件复制+修改清单；SHA-256 冻结清单对账）
- [x] 真实运行结果有证据支持（REPRODUCIBILITY_REPORT 全部附命令与数值）
- [x] 不含未授权数据（§3.1）、无用文件（§1）、虚构内容（全部权重/结果为
      原项目冻结产物，无任何生成/伪造）
- [x] 未修改原始项目（全部操作在 `DMU-NACI/` 与 /tmp 沙箱；原项目只读）
- [x] 未向 GitHub 推送（无 remote、无 push 动作）
