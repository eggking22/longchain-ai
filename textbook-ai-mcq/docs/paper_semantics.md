# Paper Figure Semantic Reconstruction — 设计与用法

> **定位**：从生物学科研论文的**文本**（Caption / Results / Methods / Discussion / 引用关系）恢复每个 Figure/Table 背后的实验语义与结论。
> 不做：图片解析、OCR、Vision Model、数值重建、MCQ 生成。

## 1. 架构

```text
data/structure/{doc_id}/document.json      (Phase 1 产物，只读)
data/index/{doc_id}/                        (Phase 2 产物，只读，可选)
        ↓
flatten + IMRaD 章节分类                sections.py
        ↓
Caption / 正文引用抽取（规范化 id）        figure_reference.py
Figure 2 ← "Fig. 2B" "figure 2a" 等
        ↓
证据收集（角色 + provenance）             evidence_collector.py
caption=direct / Results=direct / Methods=supporting / Discussion=interpretation
        ↓
确定性实验语义重建                        experiment_model.py + conclusion.py
组别 / 自变量 / 因变量 / 方向 / 显著性 / 关系类型 / 结论（全部绑定 evidence_id）
        ↓
Semantic Evidence Gate                    gate.py
SUFFICIENT / PARTIAL / INSUFFICIENT
        ↓
可选 LLM 归一化（证据绑定校验，违规即 REJECT）  llm_normalizer.py
        ↓
data/paper_semantics/{doc_id}/            persistence.py
figures.json + experiments.json + manifest.json
```

**只读复用承诺**：Phase 1/2/3 的代码与产物零修改。`RetrievalEngine` 通过
`retrieval_adapter.py` 包装（index 不存在时自动跳过）；`EvidenceGate`（Phase 3）
不参与本模块，新的 `SemanticEvidenceGate` 独立实现三级判定。

## 2. CLI

```bash
# 基本用法（需先经 Phase 1 解析：python scripts/parse_pdf.py paper.pdf --doc-id example-paper）
python scripts/reconstruct_figures.py --doc-id example-paper

# 只看某个 figure / 完整 JSON
python scripts/reconstruct_figures.py --doc-id example-paper --figure 2
python scripts/reconstruct_figures.py --doc-id example-paper --json

# 关闭可选能力
python scripts/reconstruct_figures.py --doc-id example-paper --no-llm --no-retrieval
```

Exit codes：`0` = 重建完成（不区分各 figure 的状态）；`1` = 输入缺失等错误。

人类可读输出示例：

```text
Figure 2
  status: SUFFICIENT  confidence=1.0  method=deterministic
  experiment: Does Treatment A affect gene X expression?
  groups: Treatment A vs control
  observation: Treatment A significantly increased expression of gene X compared with control.
      direction=increase  significance=significant  relationship=causal
  conclusion: Treatment A increases gene X expression.  [ev_002]
  evidence: 4 item(s)
--------------------------------------------------------------------
Figure 3
  status: INSUFFICIENT  confidence=0.0  method=deterministic
  missing:
    - figure caption
    - independent variable / intervention
    - ...
```

## 3. 数据模型（app/schemas/paper_semantics.py）

| 模型 | 说明 |
|---|---|
| `PaperEvidence` | 一条文本证据：`evidence_id`（ev_NNN）、`role`（来源维度：caption/direct/supporting/interpretation）、`evidence_type`（语义维度，见 3.1）、`section_type`、`paragraph_id`、`page_no`、`breadcrumb`、`chunk_id`（检索来源）、`panel_id`（面板绑定） |
| `FigureReference` | 规范化 figure/table 身份：caption 段落（含段内切片偏移）、正文引用段落、面板归档（`panel_texts` / `panel_mention_paragraph_ids`） |
| `Observation` | **第一层（观察）**：Results/caption 中的结果陈述，direction / significance / relationship_type / p_value（仅字面）/ evidence_ids |
| `Interpretation` | **第二层（解释）**：Discussion 中作者解释句的**原文**（作者归因主张，非系统断言），relationship_type 保持作者原始强度 |
| `Conclusion` | **第三层（结论）**：仅由 Observation 模板合成；`evidence_ids` 指向观察证据，`interpretation_ids` 关联同方向解释层（不改结论文本） |
| `ExperimentModel` | 实验逻辑：研究问题、假设、subjects、自变量/因变量、实验组/对照组、干预、测量、observations、interpretations、statistical_results（仅字面）、conclusions |
| `PanelSemantic` | 面板级独立语义单元（2a/2b/…）：自己的证据包、实验模型、门控判定；面板证据 id 前缀 `ev_2a_001` |
| `FigureSemantic` | 单 figure 重建结果 + `panels` 列表；`evidence_of_type()` 按语义类型查询证据 |
| `PaperSemanticsReport` | 顶层产物：doc_id、figures、stats（含 panel_status_counts） |

### 3.1 证据双维度：role（来源）与 evidence_type（语义）

| 来源（role） | evidence_type |
|---|---|
| Results 正文段（direct） | `direct_observation` |
| Methods 段（supporting） | `experimental_design` |
| Discussion 段（interpretation） | `author_interpretation` |
| caption | 按内容：方向/显著性标记 → `direct_observation`；组别/处理词 → `experimental_design`；统计注记 → `statistical_result` |
| 检索补充（adapter） | `mixed` |

caption 中的统计注记（"Data were analyzed by …; ****P < 0.0001"）保留为**句级
`statistical_result` 证据单元**（不进 observation/conclusion，字面 p 值规则不变）。

未来出题约束示例（本阶段只提供查询能力）：

```python
figure.evidence_of_type("direct_observation")  # 结论型题必须有 Results 证据
```

### 3.2 Panel 语义

- caption 按面板边界切分（Nature 风格 `a, GFP intensity … b, Expression …`），
  正文引用按面板字母归档（`Fig. 2B` → panel b；无字母引用留在图级，不摊派）；
- 面板证据包 = 面板 caption 切片 + 明确引用该面板的正文段落（Methods/Discussion
  不重复切割，留在图级共享）；
- 面板独立走 `build_experiment` + `SemanticEvidenceGate`（复用，不复制）；
- **图级状态不随面板自动升降级**（面板是同一证据的更细粒度诚实判定）。

## 4. 证据规则（Evidence Rules）

优先级：`caption > Results > Methods > Discussion`。

- **角色**：caption 与 Results 中明确提及该 figure 的段落是 `direct`；Methods 是
  `supporting`（按与 caption/direct 的内容词重叠 ≥2 打分，取前 N 段，避免多篇混
  实验论文串台）；Discussion 是 `interpretation`。
- **每条证据可溯源**：evidence_id → paragraph_id（DocNode node_id）/ page_no /
  breadcrumb / section。结论与观察必须引用 evidence_id。

### 4.1 三级判定

| 状态 | 判据 |
|---|---|
| `SUFFICIENT` | 有 caption + 实验设计（组别，或自变量+因变量）+ 结果方向 |
| `PARTIAL` | 能识别实验（caption 或变量/组别）但缺核心槽位（典型：缺结果方向） |
| `INSUFFICIENT` | 只有 "Figure 3 shows the results." 类引用，无 caption、无设计、无结果描述 |

`confidence` = 已恢复核心槽位的加权和（caption .2 / IV .15 / DV .15 / 组别 .15 /
方向 .25 / 显著性 .1）。

### 4.2 科学性安全约束

- **association ≠ causation**：`correlated with` → correlation、`associated with` →
  association，绝不改写为因果句式；因果读法需要显式因果动词或对照设计语言
  （`compared with control` / `in response to` 等）。
- **禁止数值伪造**：`significantly increased` 只产生 `significance=significant`，
  不会变成 `p < 0.05`；p 值 / 倍数变化只记录正文中字面写出的值。
- **证据不足即 INSUFFICIENT**：不猜测方向、组别或结论。

## 5. LLM 归一化（可选）

激活条件与 Phase 3 一致：`.env` 同时配置 `LLM_API_KEY` 与 `LLM_MODEL`（CLI
`--no-llm` 可强制关闭）。复用 `LlmEvidenceEvaluator` 的访问模式：OpenAI 兼容
chat/completions、temperature 0、严格 JSON（一次 re-ask）、可注入 httpx client。

流程：确定性草稿 → LLM 只做措辞归一/补全（prompt 中仅提供带 id 的证据文本）→
确定性校验：

1. 每条 conclusion 引用的 `evidence_ids` 必须存在于证据包；
2. 输出中出现的数字 / p 值必须字面存在于证据文本；
3. conclusion 的方向词必须被证据支持。

任一违规 → **REJECT 整个 verdict**，保留确定性结果（`FigureSemantic.detail.llm_rejected`
记录原因）；网络/JSON 失败同样不阻塞重建（`detail.llm_error`）。

已知限制：校验保证"证据绑定 + 无伪造数值 + 方向有据"，但不做自由文本的完全
忠实性验证（如 LLM 把 Treatment A 的结论安到 Treatment C 上且方向词恰好存在时
无法确定性判伪）。MVP 依赖 prompt 约束 + 上述客观校验。

## 6. 产物格式

存储设计参考开源文献阅读栈：S2ORC（参考实体单独存一份、其余按 id 引用）、
Docling（机器 JSON + 人类可读 Markdown 伴侣、全元素 provenance）、PDFFigures2
（紧凑 figure 记录与 bulk 数据分离）、GROBID/PubReader（章节标题变体词表）。

```text
data/paper_semantics/{doc_id}/
├── figures.json     # 语义索引：background（Abstract+Introduction 提取式摘要）
│                    #   → summary（图/面板/证据统计）→ 每 figure 紧凑语义块
│                    #   （三层链 + panels + text_block；证据只存 id 引用）
├── evidence.jsonl   # 单一证据库：一行一个证据单元（图级+面板级，按
│                    #   evidence_id 去重；含 role/evidence_type/assignment）
├── experiments.json # 完整 ExperimentModel（图级 exp_f02 + 面板级 exp_f02a）
├── report.md        # 人类可读 Markdown：Background → 逐图分节（OBS/INT/CON）
└── manifest.json    # reading_index（IMRaD 段落分布、图表清单、L1 划分覆盖率）
                    #   + 输入 sha256 + 文件清单 + created_at（唯一时间戳）
```

所有内容文件零时间戳、字节级可复现；不触碰 `data/raw|structure|chunks|index`。

### 6.1 Background（开头摘要，纯提取式零改写）

- **Abstract 双路径**：① Abstract/Overview 标题下段落；② Nature 无标题格式 →
  开头正文段中词数最大的一段（标题/作者行/上标人名段被确定性过滤）
- **Introduction**：introduction 章节段落（句边界截断 ~1500 字符）；无标题时回退
  为"摘要段之后、首个 figure 锚点之前"的叙述段（词预算 400）
- 两者均保留 paragraph_ids + pages 溯源

### 6.2 L1 正文划分（Results → per-figure text blocks）

```text
锚点：显式引用某图的段落（面板引用归父图）
继承：锚点之后的无引用段落归当前块（assignment=continuation）
换块：引用另一图的锚点关闭当前块
多图段：主归属首个提及，shared_with 记录其余
安全阀：首个锚点之前的段落不归属任何图
回退：无 Results 章节分类时（Nature 小字体标题常被 Phase 1 漏检），流 =
  other/results 段 ∩ [首锚点, 末锚点] 区间（IMRaD 两种排版下 Methods 均在区间外）
```

**语义安全**：继承段只扩大证据存档范围（每图至多
`max_continuation_evidence=3` 段，避免挤占 Methods/Discussion 名额），**不参与
语义抽取**——观察/组别/变量/结论仍只来自锚点、caption、Methods、Discussion。
L2（可选 LLM 仲裁）为后续独立阶段，本轮未实现。

## 7. 环境变量

```bash
# Paper figure semantics（均有默认值）
PAPER_SEMANTICS_MAX_EVIDENCE_PER_FIGURE=12   # 单 figure 证据上限
PAPER_SEMANTICS_MAX_METHODS_PARAGRAPHS=4     # Methods 支持证据取前 N 段
PAPER_SEMANTICS_CAPTION_MAX_CHARS=400         # 超长的 caption 起始段不视为 caption

# LLM 归一化（可选，同 Phase 3 约定）
LLM_API_KEY=...
LLM_MODEL=...
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
```

## 8. 验证集（tests/paper_semantics/，286 全量 pytest 通过）

| 用例 | 输入要点 | 期望 |
|---|---|---|
| Figure 2 | caption + Results(方向/显著性/对照) + Methods + Discussion | SUFFICIENT；结论 "Treatment A increases gene X expression." 绑定 ev_002 |
| Figure 3 | 仅 "Figure 3 shows the experimental results." | INSUFFICIENT；missing 含方向/组别；零观察零结论 |
| Figure 4 | caption 给出因变量、无结果句 | PARTIAL；missing 含 direction of change |
| Figure 5 | "was associated with increased ..." | association（非 causal）；结论用 is associated with |
| Table 1 | 表格 caption + decrease 结果 | SUFFICIENT；方向 decrease |
| p 值安全 | "significantly increased" | significance=significant 且 p_value=None |
| 数值安全 | 各结论 | 不含任何正文没有的数字 |
| LLM 幻觉 | conclusion 引用 ev_999 / 捏造 2.37-fold / p<0.01 / 反向 | 全部 REJECT，确定性结果保留 |
| 端到端 | 合成英文论文 PDF → 真实 ingest → 重建 | Figure 2 SUFFICIENT；Phase 1 产物字节不变 |
| 可复现 | 连续两次运行 | figures.json/evidence.jsonl/report.md 字节一致 |
| 面板切分 | `a, GFP intensity … b, Expression …`；`a, b, c` 枚举不误切 | panel_texts 归一化小写标签；无假面板 |
| 面板归因 | "Fig. 2a shows…" vs "Figure 2 shows…" | 面板引用进 panel_mention_paragraph_ids，图级引用不摊派 |
| 面板重建 | 面板 a（caption 切片 + Results 面板引用） | 独立 SUFFICIENT；experiment_id=exp_f02a；证据 id ev_2a_001 |
| baseline 锁定 | 面板存在时图级判定 | 图级 status/confidence/结论不变（黄金集 3/1/1 保持） |
| 证据类型 | role/evidence_type 映射矩阵 | direct→direct_observation、supporting→experimental_design、interpretation→author_interpretation |
| 统计注记 | caption 含 "Data were analyzed …; ****P < 0.0001" | 保留为 statistical_result 句级证据；不进 observation |
| 出题约束 | `evidence_of_type("direct_observation")` | Figure 2 非空；Figure 3 有类型但 gate 阻止结论 |
| 三层分离 | Figure 2 黄金集 | 观察证据 ∩ 解释证据 = ∅；结论引用观察证据并链接 int_001 |
| 方向词安全 | "treated with the inhibitor" | 施动名词（inhibitor/promoter/enhancer）不产生假方向 |
| 解释层 | Discussion 句 | 原文保留 + int_NNN 编号；association 不改因果；方向不匹配不链接 |
| L1 划分 | 锚点/继承/换块/多图共享/面板归父图 | 见 test_text_partition.py 全规则矩阵 |
| L1 安全阀 | Results 开头无锚点段；Methods/Discussion | 不归属任何图；不参与划分 |
| 继承安全 | 继承段含方向标记句 | 入证据库（assignment=continuation）但绝不产生观察 |
| Background-Abstract | 标题路径 / Nature 无标题（最长段）/ Reporting summary 误报 | 双路径正确；文末 Reporting summary 不误抓 |
| Background-Intro | introduction 章节 / 无标题回退（摘要后至首锚点） | 作者名单段过滤；词/句预算截断；溯源完整 |
| 存储可读性 | figures.json 结构 | background 最前；证据只存 id（无 paragraph_id 内联）；text_block 呈现 |
| 证据库 | evidence.jsonl | 跨图/面板无重复 id；figures.json 引用可全部回链 |

```bash
pytest tests/paper_semantics/ -q     # 152 passed
pytest -q                            # 286 passed（旧 134 + paper_semantics 152）
```

## 9. 局限（MVP 边界）

- **仅英文论文模式**：中文论文（图N/表N）当前只会得到 0 figure 或 INSUFFICIENT，
  不做猜测（后续可按需扩展双语模式库）。
- 语义模式覆盖常见句式（increase/decrease、表达/活性类指标、correlation/
  association、knockout/overexpression、dose-dependent 等）；复杂机制图、时间序列
  图的面板级归因仍是句级启发（面板引用不指定字母时留在图级）。
- 方向词只认动词/事件形式（"decreased"、"upregulation"）；施动名词
  （inhibitor/promoter）不触发方向——但 "inhibitor" 仍会把实验标记为 inhibition
  **关系类型**（实验设计属性，非结果断言）。
- 物理干预（confinement、stretching 等）不在处理模式词表内，相关 figure 诚实保持
  PARTIAL 并报告缺失的自变量/组别。
- 精确数值恢复、图像视觉分析、MCQ 生成均不在本阶段范围；LLM 归一化已实现但默认
  关闭（未配置 LLM_API_KEY 时纯确定性路径即为推荐 baseline）。
- 依赖 Phase 1 的结构质量：若 IMRaD 标题未被识别，章节类型回退为 other，此时
  引用段落仍按 direct 收集（鲁棒回退），但 Methods/Discussion 的角色区分失效；
  双栏断行可能截断引用（"(Fig." 后无编号），无法关联。

## 10. 设计依据

- Evidence-bound generation：与 Phase 3 的两级证据门思想一致（Sufficient
  Context、CRAG、RAGAS，见 docs/evidence.md），本模块把"问题覆盖度"换成
  "figure 语义槽位完备度"。
- LLM 归一化 + 确定性校验的模式对应 "LLM 负责语义归纳、不负责无证据猜测" 的
  约束；引用绑定式结论生成参考科学信息抽取（SciREX / RE 任务）中以 span 为
  锚点的做法。
