# textbook-ai-mcq

基于生物学教材的长文本选择题生成系统（Long-text MCQ generation from biology textbooks）。

当前处于 **Phase 5：Question Blueprint（已完成）**——
在 Paper Semantic Reconstruction → Experiment Model 之上新增确定性出题蓝图层：
RESULT_INTERPRETATION（结果理解）/ EXPERIMENTAL_DESIGN（实验设计）/
SIMPLE_PREDICTION（简单预测，仅因果许可关系且答案锚定已观测组）/
DATA_STATEMENT（数据陈述，仅正文字面数值）。Evidence-first：每条蓝图的
evidence_ids 全部回链 Evidence Store；association 图零预测题；证据不足即不出题
（跳过原因透明计数）。无 MCQ 题干/选项/答案键、无 LLM（deterministic baseline，
未来 L2 refinement 可在其上扩展）。

此前 **Phase 4：Paper Figure Semantic Reconstruction（已完成）**——
从英文科研论文的文本证据（Caption / Results / Methods / Discussion / 引用关系）
恢复每个 Figure/Table 背后的实验语义与结论：实验组/对照组、自变量/因变量、
结果方向、显著性、关系类型（association ≠ causation）、可溯源结论。
证据不足时给出 `INSUFFICIENT`（不猜测），可选 LLM 归一化必须绑定 evidence_id
（违规即 REJECT 回退确定性结果）。不含 MCQ Generator（下一阶段）。

此前 **Phase 3：Evidence Gate（已完成）**——
Query → Hybrid Retrieval → 证据充分性评估（Level 1 启发式 / Level 2 可选 LLM 复核）
→ 门控决策。证据不足时给出 `INSUFFICIENT_EVIDENCE` 与缺失信息清单，不强行生成。

此前 **Phase 2：Hybrid RAG 检索层（已完成）**——
Phase 1 的 Chunks → 幂等索引 → dense（余弦）+ sparse（BM25）双路 → RRF 融合 →
带完整 provenance 的 Top-K 结果。embedding 走 OpenAI 兼容 API（默认智谱 embedding-3），
测试/离线场景用确定性 hash provider。

**Phase 1：可控的 Hierarchical PDF Parser（已完成）**——
PDF → 结构化教材树（Document→Chapter→Section→Paragraph）→ 检索用 Chunk。

## 技术栈

- Python 3.11（conda 环境 `bio-ai`，禁止新建虚拟环境）
- FastAPI + Pydantic
- SQLAlchemy（PostgreSQL / pgvector 接口已预留，暂未连接）
- PyMuPDF（PDF 解析核心，自研层级解析，未引入 Docling/pymupdf4llm 依赖）
- python-docx（Word 导出，后续阶段使用）
- jieba + rank-bm25 + numpy（Phase 2 检索：中文分词、BM25、精确余弦向量检索）
- httpx（embedding API 直调，OpenAI 兼容端点）
- pytest

## Parser 流水线

```
uploads/example.pdf
   │  parser.py    raw lines/spans + TOC 书签 + 字体统计
   ▼
data/raw/{doc_id}/lines.json / toc.json / fonts.json
   │  cleaner.py   页眉页脚/页码剔除（跨页重复组）、空白归一
   ▼
data/raw/{doc_id}/lines.clean.json + cleaner.debug.json
   │  structure.py 标题检测（toc > font > numbering > spatial 优先级瀑布）
   │               栈式建树 + 段落归并（缩进/句末标点/跨页续接）
   ▼
data/structure/{doc_id}/document.json + headings.debug.json + stats.json
   │  chunking.py  段落感知分块（默认不跨节、面包屑元数据、句级重叠）
   ▼
data/chunks/{doc_id}/chunks.jsonl
```

可控性：所有阈值集中在 `app/services/parser/config.py`（可经 `.env` 覆盖）；
每个标题的判定规则与置信度记录在 `headings.debug.json`；每阶段产物落盘，
可用 `--reuse-raw` 断点重跑。

## Retrieval 流水线（Phase 2）

```
data/chunks/{doc_id}/chunks.jsonl                （Phase 1 产物，只读）
   │  indexer.py   补 document_id/chunk_index → 过短过滤 → jieba 分词 → embedding
   ▼
data/index/{doc_id}/manifest.json + records.jsonl + embeddings.npy
   │  engine.py    query → [dense 余弦 top-20] + [BM25 top-20] → RRF(k=60) → top-k
   ▼
RetrievalResult（breadcrumb/pages/双路分数/命中来源，可经 scripts/search.py 输出）
```

幂等可复现：manifest 记录 config_hash + chunk_set_hash + embedder 身份，
未变更时重复建索引直接 skip（不产生 API 费用）。CLI 与产物格式详见
`docs/retrieval.md`。

```bash
python scripts/build_index.py --doc-id physiology          # 用 .env 里的 embedding API
python scripts/build_index.py --doc-id physiology --embedder hash   # 离线确定性 provider
python scripts/search.py "心动周期" --doc-id physiology --top-k 5
```

## Evidence Gate（Phase 3）

```
Query → Hybrid Retrieval → EvidenceGate.evaluate()
   ├─ Level 1 启发式（永远运行）：0.75×词项覆盖(union+focus) + 0.15×BM25强度 + 0.10×命中深度
   │    sufficient ⟺ coverage_score ≥ 0.80；missing_information = 未覆盖的查询实质词
   └─ Level 1 未过 且 LLM 已配置（LLM_API_KEY+LLM_MODEL）→ LLM 分解复核（RAGAS 模式），
        失败自动降级 Level 1
→ CoverageReport{sufficient, coverage_score, evidence[], missing_information[]}
→ gate.require() 在不足时抛 InsufficientEvidenceError（下游生成器的强制入口）
```

验证集（`tests/test_evidence_suite.py`）：5 个 sufficient + 5 个 insufficient 查询，
合成语料与 physiology 真实教材均 **10/10，precision=recall=1.0**。
设计与调研依据详见 `docs/evidence.md`。

```bash
python scripts/evidence_gate.py "糖酵解的发生部位是哪里？" --doc-id physiology     # SUFFICIENT, exit 0
python scripts/evidence_gate.py "2025年诺贝尔生理学奖对糖酵解研究的影响" --doc-id physiology  # INSUFFICIENT, exit 2
```

## Paper Figure Semantic Reconstruction（Phase 4）

```
data/structure/{doc_id}/document.json        （Phase 1 产物，只读）
   │  sections.py            DocNode 树展平 + IMRaD 章节分类（Introduction/Methods/Results/Discussion）
   ▼
   │  figure_reference.py    Caption 段落识别 + 正文引用抽取（Figure 2 / Fig. 2B / Table 1 → 规范化 id）
   ▼
   │  evidence_collector.py  按 caption > Results > Methods > Discussion 收集证据（角色 + provenance）
   ▼
   │  experiment_model.py    确定性重建：组别/自变量/因变量/方向/显著性/关系类型（英文模式库）
   │  conclusion.py          结论合成（全部绑定 evidence_id；association/correlation 绝不写成因果）
   ▼
   │  gate.py                Semantic Evidence Gate：SUFFICIENT / PARTIAL / INSUFFICIENT
   ▼
   │  llm_normalizer.py      可选 LLM 归一化：仅措辞补全，证据 id/数值/方向校验违规即 REJECT
   ▼
data/paper_semantics/{doc_id}/figures.json + experiments.json + manifest.json（字节级可复现）
```

Phase 2 检索经 `retrieval_adapter.py` 只读包装为 figure 感知的补充证据源
（index 不存在时自动跳过）。证据带双维度标注（来源 role + 语义
evidence_type：direct_observation / experimental_design / statistical_result /
author_interpretation），多面板图按 `2a/2b/…` 独立重建（panel 级证据包与门控
判定，图级判定不受影响），推理链分三层：Observation（Results 观察）→
Interpretation（Discussion 作者解释，原文保留）→ Conclusion（仅由观察合成，
按 id 链接解释层）。存储参考 S2ORC/Docling/PDFFigures2 模式：`figures.json`
以 **Background（Abstract+Introduction 提取式摘要）** 开头，语义索引只存证据
id 引用；`evidence.jsonl` 单一证据库；`report.md` 人类可读逐图分节报告；
Results 正文按 **L1 确定性锚点+继承** 划分为 per-figure 文本块（锚点=显式引用
为硬事实，继承段只入存档不参与语义抽取）。验证集（`tests/paper_semantics/`，
152 项）：合成论文黄金集 5 figure 全对、"significantly" 不伪造 p 值、LLM 幻觉
证据全 REJECT、真实 PDF ingest 端到端通过、面板/划分存在时图级 baseline 不变。
设计与证据规则详见 `docs/paper_semantics.md`。

```bash
python scripts/reconstruct_figures.py --doc-id example-paper           # 全文重建
python scripts/reconstruct_figures.py --doc-id example-paper --figure 2 --json
```

## 目录结构

```
textbook-ai-mcq/
├── app/
│   ├── main.py              # FastAPI 入口
│   ├── core/                # 配置（pydantic-settings）
│   ├── db/                  # 数据库（后续阶段）
│   ├── models/              # ORM（后续阶段）
│   ├── schemas/
│   │   ├── document.py      # DocNode 树 / Chunk / Provenance
│   │   ├── retrieval.py     # IndexManifest / IndexedRecord / RetrievedChunk / RetrievalResult
│   │   ├── evidence.py      # EvidenceItem / CoverageReport / LlmCoverageVerdict
│   │   ├── paper_semantics.py # PaperEvidence / FigureReference / ExperimentModel / FigureSemantic 等
│   │   └── question_blueprint.py # QuestionBlueprint / QuestionBlueprintReport
│   ├── routers/
│   │   ├── health.py        # GET /api/v1/health
│   │   └── documents.py     # 上传/结构/chunks 接口
│   ├── services/
│   │   ├── parser/           # Phase 1：解析流水线（parser/cleaner/structure/chunking/…）
│   │   ├── retrieval/        # Phase 2：混合检索（indexer/engine/embeddings/vector_store/bm25/fusion/tokenizer）
│   │   ├── evidence/         # Phase 3：证据门控（evaluator L1+LLM/gate/config/stopwords）
│   │   ├── paper_semantics/  # Phase 4：论文图表语义重建（figure_reference/evidence_collector/experiment_model/gate/llm_normalizer/…）
│   │   └── question_blueprint/ # Phase 5：出题蓝图（numeric 字面数值抽取/generators 四题型门控/pipeline）
│   ├── llm/                 # LLM 封装（后续阶段）
│   └── utils/
├── data/                    # 解析与索引产物（raw/structure/chunks/index/paper_semantics，均 gitignored）
├── uploads/                 # 上传的 PDF
├── docs/                    # 说明文档（parse_pdf.md、retrieval.md、evidence.md、paper_semantics.md、question_blueprint.md）
├── prompts/                 # 提示词模板（后续阶段）
├── scripts/parse_pdf.py           # Phase 1 解析 CLI
├── scripts/build_index.py         # Phase 2 建索引 CLI
├── scripts/search.py              # Phase 2 检索 CLI
├── scripts/evidence_gate.py       # Phase 3 证据门控 CLI
├── scripts/reconstruct_figures.py # Phase 4 图表语义重建 CLI
├── scripts/generate_blueprints.py # Phase 5 出题蓝图 CLI
├── tests/                   # 合成 PDF 单测 + e2e golden + API 测试
│   └── paper_semantics/     # Phase 4/5 测试（引用抽取/证据收集/实验模型/结论/门控/LLM/检索适配/划分/背景/存储/蓝图）
├── requirements.txt
├── .env.example
├── .gitignore
└── pytest.ini
```

## 快速开始

```bash
# 1. 安装依赖（使用已存在的 bio-ai conda 环境）
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env   # 按需修改

# 3. 运行测试
pytest

# 4. 解析一本教材 PDF（CLI）
python scripts/parse_pdf.py uploads/example.pdf --doc-id bio-grade1
#    参数、产物位置与格式的详细说明见 docs/parse_pdf.md

# 5. 启动开发服务器
uvicorn app.main:app --reload
```

## API

| 方法 | 路径 | 说明 |
| ---- | ---- | ---- |
| GET | `/api/v1/health` | 健康检查，返回 `{"status": "ok"}` |
| POST | `/api/v1/documents` | 上传 PDF（multipart `file`），同步解析并返回统计 |
| GET | `/api/v1/documents/{doc_id}/structure` | 返回 Document→Chapter→Section→Paragraph 树 |
| GET | `/api/v1/documents/{doc_id}/chunks` | 返回全部检索 Chunk（含面包屑与回链段落 id） |
| GET | `/api/v1/paper-questions/{doc_id}?figure_id=` | 题目草稿总览：按 Figure 分组，含实验摘要与证据预览 |
| GET | `/api/v1/paper-questions/{doc_id}/{set_id}` | 单题集合详情：完整陈述（中英）与证据原文 |

审核前端：启动后访问 <http://127.0.0.1:8000/review>（论文 → Figure → 题组 → 陈述逐条审核）

Swagger 文档：启动后访问 <http://127.0.0.1:8000/docs>

## 路线图

- [x] Phase 0：项目初始化、健康检查接口
- [x] Phase 1：Hierarchical PDF Parser（TOC/字体/编号/空间四级标题检测 + 层级树 + 段落感知分块）
- [x] Phase 2：Hybrid RAG 检索层（embedding API/hash 双 provider + numpy 精确余弦 + BM25 + RRF，幂等索引）
- [x] Phase 3：Evidence Gate（L1 启发式词项覆盖门控 + 可选 LLM 复核，10 查询验证 precision=recall=1.0）
- [x] Phase 4：Paper Figure Semantic Reconstruction（文本证据 → 图表实验语义：三级门控 + 证据绑定结论 + 可选 LLM 归一化）
- [x] Phase 5：Question Blueprint（确定性出题蓝图：四题型严格门控 + Evidence Store 溯源 + 字面数值抽取）
- [x] MCQ Step 1：Statement Draft（每集 1 真 + ≤4 假：十类受控扰动、替换素材全部来自论文证据池、association 仅可被错误升级）
- [x] MCQ Step 2：中文翻译层（statement_zh，术语表+模板，英文原文逐字保留）+ Review 审核前端（/review，通过/拒绝/需改 + 证据回链）
- [x] DATA 陈述锚点引用：证据句原文（排版级清洗：细空格/断行连字符/面板标号）+ 图锚点；可选 LLM 对象抽取（verbatim-span 门控，--no-llm 可关）
- [x] 可选 LLM 整句中文翻译（确定性验证门：数值/图锚点/方向极性/基因名逐字校验，失败回退术语表翻译；glm-4-flash ~2s/句，95%+ 过率）
- [ ] PostgreSQL + pgvector 向量库接入（接口已对齐：vector(1024) + HNSW + tokens→text[]/GIN）
- [ ] RAG 检索 API 化
- [ ] LLM 出题（选择题生成）
- [ ] Reviewer 质量审核

### 已知限制（Phase 4）

- 仅支持英文论文模式；中文论文（图N/表N）当前不解析（按需扩展双语模式库）
- 语义恢复为槽位级（组别/变量/方向/显著性/关系），不做精确数值恢复与面板级重建
- 依赖 Phase 1 结构质量：IMRaD 标题未识别时章节类型回退 other，证据仍可收集但角色区分变弱

### 已知限制（Phase 1）

- 扫描版（图片型）PDF 不支持，会返回 422（OCR 不在本阶段范围）
- 双栏版式未做列检测，按单栏阅读序处理
- 纯字体加粗、无字号差、无编号且无书签的英文排版仅依赖 bold/空间信号
