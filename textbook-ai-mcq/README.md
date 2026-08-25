# textbook-ai-mcq

基于生物学教材的长文本选择题生成系统（Long-text MCQ generation from biology textbooks）。

当前处于 **Phase 2：Hybrid RAG 检索层（已完成）**——
Phase 1 的 Chunks → 幂等索引 → dense（余弦）+ sparse（BM25）双路 → RRF 融合 →
带完整 provenance 的 Top-K 结果。embedding 走 OpenAI 兼容 API（默认智谱 embedding-3），
测试/离线场景用确定性 hash provider；不含 reranker、LLM、MCQ（后续阶段）。

此前 **Phase 1：可控的 Hierarchical PDF Parser（已完成）**——
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

## 目录结构

```
textbook-ai-mcq/
├── app/
│   ├── main.py              # FastAPI 入口
│   ├── core/                # 配置（pydantic-settings）
│   ├── db/                  # 数据库（后续阶段）
│   ├── models/              # ORM（后续阶段）
│   ├── schemas/
│   │   └── document.py      # DocNode 树 / Chunk / Provenance
│   ├── routers/
│   │   ├── health.py        # GET /api/v1/health
│   │   └── documents.py     # 上传/结构/chunks 接口
│   ├── services/
│   │   ├── parser/           # Phase 1：解析流水线（parser/cleaner/structure/chunking/…）
│   │   └── retrieval/        # Phase 2：混合检索（indexer/engine/embeddings/vector_store/bm25/fusion/tokenizer）
│   ├── llm/                 # LLM 封装（后续阶段）
│   └── utils/
├── data/                    # 解析与索引产物（raw/structure/chunks/index，均 gitignored）
├── uploads/                 # 上传的 PDF
├── docs/                    # 说明文档（parse_pdf.md、retrieval.md）
├── prompts/                 # 提示词模板（后续阶段）
├── scripts/parse_pdf.py     # Phase 1 解析 CLI
├── scripts/build_index.py   # Phase 2 建索引 CLI
├── scripts/search.py        # Phase 2 检索 CLI
├── tests/                   # 合成 PDF 单测 + e2e golden + API 测试
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

Swagger 文档：启动后访问 <http://127.0.0.1:8000/docs>

## 路线图

- [x] Phase 0：项目初始化、健康检查接口
- [x] Phase 1：Hierarchical PDF Parser（TOC/字体/编号/空间四级标题检测 + 层级树 + 段落感知分块）
- [x] Phase 2：Hybrid RAG 检索层（embedding API/hash 双 provider + numpy 精确余弦 + BM25 + RRF，幂等索引）
- [ ] PostgreSQL + pgvector 向量库接入（接口已对齐：vector(1024) + HNSW + tokens→text[]/GIN）
- [ ] RAG 检索 API 化与 Evidence Gate
- [ ] LLM 出题（选择题生成）
- [ ] Reviewer 质量审核

### 已知限制（Phase 1）

- 扫描版（图片型）PDF 不支持，会返回 422（OCR 不在本阶段范围）
- 双栏版式未做列检测，按单栏阅读序处理
- 纯字体加粗、无字号差、无编号且无书签的英文排版仅依赖 bold/空间信号
