# textbook-ai-mcq

基于生物学教材的长文本选择题生成系统（Long-text MCQ generation from biology textbooks）。

当前处于 **Phase 1：可控的 Hierarchical PDF Parser（已完成）**——
PDF → 结构化教材树（Document→Chapter→Section→Paragraph）→ 检索用 Chunk。
不包含 embedding、向量库、RAG、LLM 出题（后续阶段）。

## 技术栈

- Python 3.11（conda 环境 `bio-ai`，禁止新建虚拟环境）
- FastAPI + Pydantic
- SQLAlchemy（PostgreSQL / pgvector 接口已预留，暂未连接）
- PyMuPDF（PDF 解析核心，自研层级解析，未引入 Docling/pymupdf4llm 依赖）
- python-docx（Word 导出，后续阶段使用）
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
│   │   └── parser/
│   │       ├── parser.py    # 阶段1：原始抽取
│   │       ├── cleaner.py   # 阶段2：清洗
│   │       ├── structure.py # 阶段3：标题检测+层级+段落
│   │       ├── chunking.py  # 阶段4：分块
│   │       ├── ingestion.py # 编排与产物落盘
│   │       ├── patterns.py  # 中英编号正则库
│   │       └── config.py    # ParserConfig 全部阈值
│   ├── llm/                 # LLM 封装（后续阶段）
│   └── utils/
├── data/                    # 解析产物（raw/structure/chunks）
├── uploads/                 # 上传的 PDF
├── docs/                    # 说明文档（如 parse_pdf.md：CLI 参数与产物格式）
├── prompts/                 # 提示词模板（后续阶段）
├── scripts/parse_pdf.py     # 解析 CLI
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
- [ ] 文本切分优化与 embedding
- [ ] PostgreSQL + pgvector 向量库接入
- [ ] RAG 检索
- [ ] LLM 出题（选择题生成）
- [ ] Reviewer 质量审核

### 已知限制（Phase 1）

- 扫描版（图片型）PDF 不支持，会返回 422（OCR 不在本阶段范围）
- 双栏版式未做列检测，按单栏阅读序处理
- 纯字体加粗、无字号差、无编号且无书签的英文排版仅依赖 bold/空间信号
