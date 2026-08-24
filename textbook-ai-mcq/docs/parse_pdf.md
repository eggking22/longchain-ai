# parse_pdf.py 使用说明

`scripts/parse_pdf.py` 是 Hierarchical PDF Parser 的命令行入口：
把一本教材 PDF 解析为 `Document → Chapter → Section → Paragraph` 结构树，
并生成用于后续检索的 Chunk。

本文档说明它的调用方式、全部参数、产物存储位置与存储格式。

---

## 1. 调用方式

```bash
# 必须在项目根目录 textbook-ai-mcq/ 下执行
cd textbook-ai-mcq

# 使用 bio-ai 环境的 Python（Scripts 目录不在 PATH 时用 -m 方式）
python scripts/parse_pdf.py uploads/example.pdf --doc-id bio-grade1

# 等价写法（显式指定解释器路径）
/c/Users/Lenovo/anaconda3/envs/bio-ai/python.exe scripts/parse_pdf.py uploads/example.pdf --doc-id bio-grade1
```

前置条件：

- 使用 bio-ai conda 环境（Python 3.11，已装 PyMuPDF 1.28.2）
- 工作目录必须是项目根目录（`pytest.ini` 的 `pythonpath` 与相对路径 `uploads/`、`data/` 都以此为基准）
- 输入必须是文本型 PDF；扫描版（图片型）PDF 会直接报错退出，本阶段不支持 OCR

执行成功后终端会先打印统计摘要（JSON），再打印整棵结构树：

```
============================================================
Document: bio-grade1  (uploads\example.pdf)
{
  "doc_id": "bio-grade1",
  "num_pages": 3,
  "raw_lines": 22,
  "cleaned_lines": 16,
  "chapters": 2,
  "sections": 4,
  "paragraphs": 5,
  "heading_rules": { "toc": 5, "font": 1 },
  "toc_entries": 5,
  "chunks": { "count": 4, "min_chars": 46, "max_chars": 91, "avg_chars": 67.5 }
}
============================================================
[document] bio-grade1
  [chapter] 第1章 走近细胞  (rule=toc, conf=0.95)
    [section] 第1节 细胞是生命活动的基本单位  (rule=toc, conf=0.95)
      [paragraph] 细胞是生物体结构和功能的基本单位，除病毒外…  p[1]
      ...
```

退出码：成功 `0`；PDF 文件不存在或解析失败 `1`；扫描版 PDF 抛
`ScannedPdfError`（错误信息中会说明原因）。

---

## 2. 参数详解

| 参数 | 必填 | 默认值 | 说明 |
| ---- | ---- | ------ | ---- |
| `pdf`（位置参数） | 否 | `uploads/example.pdf` | 待解析的 PDF 路径，相对或绝对均可。文件不存在时报错退出（code 1） |
| `--doc-id` | 否 | PDF 文件名去扩展名（如 `example.pdf` → `example`） | 文档标识。决定产物存放目录名（`data/*/{doc_id}/`）以及结构树根节点标题；重复执行同一 doc-id 会直接覆盖旧产物 |
| `--artifacts-root` | 否 | `settings.ARTIFACTS_DIR`（默认 `data`，可被 `.env` 覆盖） | 产物根目录。传绝对路径可把产物放到项目外（如 `--artifacts-root D:/tmp/parser-out`） |
| `--reuse-raw` | 否 | 关闭 | 复用阶段 1 已落盘的 raw 产物（`lines.json`/`toc.json`/`fonts.json`），跳过重新读 PDF，直接从清洗阶段重跑。调阈值时免重复解析大文件 |

影响解析行为的环境变量（写在 `.env`，模板见 `.env.example`，全部可在运行时覆盖默认值）：

| 环境变量 | 默认 | 作用 |
| -------- | ---- | ---- |
| `PARSER_MAX_HEADING_LEVELS` | 4 | 标题层级深度上限（类似 pymupdf4llm 的 `max_levels`） |
| `PARSER_HEADER_FOOTER_BAND` | 0.09 | 页面顶部/底部多大比例算页眉页脚带 |
| `PARSER_REPEAT_RATIO` | 0.30 | 带内文本在多少比例页面重复才整组剔除 |
| `PARSER_CHUNK_TARGET_CHARS` | 600 | Chunk 目标字符数（多个段落聚合的软上限） |
| `PARSER_CHUNK_MAX_CHARS` | 1200 | Chunk 硬上限（超长段落按句切分的依据） |
| `PARSER_CHUNK_OVERLAP_SENTENCES` | 1 | 切分时相邻块的句级重叠数 |
| `UPLOADS_DIR` / `ARTIFACTS_DIR` | `uploads` / `data` | 上传目录与产物根目录 |

---

## 3. 产物存储位置

默认全部落在项目根目录的 `data/` 下（`data/` 与 `uploads/` 已被
`.gitignore` 忽略内容、只保留目录骨架）：

```
textbook-ai-mcq/
└── data/                                  ← 产物根目录（--artifacts-root 可改）
    ├── raw/{doc_id}/                      ← 阶段1+2：原始与清洗产物
    │   ├── meta.json                      PDF 路径、页数
    │   ├── lines.json                     阅读序原始行（含 span/字体/字号/bbox）
    │   ├── toc.json                       PDF 书签条目（级别/标题/页码）
    │   ├── fonts.json                     字体使用直方图（"font@size" → 字符数）
    │   ├── lines.clean.json               清洗后的行（供阶段3输入/人工检查）
    │   └── cleaner.debug.json             剔除审计：页眉页脚组、页码、剔除样例
    ├── structure/{doc_id}/                ← 阶段3：结构树与调试报告
    │   ├── document.json                  完整层级树（核心产物）
    │   ├── headings.debug.json            每个标题的判定规则/置信度/证据
    │   └── stats.json                     统计摘要（与终端输出一致）
    └── chunks/{doc_id}/
        └── chunks.jsonl                   ← 阶段4：检索用 Chunk（每行一个 JSON）
```

同一 doc-id 重复执行会整体覆盖该 doc-id 下的旧产物；不同 doc-id 互不影响。

---

## 4. 存储形式与字段说明

所有文件均为 UTF-8 编码的 JSON / JSONL，可直接用任何语言或工具读取。

### 4.1 `document.json` — 结构树（DocNode）

树的根是 document，子节点按顺序为 chapter → section → paragraph，
每个节点字段：

| 字段 | 类型 | 说明 |
| ---- | ---- | ---- |
| `node_id` | str | 16 位十六进制稳定 id（doc_id + 层级路径哈希） |
| `node_type` | str | `document` / `chapter` / `section` / `paragraph` |
| `title` | str | 章节标题（paragraph 为空字符串） |
| `level` | int | document=0，chapter=1，section≥2 |
| `text` | str | 段落正文（仅 paragraph 非空） |
| `children` | list[DocNode] | 子节点 |
| `provenance` | obj / null | 溯源：`page_no`（1 起）+ `bbox`（PDF 坐标） |
| `heading_rule` | str / null | 判定规则：`toc` / `font` / `numbering` / `spatial` / `synthetic` / `fallback` |
| `heading_confidence` | float / null | 置信度 0–1 |
| `pages` | list[int] | 段落横跨的页码（跨页段落会有多页） |

真实示例（节选自 `example` 文档的 `data/structure/example/document.json`）：

```json
{
  "node_id": "fd229ff6422ec877",
  "node_type": "section",
  "title": "第1节 细胞是生命活动的基本单位",
  "level": 2,
  "text": "",
  "children": [ /* paragraph ... */ ],
  "provenance": { "page_no": 1, "bbox": [72.0, 136.0, 296.0, 152.8] },
  "heading_rule": "toc",
  "heading_confidence": 0.95,
  "pages": []
}
```

### 4.2 `chunks.jsonl` — 检索块（Chunk）

每行一个独立 JSON 对象（JSONL），字段：

| 字段 | 类型 | 说明 |
| ---- | ---- | ---- |
| `chunk_id` | str | 16 位十六进制 id |
| `text` | str | 块正文（多个段落按序拼接，绝不跨章节边界） |
| `breadcrumb` | list[str] | 所属层级路径，如 `["第1章 走近细胞", "第1节 细胞是生命活动的基本单位"]` |
| `pages` | list[int] | 来源页码 |
| `char_count` | int | 字符数 |
| `paragraph_ids` | list[str] | 回链 `document.json` 中的段落 `node_id`，出题阶段可据此引用原文 |

真实示例（`data/chunks/example/chunks.jsonl` 第 1 行）：

```json
{
  "chunk_id": "b915d22ea764e379",
  "text": "细胞是生物体结构和功能的基本单位，除病毒外，一切生物体都是由细胞构成的，病毒没有细胞结构，必须寄生在活的宿主细胞内才能生活和增殖。显微镜下的细胞形态多种多样，但都具有相似的基本结构。",
  "breadcrumb": ["第1章 走近细胞", "第1节 细胞是生命活动的基本单位"],
  "pages": [1],
  "char_count": 91,
  "paragraph_ids": ["fd36e05d762f8159", "d3602afe035964ec"]
}
```

### 4.3 `headings.debug.json` — 标题判定报告

两部分：`accepted`（被采纳的标题，含 `line_index`/`text`/`level`/
`rule`/`confidence`/`evidence`）与 `trace`（每个候选行的全部命中规则，
含被拒绝的）。排查"某行为什么没被识别为标题/为什么被识别"看这个文件。

### 4.4 `stats.json` — 统计摘要

与终端输出的 JSON 完全一致：页数、行数（原始/清洗后）、章/节/段落计数、
标题规则分布（`toc`/`font`/`numbering`…各命中几个）、chunk 数量与字符数
min/max/avg。

---

## 5. 与 API 的关系

CLI 与 `POST /api/v1/documents` 走完全相同的流水线（同一个
`ingest()`）：

| CLI | API |
| --- | --- |
| `python scripts/parse_pdf.py uploads/x.pdf --doc-id X` | `curl -F "file=@uploads/x.pdf" http://127.0.0.1:8000/api/v1/documents`（doc-id 由服务端生成） |
| 终端打印 stats | 响应体返回 stats |
| `data/structure/{doc_id}/document.json` | `GET /api/v1/documents/{doc_id}/structure` |
| `data/chunks/{doc_id}/chunks.jsonl` | `GET /api/v1/documents/{doc_id}/chunks` |

区别：API 版本会把上传文件先存到 `UPLOADS_DIR`，doc_id 自动生成；
CLI 版本允许自定义 doc-id、覆盖产物目录、断点重跑，适合批量预处理和调试。

---

## 6. 常见用法示例

```bash
# 基本用法
python scripts/parse_pdf.py uploads/example.pdf --doc-id bio-grade1

# 只调 chunk 参数：复用已解析的 raw，秒级重跑
python scripts/parse_pdf.py uploads/example.pdf --doc-id bio-grade1 --reuse-raw

# 产物放到项目外
python scripts/parse_pdf.py uploads/example.pdf --doc-id test1 --artifacts-root D:/tmp/parser-out

# 调整 chunk 尺寸后重跑（改 .env 或临时环境变量）
PARSER_CHUNK_TARGET_CHARS=400 python scripts/parse_pdf.py uploads/example.pdf --doc-id bio-grade1 --reuse-raw
```
