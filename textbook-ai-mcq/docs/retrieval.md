# Hybrid Retrieval（Phase 2）— CLI 与产物格式

_chunks → index → dense + BM25 → RRF fusion → top-k（带 provenance）_ 的检索层。
本阶段只交付 service 层 + CLI；不含 reranker、LLM、MCQ 与 FastAPI 检索端点。

## 1. 前置

Phase 1 已产出 `data/chunks/{doc_id}/chunks.jsonl`（见 [parse_pdf.md](parse_pdf.md)）。

## 2. 建索引

```bash
# 使用 .env 配置的 embedding（默认智谱 embedding-3，需 EMBEDDING_API_KEY 或 LLM_API_KEY）
python scripts/build_index.py --doc-id physiology

# 离线确定性 provider（测试/无 key 时用；无真实语义，仅打通链路）
python scripts/build_index.py --doc-id physiology --embedder hash

# 参数
--artifacts-root PATH   # 产物根目录，默认 settings.ARTIFACTS_DIR（data）
--force                 # manifest 匹配时也强制重建（重新计费 embedding）
```

幂等性：`manifest.json` 记录 `config_hash`（检索配置的规范化哈希）+ `chunk_set_hash`
（按 chunk_id 排序后的集合哈希）+ embedder 身份。三者与磁盘上的一致且未 `--force` 时，
重复执行直接 `status=skipped`，不产生任何 API 调用。

## 3. 检索

```bash
python scripts/search.py "心动周期的概念" --doc-id physiology
python scripts/search.py "细胞膜" --doc-id physiology --mode sparse --top-k 3 --json

# 参数
--mode {hybrid,dense,sparse}   # 默认 hybrid
--top-k N                      # 最终保留条数（默认 5；每路先 over-fetch 20 再融合）
--json                         # 输出完整 RetrievalResult JSON
```

三种 mode：

| mode | dense 路余弦 | BM25 路 | 融合 |
| ---- | ---- | ---- | ---- |
| hybrid | ✔ | ✔ | RRF（k=60） |
| dense | ✔ | ✘ | — |
| sparse | ✘ | ✔ | — |

命中结果每条包含：`chunk_id / document_id / chunk_index / text / breadcrumb（章节路径）/
pages / char_count` + `dense_score / sparse_score / fused_score`（未出现在该路的为
`null`）+ `rank` + `sources`（命中来源 `["dense","sparse"]` 子集）。

## 4. 索引产物（`data/index/{doc_id}/`）

| 文件 | 格式 | 说明 |
| ---- | ---- | ---- |
| `manifest.json` | JSON | IndexManifest：embedder/model/dim、config_hash、chunk_set_hash、num_chunks、created_at |
| `records.jsonl` | JSONL | 每行 IndexedRecord：chunk 全量元数据 + 索引时补齐的 `document_id`/`chunk_index`（chunks.jsonl 行号，原编号保留）+ `tokens`（jieba 分词） |
| `embeddings.npy` | npy | float32 矩阵 `(n, dim)`，行序 = records 行序，行已 L2 归一化（点积即余弦） |
| `embeddings.ids.json` | JSON | 行号 → chunk_id 列表 |

> Phase 1 的 `Chunk` schema 不含 `document_id`/`chunk_index`，索引层负责补齐，
> `data/chunks/` 产物只读不写。

## 5. 环境变量（.env）

```
EMBEDDING_BASE_URL=https://open.bigmodel.cn/api/paas/v4
EMBEDDING_API_KEY=            # 为空时回退 LLM_API_KEY
EMBEDDING_MODEL=embedding-3   # 置为 hash 则用离线确定性 provider
EMBEDDING_DIM=1024
EMBEDDING_BATCH_SIZE=64
RETRIEVAL_DENSE_TOP_K=20      # dense 路 over-fetch
RETRIEVAL_SPARSE_TOP_K=20     # sparse 路 over-fetch
RETRIEVAL_RRF_K=60
RETRIEVAL_MIN_CHUNK_CHARS=10  # 建索引时过滤过短 chunk（前置页噪声）
```

## 6. 架构与迁移

```
app/services/retrieval/
├── indexer.py     # 编排：load_chunks → 过滤 → tokenize → embed → 落盘（幂等）
├── engine.py      # RetrievalEngine.load()/retrieve()，三路 mode + hydrate
├── embeddings.py  # Hash（离线确定性）/ Http（OpenAI 兼容，批+重试）双 provider
├── vector_store.py# NumpyVectorStore：精确余弦；接口对齐 pgvector 语义
├── bm25.py        # Bm25Index：rank-bm25 BM25Okapi + jieba tokens
├── fusion.py      # RRF / 加权归一融合（纯函数）
├── tokenizer.py   # 索引与检索共用分词
└── config.py      # RetrievalConfig（config_hash 进 manifest）
```

迁移到 PostgreSQL + pgvector 时的对应关系：

| 现在 | 未来 |
| ---- | ---- |
| `embeddings.npy` + 余弦 matvec | `vector(1024)` 列 + HNSW `vector_cosine_ops`，`ORDER BY embedding <=> q`，相似度 = 1 − 距离 |
| `records.jsonl` 的 `tokens` | `text[]` 列 + GIN（jieba 分词结果直接入列，规避 zhparser 依赖） |
| `Bm25Index`（内存） | 应用层保留即可，或换 SQL 端 RRF CTE（pgvector 官方 hybrid 示例结构） |

## 7. 测试

```bash
pytest                       # 全部离线（HTTP 走 httpx.MockTransport；向量走 hash provider）
pytest tests/test_engine.py  # e2e：合成 PDF → ingest → build_index → retrieve
```
