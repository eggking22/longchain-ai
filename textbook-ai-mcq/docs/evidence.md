# Evidence Gate（Phase 3）— 设计与用法

判断"**当前检索证据是否足以支持回答该问题/生成题目**"。检索不再无条件交给
LLM，而是先过门控：证据充分才放行，否则给出 `INSUFFICIENT_EVIDENCE` 与缺失
信息清单。

```
Query → Hybrid Retrieval (Phase 2)
      → Evidence Evaluation（Level 1 启发式 → 可选 Level 2 LLM 复核）
      → Evidence Gate（coverage_score >= threshold ?）
      → CoverageReport / InsufficientEvidenceError
```

## 1. CLI

```bash
python scripts/evidence_gate.py "糖酵解的发生部位是哪里？" --doc-id physiology
#   → verdict: SUFFICIENT ✔  coverage=0.875 …（exit code 0）
python scripts/evidence_gate.py "2025年诺贝尔生理学奖对糖酵解研究的影响" --doc-id physiology
#   → INSUFFICIENT_EVIDENCE ✘ missing: [2025, 诺贝尔, …]（exit code 2）
python scripts/evidence_gate.py "…" --doc-id physiology --json   # 完整 CoverageReport JSON
```

## 2. 输出契约（CoverageReport）

```json
{
  "query": "糖酵解的发生部位是哪里？",
  "sufficient": true,
  "coverage_score": 0.875,
  "threshold": 0.8,
  "evidence": [
    {"chunk_id": "…", "score": 0.0164, "rank": 1, "document_id": "physiology",
     "breadcrumb": ["第四章 血液循环", "…"], "pages": [18], "text_preview": "…"}
  ],
  "missing_information": [],
  "level": "heuristic",
  "detail": {"union_coverage": 1.0, "focus_coverage": 0.6667, "…": "…"}
}
```

## 3. 两级评估

**Level 1 — HeuristicEvidenceEvaluator**（零成本、确定性、provider 无关）：

```
coverage_score = 0.75 × term_signal + 0.15 × sparse_strength + 0.10 × hit_depth

term_signal    = 0.5 × union_coverage + 0.5 × focus_coverage
  union_coverage：查询实质词（去停用词/单字符，含同义词等价类 O2↔氧气 等）
                  在覆盖池（top-8）全部命中词的并集占比
  focus_coverage：池内**单个最好 chunk** 对查询词的覆盖率——惩罚"散落命中"
                  （如"人工智能"在序言、"心电图"在心电图章拼出的假覆盖）
sparse_strength= min(1, 池内最高 BM25 分 / 12)   —— 饱和归一，避免裸 BM25 阈值
hit_depth      = min(1, BM25 ≥ 2 的命中数 / 3)
sufficient ⟺ coverage_score ≥ 0.80（全部参数可 .env 覆盖）
missing_information = 未被证据覆盖的查询实质词（如 ["2025", "诺贝尔"]）
```

刻意不使用 dense 分数：当前索引为离线 hash provider（无语义），词项覆盖与
BM25 信号对 embedding provider 无关、可复现。接入真实 embedding 后检索质量
本身提升，公式不变。

**Level 2 — LlmEvidenceEvaluator**（可选，OpenAI 兼容 chat/completions，httpx 直调）：

- 激活条件：`.env` 配置了 `LLM_API_KEY` + `LLM_MODEL`（未配置 ⇒ 纯 Level 1）。
- 触发时机（CRAG 便宜门控思想）：**仅当 L1 未通过时**才调 LLM——L1 明确通过
  不花 token；L1 拒绝的查询由 LLM 语义复核，可挽救词形不匹配的改写型查询
  （"呼吸作用/细胞呼吸"），这正是词项覆盖的已知盲区。
- 评审模式（RAGAS context-recall 分解法，对抗 LLM 评审系统性偏宽）：要求
  LLM 先把问题拆成"回答所需的必要事实点"，逐条对照证据，再产出严格 JSON
  `{"sufficient", "coverage_score", "missing_information", "reasoning"}`；
  措辞要求"只有证据包含全部关键信息才 sufficient=true，不确定判 false"。
- LLM 结论权威（sufficient/coverage_score/missing_information），L1 诊断保留
  在 `detail.heuristic`；LLM 调用/解析失败 ⇒ 自动降级用 L1 结论并记录
  `detail.llm_error`。

## 4. Gate 语义

- `gate.evaluate(query, retrieval) -> CoverageReport`：永不抛异常。
- `gate.require(query, retrieval)`：insufficient 时抛 `InsufficientEvidenceError`
  （`.report` 携带完整报告）——未来 MCQ Generator 必须走此入口，从类型系统上
  保证"证据不足不能强行生成"。

## 5. 环境变量

```
EVIDENCE_COVERAGE_THRESHOLD=0.8    # 通过线
EVIDENCE_TERM_WEIGHT=0.75
EVIDENCE_SPARSE_WEIGHT=0.15
EVIDENCE_HIT_WEIGHT=0.1            # 三权重和必须为 1
EVIDENCE_SPARSE_SATURATE=12.0
EVIDENCE_HIT_FLOOR=2.0
EVIDENCE_WINDOW=5                  # 报告引用的证据条数
EVIDENCE_COVERAGE_POOL=8           # 词项覆盖检查池（≥ window）
# Level 2：复用 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
```

## 6. 验证集（tests/test_evidence_suite.py）

5 个 sufficient + 5 个 insufficient 查询；两个语料均 **10/10，precision=recall=1.0**：

- 合成迷你语料（确定性金标准，CI 必跑）
- physiology 真实教材索引（存在 `data/index/physiology` 时自动运行，否则跳过）

| 类型 | 查询示例 | 期望 |
| ---- | ---- | ---- |
| sufficient | 糖酵解的发生部位是哪里？ | 通过 |
| sufficient | 心动周期由什么活动构成？ | 通过 |
| insufficient | 2025年诺贝尔生理学奖对糖酵解研究有什么影响？ | 拒绝（missing: 2025/诺贝尔） |
| insufficient | 互联网医院慢病管理模式的效果如何？ | 拒绝（missing: 互联网/慢病） |

## 7. 设计依据（开源/文献调研）

- **门控前置**：Google "Sufficient Context"（ICLR'25）——LLM 在证据不足时倾向
  回退参数记忆而非弃答，必须在生成前判定。
- **词项覆盖为 L1 主信号**：ES `minimum_should_match` 的类比；有界、可解释、
  直接产出 missing_information；裸 BM25 绝对阈值随 IDF/查询长度漂移（业界共
  识），故仅做饱和归一分量。
- **CRAG 分带**：便宜信号先判，昂贵 LLM 只复核必要样本。
- **RAGAS context-recall 分解**：先拆事实点再逐条核对，缓解 LLM 评审偏宽
  （CRAG 实测 prompted-LLM 评审 58-65% vs 微调评估器 84%）。
- 已知局限：纯 L1 对同义改写偏保守（由 L2 救济）；`focus` 防散落假覆盖。
