"""Golden evidence-gate suite: 5 sufficient + 5 insufficient queries.

Deterministic end-to-end: a synthetic physiology mini-corpus is chunked,
indexed (offline hash embedder), retrieved and gated offline. The suite
mirrors the Phase 3 acceptance criteria — 5 correct passes, 5 correct
rejections, precision = recall = 1.0.

A second, opt-in test runs the same 10 queries against the real
physiology index (data/index/physiology) when it exists locally.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.schemas.document import Chunk
from app.services.evidence.config import EvidenceConfig
from app.services.evidence.evaluator import HeuristicEvidenceEvaluator
from app.services.evidence.gate import EvidenceGate
from app.services.retrieval.config import RetrievalConfig
from app.services.retrieval.embeddings import HashEmbeddingProvider
from app.services.retrieval.engine import RetrievalEngine
from app.services.retrieval.indexer import build_index

DOC_ID = "physio-mini"

# --- synthetic corpus: controlled so the expected verdicts are unambiguous ---
CHUNK_TEXTS = [
    ("m0", "第四章 血液循环", "糖酵解的发生部位是细胞质基质。糖酵解将一分子葡萄糖分解为两分子丙酮酸，此过程不需要氧气，释放少量能量。"),
    ("m1", "第四章 血液循环", "心脏的一次收缩和舒张构成的一个机械活动周期，称为心动周期。心率加快时，心动周期缩短，以舒张期缩短为主。"),
    ("m2", "第四章 血液循环", "足够的血液充盈量是形成动脉血压的前提，心室射血和外周阻力是形成动脉血压的两个基本因素。"),
    ("m3", "第五章 呼吸", "血液中绝大部分氧气与红细胞内的血红蛋白结合，以氧合血红蛋白的形式运输到全身各处组织。"),
    ("m4", "第三章 血液", "红细胞的主要功能是运输氧气和二氧化碳。红细胞呈双凹圆盘状，成熟的红细胞没有细胞核。"),
    ("m5", "第四章 血液循环", "心室收缩将血液射入动脉，心室舒张时血液由静脉返回心脏，如此循环往复，实现心脏的泵血功能。"),
    ("m6", "第三章 血液", "血红蛋白是一种含铁的蛋白质，容易与氧气结合，也容易与氧气分离。"),
    ("m7", "前言", "本书由人民卫生出版社出版，供基础医学类专业教学使用。"),
]

SUFFICIENT_QUERIES = [
    "糖酵解的发生部位是哪里？",
    "心动周期由什么活动构成？",
    "动脉血压是如何形成的？",
    "氧气在血液中怎样运输？",
    "红细胞的主要功能是什么？",
]

INSUFFICIENT_QUERIES = [
    "2025年诺贝尔生理学奖对糖酵解研究有什么影响？",
    "CRISPR基因编辑技术在临床应用中的最新进展？",
    "互联网医院慢病管理模式的效果如何？",
    "COVID-19对呼吸系统的影响机制是什么？",
    "人工智能在心电图诊断中的应用现状如何？",
]

# tokens that MUST be flagged as missing when the query is rejected
SIGNATURE_MISSING = {
    INSUFFICIENT_QUERIES[0]: {"诺贝尔"},
    INSUFFICIENT_QUERIES[1]: {"crispr"},
    INSUFFICIENT_QUERIES[2]: {"互联网", "慢病"},
    INSUFFICIENT_QUERIES[3]: {"covid"},
    INSUFFICIENT_QUERIES[4]: {"人工智能"},
}

PHYSIO_INDEX = Path("data/index/physiology/manifest.json")


def _write_corpus(root: Path) -> None:
    d = root / "chunks" / DOC_ID
    d.mkdir(parents=True)
    with (d / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for i, (chunk_id, crumb, text) in enumerate(CHUNK_TEXTS):
            chunk = Chunk(
                chunk_id=chunk_id,
                text=text,
                breadcrumb=[crumb],
                pages=[i + 1],
                char_count=len(text),
                paragraph_ids=[f"p{i}"],
            )
            f.write(chunk.model_dump_json() + "\n")


@pytest.fixture(scope="module")
def engine_and_gate(tmp_path_factory):
    root = tmp_path_factory.mktemp("evidence")
    _write_corpus(root)
    stats = build_index(
        DOC_ID, root, embedder=HashEmbeddingProvider(dim=16),
        config=RetrievalConfig(min_chunk_chars=0),
    )
    assert stats["status"] == "built"
    engine = RetrievalEngine.load(DOC_ID, root)
    gate = EvidenceGate(HeuristicEvidenceEvaluator(EvidenceConfig()))
    return engine, gate


def _run_suite(engine: RetrievalEngine, gate: EvidenceGate):
    """Run all 10 queries; return (rows, precision, recall).

    Retrieval over-fetches (top_k=10) so the gate's coverage pool (8) is
    deeper than its cited evidence window (5).
    """
    rows = []
    tp = fp = fn = tn = 0
    for query in SUFFICIENT_QUERIES:
        result = engine.retrieve(query, mode="hybrid", top_k=10)
        report = gate.evaluate(query, result)
        if report.sufficient:
            tp += 1
        else:
            fn += 1
        rows.append((query, True, report))
    for query in INSUFFICIENT_QUERIES:
        result = engine.retrieve(query, mode="hybrid", top_k=10)
        report = gate.evaluate(query, result)
        if report.sufficient:
            fp += 1
        else:
            tn += 1
        rows.append((query, False, report))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return rows, precision, recall, (tp, fp, fn, tn)


def _print_table(rows) -> None:
    print(f"\n{'query':<38} {'expect':<6} {'suff':<6} {'score':<7} missing")
    for query, expected, report in rows:
        print(
            f"{query:<38} {str(expected):<6} {str(report.sufficient):<6} "
            f"{report.coverage_score:<7} {report.missing_information[:4]}"
        )


def test_synthetic_golden_suite(engine_and_gate):
    engine, gate = engine_and_gate
    rows, precision, recall, (tp, fp, fn, tn) = _run_suite(engine, gate)
    _print_table(rows)
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}  precision={precision} recall={recall}")
    # acceptance: 5 correct passes + 5 correct rejections
    assert (tp, fp, fn, tn) == (5, 0, 0, 5), "gate misclassified a query (see table above)"
    assert precision == 1.0 and recall == 1.0

    # every rejected query flags its signature missing information
    for query, expected, report in rows:
        if not expected:
            flagged = " ".join(report.missing_information).lower()
            for token in SIGNATURE_MISSING[query]:
                assert token.lower() in flagged, f"{query!r} should flag missing {token!r}"


@pytest.mark.skipif(not PHYSIO_INDEX.exists(), reason="physiology index not built (run scripts/build_index.py --doc-id physiology)")
def test_physiology_real_corpus():
    engine = RetrievalEngine.load("physiology", "data")
    gate = EvidenceGate(HeuristicEvidenceEvaluator(EvidenceConfig()))
    rows, precision, recall, (tp, fp, fn, tn) = _run_suite(engine, gate)
    _print_table(rows)
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}  precision={precision} recall={recall}")
    assert (tp, fp, fn, tn) == (5, 0, 0, 5), "gate misclassified a physiology query (see table above)"
