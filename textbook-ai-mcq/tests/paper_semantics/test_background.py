"""Background extraction: Abstract dual-path detection + Introduction summary."""

from __future__ import annotations

from app.services.paper_semantics import extract_background
from app.services.paper_semantics.sections import ParagraphRecord


def _pr(pid, text, section="other", page=1, order=0):
    return ParagraphRecord(paragraph_id=pid, text=text, section_type=section, page_no=page, order=order)


LONG_INTRO_SENTENCES = [
    "Dendritic cells patrol peripheral tissues and sample antigens continuously.",
    "Upon encountering danger signals they undergo a maturation program.",
    "CCR7 expression licenses migration toward lymph node chemokines.",
    "The mechanosensing machinery linking cell shape to gene expression remains unclear.",
    "Here we show that confinement at the scale of the nucleus is sufficient.",
]


class TestAbstractHeadingPath:
    def test_abstract_section_detected(self):
        paragraphs = [
            _pr("a1", "Dendritic cells migrate to lymph nodes upon activation.", section="abstract"),
            _pr("a2", "We show that cell shape sensing controls this process.", section="abstract"),
            _pr("i1", "Immune cells constantly survey peripheral tissues.", section="introduction"),
        ]
        background = extract_background(paragraphs)
        assert background["abstract"]["source"] == "abstract_heading"
        assert "Dendritic cells migrate" in background["abstract"]["text"]
        assert background["abstract"]["paragraph_ids"] == ["a1", "a2"]
        assert background["abstract"]["pages"] == [1]


class TestLeadingParagraphsPath:
    def test_nature_style_no_abstract_heading(self):
        abstract_text = (
            "Dendritic cells patrol peripheral tissues and migrate toward draining lymph nodes "
            "when they receive activation signals, a journey that requires rapid transcriptional "
            "reprogramming of chemokine receptors and motility genes in response to cues from the "
            "tissue microenvironment, yet the upstream sensory events remain poorly understood."
        )
        paragraphs = [
            _pr("t1", "Cell shape sensing licenses dendritic cells", order=0),  # title (short)
            _pr("auth", "Zahraa Alraies, Claudia Rivera and Maria-Graciela Delgado", order=1),  # authors (short)
            _pr("lead1", abstract_text, order=2),
            _pr("narr1", "The immune system constantly surveys barrier tissues.", order=3),  # intro narrative
            _pr("r1", "Confinement increased CCR7 expression (Fig. 1).", section="results", order=4),
        ]
        background = extract_background(paragraphs)
        assert background["abstract"]["source"] == "leading_paragraphs"
        assert background["abstract"]["paragraph_ids"] == ["lead1"]  # longest leading paragraph
        assert "Dendritic cells patrol" in background["abstract"]["text"]

    def test_word_budget_truncates_at_sentence_boundary(self):
        long_text = " ".join(
            f"Sentence number {i} describes an experimental finding about dendritic cell migration." for i in range(60)
        )
        paragraphs = [_pr("lead1", long_text), _pr("r1", "Results start here (Fig. 1).", section="results")]
        background = extract_background(paragraphs)
        words = len(background["abstract"]["text"].split())
        assert words <= 420  # ~400 word budget, never mid-sentence
        assert background["abstract"]["text"].endswith(".")

    def test_captions_and_short_lines_excluded(self):
        paragraphs = [
            _pr("t1", "Cell shape sensing", order=0),
            _pr("cap", "Figure 1. Overview of the experimental system used throughout this work.", order=1),
            _pr(
                "lead1",
                "Dendritic cells migrate toward draining lymph nodes upon activation signals and "
                "must reprogram their transcriptional profile to complete the journey successfully.",
                order=2,
            ),
            _pr("r1", "Results start here (Fig. 1).", section="results"),
        ]
        background = extract_background(paragraphs)
        assert background["abstract"]["paragraph_ids"] == ["lead1"]

    def test_reporting_summary_not_mistaken_for_abstract(self):
        """Nature papers end with a 'Reporting summary' section — never the Abstract."""
        paragraphs = [
            _pr(
                "lead1",
                "Dendritic cells migrate toward lymph nodes when activated and need transcriptional "
                "reprogramming to complete their journey through tissues toward the draining nodes.",
                order=0,
            ),
            _pr("rs", "Further information on research design is available in the Reporting Summary.", order=5),
            _pr("r1", "Confinement increased CCR7 expression (Fig. 1).", section="results", order=6),
        ]
        background = extract_background(paragraphs)
        assert "Reporting Summary" not in background["abstract"]["text"]


class TestIntroductionFallback:
    def test_narrative_before_first_anchor_is_introduction(self):
        abstract_text = (
            "Dendritic cells patrol peripheral tissues and migrate toward draining lymph nodes "
            "when they receive activation signals that reprogram chemokine receptor expression "
            "and endow the cells with intrinsic motility during homeostatic immune surveillance."
        )
        paragraphs = [
            _pr("lead1", abstract_text, order=0),
            _pr("n1", "Tissue-resident dendritic cells continuously sample environmental antigens in peripheral organs.", order=1),
            _pr("n2", "Upon danger recognition they upregulate CCR7 and migrate toward draining lymph nodes rapidly.", order=2),
            _pr("r1", "Confinement increased CCR7 expression (Fig. 1).", section="results", order=3),
        ]
        background = extract_background(paragraphs)
        assert background["introduction"]["paragraph_ids"] == ["n1", "n2"]
        assert "sample environmental antigens" in background["introduction"]["text"]
        assert "lead1" not in background["introduction"]["paragraph_ids"]  # abstract excluded


class TestIntroductionSummary:
    def test_introduction_concatenated_with_budget(self):
        paragraphs = [
            _pr("i1", " ".join(LONG_INTRO_SENTENCES[:2]), section="introduction", page=1),
            _pr("i2", " ".join(LONG_INTRO_SENTENCES[2:]), section="introduction", page=2),
        ]
        background = extract_background(paragraphs)
        assert len(background["introduction"]["text"]) <= 1500
        assert background["introduction"]["text"].endswith(".")
        assert background["introduction"]["paragraph_ids"] == ["i1", "i2"]
        assert background["introduction"]["pages"] == [1, 2]

    def test_no_introduction_yields_empty(self):
        background = extract_background([_pr("r1", "Results text only.", section="results")])
        assert background["introduction"] == {}

    def test_never_rewrites_author_text(self):
        paragraphs = [_pr("i1", "The authors wrote exactly this sentence.", section="introduction")]
        background = extract_background(paragraphs)
        assert "The authors wrote exactly this sentence." in background["introduction"]["text"]
