"""Hierarchical PDF parser package.

Pipeline: parser.py (raw extraction) -> cleaner.py -> structure.py
(heading detection + hierarchy) -> chunking.py, orchestrated by
ingestion.py. Knobs live in config.py, numbering regexes in patterns.py.
"""

from .chunking import chunk_document
from .cleaner import clean_lines
from .config import ParserConfig
from .ingestion import ingest
from .parser import RawDoc, ScannedPdfError, extract_raw, load_raw, save_raw
from .structure import DetectionResult, HeadingCandidate, build_document, detect_headings

__all__ = [
    "RawDoc",
    "ScannedPdfError",
    "ParserConfig",
    "extract_raw",
    "save_raw",
    "load_raw",
    "clean_lines",
    "detect_headings",
    "DetectionResult",
    "HeadingCandidate",
    "build_document",
    "chunk_document",
    "ingest",
]
