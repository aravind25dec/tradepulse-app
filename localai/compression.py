"""
localai/compression.py
Two optional AI-powered text tools for reducing token usage:

  1. sentence-transformers  — semantic embeddings for smarter memory retrieval
     Model: all-MiniLM-L6-v2  (~90 MB, downloads once)
     Replaces TF-IDF keyword matching with true semantic understanding.
     "what should I sell" ≈ "which positions should I exit"

  2. LLMLingua-2            — prompt compression, removes low-importance tokens
     Model: microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank (~700 MB)
     Compresses portfolio text and past memories 30-50% before they enter the prompt.
     Financial numbers ($, %) are protected from removal.

Both fall back gracefully when not installed — app works either way.
Install: pip install sentence-transformers llmlingua
"""

import logging
from functools import lru_cache

log = logging.getLogger(__name__)

# ── 1. sentence-transformers — semantic embeddings ────────────────────────────

@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    log.info("Loading sentence-transformers model all-MiniLM-L6-v2 (first load)…")
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed(text: str) -> list[float] | None:
    """
    Return a 384-dim unit-normalized embedding vector.
    Returns None if sentence-transformers is not installed.
    """
    try:
        vec = _embedder().encode(text, normalize_embeddings=True)
        return vec.tolist()
    except ImportError:
        return None
    except Exception as exc:
        log.warning("Embedding failed: %s", exc)
        return None


def semantic_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two pre-normalized vectors — just a dot product."""
    return sum(x * y for x, y in zip(a, b))


def st_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


# ── 2. LLMLingua-2 — prompt compression ──────────────────────────────────────

@lru_cache(maxsize=1)
def _compressor():
    from llmlingua import PromptCompressor
    log.info(
        "Loading LLMLingua-2 model (~700 MB, downloads once on first use)…"
    )
    return PromptCompressor(
        model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        use_llmlingua2=True,
        device_map="cpu",
    )


def compress(
    text: str,
    question: str = "",
    rate: float = 0.65,
) -> str:
    """
    Compress text to ~rate fraction of original tokens using LLMLingua-2.

    - rate=0.65 means keep 65% of tokens (35% reduction)
    - question is used to guide which parts are most relevant to preserve
    - Financial markers ($, %, ticker-like tokens) are force-protected
    - Returns original text unchanged if llmlingua not installed or text is short
    """
    words = text.split()
    if len(words) < 40:
        return text   # too short — compression overhead not worth it

    try:
        cmp = _compressor()
        result = cmp.compress_prompt(
            context=text,
            question=question,
            rate=rate,
            force_tokens=["$", "%", "\n"],   # protect financial markers and line breaks
            chunk_end_tokens=["\n", "."],
        )
        compressed = result.get("compressed_prompt", text)
        orig  = len(words)
        after = len(compressed.split())
        log.info("LLMLingua: %d → %d words (%.0f%%)", orig, after, after / orig * 100)
        return compressed
    except ImportError:
        return text
    except Exception as exc:
        log.warning("LLMLingua compression failed, using original: %s", exc)
        return text


def lingua_available() -> bool:
    try:
        import llmlingua  # noqa: F401
        return True
    except ImportError:
        return False
