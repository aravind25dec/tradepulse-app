"""
localai/memory.py
Stores past Q&A pairs in a local SQLite database.

Why SQLite instead of JSON:
  - Embeddings stored as float32 binary blobs: 1.5 KB/entry vs 3 KB in JSON
  - Vectorized numpy cosine similarity — faster than pure-Python loops
  - SQL filtering by ticker, date, model without loading everything
  - Concurrent-safe, survives partial writes, no full-file rewrite on every save
  - Auto-migrates existing memory.json entries on first run

Schema:
  memories(id, question, answer, model, tickers TEXT, ts TEXT, embedding BLOB)
"""

import json
import math
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_DB   = Path(__file__).parent / "memory.db"
_JSON = Path(__file__).parent / "memory.json"   # old file, migrated then left alone

MAX_KEEP = 1000   # rows kept; oldest pruned when exceeded


# ── Connection ────────────────────────────────────────────────────────────────

@contextmanager
def _conn():
    con = sqlite3.connect(_DB, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")   # safe concurrent reads + writes
    con.execute("PRAGMA synchronous=NORMAL")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def _init_db() -> None:
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT    NOT NULL,
                answer   TEXT    NOT NULL,
                model    TEXT    DEFAULT '',
                tickers  TEXT    DEFAULT '[]',
                ts       TEXT    NOT NULL,
                embedding BLOB
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_ts ON memories(ts)")

    _migrate_json()


def _migrate_json() -> None:
    """One-time import of legacy memory.json entries into SQLite."""
    if not _JSON.exists():
        return
    try:
        entries = json.loads(_JSON.read_text(encoding="utf-8"))
        if not entries:
            return
        import numpy as np
        with _conn() as con:
            existing = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            if existing > 0:
                return   # already migrated
            for e in entries:
                emb_blob = None
                if "embedding" in e:
                    emb_blob = np.array(e["embedding"], dtype=np.float32).tobytes()
                con.execute(
                    "INSERT INTO memories(question,answer,model,tickers,ts,embedding) VALUES(?,?,?,?,?,?)",
                    (
                        e.get("q", ""),
                        e.get("a", ""),
                        e.get("model", ""),
                        json.dumps(e.get("tickers", [])),
                        e.get("ts", datetime.now(timezone.utc).isoformat()),
                        emb_blob,
                    ),
                )
        _JSON.rename(_JSON.with_suffix(".json.bak"))   # keep backup, stop re-migrating
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("JSON migration failed: %s", exc)


# ── Public API ────────────────────────────────────────────────────────────────

def save(question: str, answer: str, model: str = "", tickers: list[str] | None = None) -> None:
    """Persist a Q&A pair. Embeds the question if sentence-transformers is available."""
    import numpy as np

    emb_blob = None
    try:
        from compression import embed
        emb = embed(question)
        if emb is not None:
            emb_blob = np.array(emb, dtype=np.float32).tobytes()
    except Exception:
        pass

    with _conn() as con:
        con.execute(
            "INSERT INTO memories(question,answer,model,tickers,ts,embedding) VALUES(?,?,?,?,?,?)",
            (
                question.strip(),
                answer.strip(),
                model,
                json.dumps(tickers or []),
                datetime.now(timezone.utc).isoformat(),
                emb_blob,
            ),
        )
        # Prune oldest rows beyond MAX_KEEP
        con.execute("""
            DELETE FROM memories WHERE id IN (
                SELECT id FROM memories ORDER BY id ASC
                LIMIT MAX(0, (SELECT COUNT(*) FROM memories) - ?)
            )
        """, (MAX_KEEP,))


def count() -> int:
    with _conn() as con:
        return con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]


def clear() -> None:
    with _conn() as con:
        con.execute("DELETE FROM memories")
        con.execute("VACUUM")


def all_entries(limit: int = 10) -> list[dict]:
    """Return the most recent entries (without embedding bytes) for display."""
    with _conn() as con:
        rows = con.execute(
            "SELECT question,answer,model,tickers,ts FROM memories ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            "q":       r["question"],
            "a":       r["answer"],
            "model":   r["model"],
            "tickers": json.loads(r["tickers"]),
            "ts":      r["ts"],
        }
        for r in rows
    ]


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(question: str, top_k: int = 3, min_score: float = 0.08) -> list[dict]:
    """
    Return up to top_k past Q&A pairs most relevant to question.

    Strategy 1 (preferred): vectorized numpy cosine similarity on stored embeddings.
    Strategy 2 (fallback):  pure-Python TF-IDF on question text.
    """
    # ── semantic path ─────────────────────────────────────────────────────────
    try:
        import numpy as np
        from compression import embed

        q_emb = embed(question)
        if q_emb is not None:
            with _conn() as con:
                rows = con.execute(
                    "SELECT question,answer,model,tickers,ts,embedding "
                    "FROM memories WHERE embedding IS NOT NULL"
                ).fetchall()

            if rows:
                q_vec  = np.array(q_emb, dtype=np.float32)
                matrix = np.stack([
                    np.frombuffer(r["embedding"], dtype=np.float32) for r in rows
                ])
                # dot product == cosine similarity (both sides are unit-normalized)
                scores = matrix @ q_vec
                top    = int(min(top_k, len(rows)))
                idx    = np.argsort(scores)[::-1][:top]
                results = [
                    {
                        "q":       rows[i]["question"],
                        "a":       rows[i]["answer"],
                        "model":   rows[i]["model"],
                        "tickers": json.loads(rows[i]["tickers"]),
                        "ts":      rows[i]["ts"],
                        "score":   round(float(scores[i]), 3),
                    }
                    for i in idx if scores[i] >= min_score
                ]
                if results:
                    return results
    except Exception:
        pass

    # ── TF-IDF fallback ───────────────────────────────────────────────────────
    with _conn() as con:
        rows = con.execute(
            "SELECT question,answer,model,tickers,ts FROM memories"
        ).fetchall()

    if not rows:
        return []

    memories = [
        {"q": r["question"], "a": r["answer"],
         "model": r["model"], "tickers": json.loads(r["tickers"]), "ts": r["ts"]}
        for r in rows
    ]
    return _tfidf_retrieve(question, memories, top_k, min_score)


# ── Pure-Python TF-IDF ────────────────────────────────────────────────────────

_STOP = {
    "a","an","the","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall","can",
    "i","you","we","they","he","she","it","this","that","these","those","and","or",
    "but","in","on","at","to","for","of","with","by","from","up","about","into",
    "through","my","your","our","their","its","me","him","her","us","them","what",
    "which","who","how","when","where","why","need","want","get","give","make",
}

def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9$]+", text.lower())
    return [t for t in tokens if t not in _STOP and len(t) > 1]


def _tf(tokens: list[str]) -> dict[str, float]:
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    n = max(len(tokens), 1)
    return {t: c / n for t, c in freq.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(a.get(t, 0.0) * v for t, v in b.items())
    na  = math.sqrt(sum(v * v for v in a.values()))
    nb  = math.sqrt(sum(v * v for v in b.values()))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def _tfidf_retrieve(question: str, memories: list[dict], top_k: int, min_score: float) -> list[dict]:
    corpus     = [m["q"] for m in memories]
    all_tokens = [_tokenize(q) for q in corpus]

    df: dict[str, int] = {}
    for toks in all_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    N   = len(corpus)
    idf = {t: math.log((N + 1) / (c + 1)) + 1 for t, c in df.items()}

    def tfidf(tokens: list[str]) -> dict[str, float]:
        tf = _tf(tokens)
        return {t: tf[t] * idf.get(t, 1.0) for t in tf}

    q_vec  = tfidf(_tokenize(question))
    scored = [(_cosine(q_vec, tfidf(toks)), i) for i, toks in enumerate(all_tokens)]
    scored = [(s, i) for s, i in scored if s >= min_score]
    scored.sort(reverse=True)
    return [{**memories[i], "score": round(s, 3)} for s, i in scored[:top_k]]


# ── Bootstrap on import ───────────────────────────────────────────────────────
_init_db()
