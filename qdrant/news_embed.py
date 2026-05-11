"""Text embeddings: default local ``jinaai/jina-embeddings-v3`` via sentence-transformers; optional HTTP OpenAI-compatible ``/v1/embeddings``."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Sequence

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_ST_MODEL = "jinaai/jina-embeddings-v3"


def _backend() -> str:
    return (os.getenv("NEWS_EMBED_BACKEND") or "sentence_transformers").strip().lower()


def embedding_dim() -> int:
    """Hint for empty-vector edge cases; real size comes from ``len(vectors[0])`` after ``embed_texts``."""
    return int(os.getenv("EMBEDDING_DIMENSIONS") or os.getenv("NEWS_EMBED_DIM") or "1024")


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Return one vector per input string (same order)."""
    if not texts:
        return []
    backend = _backend()
    t0 = time.perf_counter()
    logger.info("Embed: start backend=%r count=%s texts", backend, len(texts))
    if backend in ("st", "sentence_transformers", "sentence-transformers"):
        out = _embed_sentence_transformers(list(texts))
    else:
        out = _embed_openai_compatible(list(texts))
    elapsed = time.perf_counter() - t0
    dim = len(out[0]) if out else 0
    logger.info("Embed: finished in %.2fs — vectors=%s dim=%s", elapsed, len(out), dim)
    return out


def _embed_openai_compatible(texts: list[str]) -> list[list[float]]:
    key = (os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "Embedding (NEWS_EMBED_BACKEND=openai): set EMBEDDING_API_KEY or OPENAI_API_KEY."
        )
    base = (os.getenv("EMBEDDING_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    model = (os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small").strip()
    dim = embedding_dim()
    url = f"{base}/embeddings"
    out_vecs: list[list[float]] = []
    chunk_size = int(os.getenv("EMBEDDING_BATCH_INPUTS") or "32")
    n_chunks = (len(texts) + chunk_size - 1) // chunk_size
    logger.info(
        "Embed HTTP: POST %s model=%r chunks=%s (batch_inputs=%s)",
        url,
        model,
        n_chunks,
        chunk_size,
    )
    for i in range(0, len(texts), chunk_size):
        batch = texts[i : i + chunk_size]
        chunk_no = i // chunk_size + 1
        logger.info(
            "Embed HTTP: chunk %s/%s (inputs %s..%s)",
            chunk_no,
            n_chunks,
            i,
            min(i + len(batch), len(texts)) - 1,
        )
        payload: dict[str, Any] = {"model": model, "input": batch}
        if "3-small" in model or "3-large" in model:
            payload["dimensions"] = dim
        with httpx.Client(timeout=120.0) as client:
            r = client.post(
                url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
            )
            if r.status_code >= 400 and "dimensions" in payload:
                payload.pop("dimensions", None)
                r = client.post(
                    url,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json=payload,
                )
            r.raise_for_status()
            data = r.json()
        embs = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
        for row in embs:
            vec = row.get("embedding")
            if not isinstance(vec, list):
                raise RuntimeError("Invalid embedding response")
            out_vecs.append([float(x) for x in vec])
    if len(out_vecs) != len(texts):
        raise RuntimeError(f"Embedding count mismatch: got {len(out_vecs)} expected {len(texts)}")
    return out_vecs


_st_model: Any = None


def _jina_encode_kwargs(model_name: str) -> dict[str, Any]:
    if "jina-embeddings-v3" not in model_name.lower():
        return {}
    task = (os.getenv("JINA_EMBED_TASK") or "retrieval.passage").strip()
    return {"task": task, "prompt_name": task}


def _jina_query_encode_kwargs(model_name: str) -> dict[str, Any]:
    """Jina v3 asymmetric retrieval: query side (indexed docs use ``retrieval.passage``)."""
    if "jina-embeddings-v3" not in model_name.lower():
        return {}
    task = (os.getenv("JINA_EMBED_QUERY_TASK") or "retrieval.query").strip()
    return {"task": task, "prompt_name": task}


def _embed_sentence_transformers(
    texts: list[str],
    *,
    encode_extra: dict[str, Any] | None = None,
) -> list[list[float]]:
    global _st_model  # noqa: PLW0603
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "Install sentence-transformers: pip install -r qdrant/requirements-ingest.txt"
        ) from exc

    model_name = (os.getenv("SENTENCE_TRANSFORMERS_MODEL") or _DEFAULT_ST_MODEL).strip()
    if encode_extra is None:
        extra = _jina_encode_kwargs(model_name)
    else:
        extra = encode_extra
    if _st_model is None:
        logger.info("Embed ST: loading model %r …", model_name)
        _st_model = SentenceTransformer(model_name, trust_remote_code=True)
    else:
        logger.info("Embed ST: using loaded model %r encode_kwargs=%s", model_name, extra or "{}")
    vecs = _st_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        **extra,
    )
    return [v.tolist() for v in vecs]


def embed_query_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed short **queries** for vector search (Jina v3 uses ``retrieval.query`` vs passage docs)."""
    if not texts:
        return []
    backend = _backend()
    t0 = time.perf_counter()
    logger.info("Embed query: start backend=%r count=%s", backend, len(texts))
    if backend in ("st", "sentence_transformers", "sentence-transformers"):
        model_name = (os.getenv("SENTENCE_TRANSFORMERS_MODEL") or _DEFAULT_ST_MODEL).strip()
        extra = _jina_query_encode_kwargs(model_name)
        out = _embed_sentence_transformers(list(texts), encode_extra=extra)
    else:
        out = _embed_openai_compatible(list(texts))
    elapsed = time.perf_counter() - t0
    dim = len(out[0]) if out else 0
    logger.info("Embed query: finished in %.2fs — vectors=%s dim=%s", elapsed, len(out), dim)
    return out
