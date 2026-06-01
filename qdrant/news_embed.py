"""Text embeddings via Alibaba DashScope online ``text-embedding-v4`` (no local models)."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import Any, Sequence

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "text-embedding-v4"
_MAX_BATCH = 10
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF_SEC = 2.0


class DashScopeEmbedError(RuntimeError):
    """Raised when DashScope TextEmbedding returns a non-OK HTTP status."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _embed_concurrency() -> int:
    raw = (os.getenv("NEWS_EMBED_CONCURRENCY") or "4").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 4
    return max(1, min(16, n))


def _embed_retries() -> int:
    raw = (os.getenv("NEWS_EMBED_RETRIES") or str(_DEFAULT_RETRIES)).strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_RETRIES


def _embed_retry_backoff_sec() -> float:
    raw = (os.getenv("NEWS_EMBED_RETRY_BACKOFF_SEC") or str(_DEFAULT_BACKOFF_SEC)).strip()
    try:
        return max(0.5, float(raw))
    except ValueError:
        return _DEFAULT_BACKOFF_SEC


def _is_retryable_embed_error(exc: BaseException) -> bool:
    """Transient network / gateway errors from DashScope HTTP layer."""
    try:
        import requests
    except ImportError:
        requests = None  # type: ignore[assignment]

    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, (TimeoutError, ConnectionError, OSError)):
            return True
        if requests is not None and isinstance(cur, requests.exceptions.RequestException):
            return True
        name = type(cur).__name__.lower()
        if "timeout" in name or "disconnect" in name or "connection" in name:
            return True
        msg = str(cur).lower()
        if any(
            token in msg
            for token in (
                "timed out",
                "timeout",
                "connection aborted",
                "remote end closed",
                "connection reset",
                "temporarily unavailable",
                "too many requests",
            )
        ):
            return True
        if isinstance(cur, DashScopeEmbedError) and cur.status_code is not None:
            return _is_retryable_dashscope_status(cur.status_code)
        cur = cur.__cause__ or cur.__context__
    return False


def _is_retryable_dashscope_status(status_code: int) -> bool:
    return status_code in (408, 429, 500, 502, 503, 504)


def _dashscope_api_key() -> str:
    """百炼 / DashScope API key: prefer ``DASHSCOPE_API_KEY`` as requested, then common aliases."""
    return (
        os.getenv("DASHSCOPE_API_KEY")
        or ""
    ).strip()


def embedding_dim() -> int:
    """Vector size for Qdrant collection and DashScope ``dimension`` (v3/v4 support this parameter)."""
    return int(os.getenv("EMBEDDING_DIMENSIONS") or os.getenv("NEWS_EMBED_DIM") or "1024")


def _parse_vectors_from_output(output: Any, *, n_inputs: int) -> list[list[float]]:
    if output is None:
        raise RuntimeError("DashScope TextEmbedding: empty output")
    if isinstance(output, dict):
        items = output.get("embeddings")
        if items is None and "embedding" in output:
            emb = output["embedding"]
            if isinstance(emb, list) and emb and isinstance(emb[0], (int, float)):
                if n_inputs != 1:
                    raise RuntimeError("DashScope: single-vector output but batch size > 1")
                return [[float(x) for x in emb]]
        if not items:
            raise RuntimeError(f"DashScope: no embeddings in output: {output!r}")
        vecs: list[list[float]] = []
        for item in items:
            if isinstance(item, dict):
                vec = item.get("embedding")
            else:
                vec = getattr(item, "embedding", None)
            if not isinstance(vec, list):
                raise RuntimeError(f"DashScope: bad embedding item: {item!r}")
            vecs.append([float(x) for x in vec])
        if len(vecs) != n_inputs:
            raise RuntimeError(f"DashScope: expected {n_inputs} vectors, got {len(vecs)}")
        return vecs
    raise RuntimeError(f"DashScope: unexpected output type: {type(output)!r}")


def _embed_dashscope_batch_once(texts: list[str]) -> list[list[float]]:
    import dashscope
    from dashscope import TextEmbedding

    key = _dashscope_api_key()
    if not key:
        raise RuntimeError(
            "DashScope embedding: set DASHSCOPE_API_KEY for text-embedding-v4."
        )
    dashscope.api_key = key

    dim = embedding_dim()
    model = (os.getenv("DASHSCOPE_EMBED_MODEL") or _DEFAULT_MODEL).strip()

    resp = TextEmbedding.call(model=model, input=texts, dimension=dim)

    if resp.status_code != HTTPStatus.OK:
        msg = getattr(resp, "message", None) or getattr(resp, "code", None) or str(resp)
        raise DashScopeEmbedError(
            f"DashScope TextEmbedding failed: status={resp.status_code!r} detail={msg!r}",
            status_code=int(resp.status_code),
        )

    output = getattr(resp, "output", None)
    if not isinstance(output, dict):
        raise RuntimeError(f"DashScope: expected dict output, got {type(output)!r}: {output!r}")
    return _parse_vectors_from_output(output, n_inputs=len(texts))


def _embed_dashscope_batch(texts: list[str], *, _allow_split: bool = True) -> list[list[float]]:
    """Call DashScope once per batch; retry transient errors, then split batch if needed."""
    if not texts:
        return []

    retries = _embed_retries()
    backoff = _embed_retry_backoff_sec()
    last_exc: BaseException | None = None

    for attempt in range(retries + 1):
        try:
            return _embed_dashscope_batch_once(texts)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_retryable_embed_error(exc):
                raise

        if attempt >= retries:
            break
        sleep_sec = backoff * (2**attempt)
        logger.warning(
            "Embed DashScope: retry %s/%s (batch_size=%s) after %s; sleeping %.1fs",
            attempt + 1,
            retries,
            len(texts),
            last_exc,
            sleep_sec,
        )
        time.sleep(sleep_sec)

    if _allow_split and len(texts) > 1:
        mid = len(texts) // 2
        logger.warning(
            "Embed DashScope: splitting batch of %s into %s + %s after failures: %s",
            len(texts),
            mid,
            len(texts) - mid,
            last_exc,
        )
        return _embed_dashscope_batch(texts[:mid], _allow_split=True) + _embed_dashscope_batch(
            texts[mid:], _allow_split=True
        )

    assert last_exc is not None
    raise last_exc


def _embed_batch_with_progress(
    batch_idx: int,
    n_batches: int,
    batch: list[str],
    *,
    text_start: int,
    n_texts: int,
) -> list[list[float]]:
    """Run one DashScope batch; log start/finish so ingest shows embedding progress."""
    n_in_batch = len(batch)
    t1 = time.perf_counter()
    text_end = text_start + n_in_batch
    logger.info(
        "Embed DashScope: [%s/%s] texts %s–%s of %s (%s in this request)",
        batch_idx + 1,
        n_batches,
        text_start + 1,
        text_end,
        n_texts,
        n_in_batch,
    )
    vecs = _embed_dashscope_batch(batch)
    dt = time.perf_counter() - t1
    logger.info(
        "Embed DashScope: [%s/%s] done in %.2fs — got %s vectors (cumulative texts done: %s/%s)",
        batch_idx + 1,
        n_batches,
        dt,
        len(vecs),
        text_end,
        n_texts,
    )
    return vecs


def _embed_job(args: tuple[int, int, list[str], int, int]) -> list[list[float]]:
    batch_idx, n_batches, batch, text_start, n_texts = args
    return _embed_batch_with_progress(
        batch_idx, n_batches, batch, text_start=text_start, n_texts=n_texts
    )


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Return one dense vector per input string (same order).

    Up to 10 texts per DashScope request; multiple batches run in parallel when
    ``NEWS_EMBED_CONCURRENCY`` > 1 (see ``_embed_concurrency``).
    """
    if not texts:
        return []
    lst = [str(t) for t in texts]
    t0 = time.perf_counter()
    model = (os.getenv("DASHSCOPE_EMBED_MODEL") or _DEFAULT_MODEL).strip()
    workers = _embed_concurrency()
    batches = [lst[i : i + _MAX_BATCH] for i in range(0, len(lst), _MAX_BATCH)]
    logger.info(
        "Embed DashScope: model=%r count=%s dimension=%s batches=%s workers=%s",
        model,
        len(lst),
        embedding_dim(),
        len(batches),
        workers,
    )
    n_texts = len(lst)
    n_batches = len(batches)
    jobs = [
        (i, n_batches, batches[i], i * _MAX_BATCH, n_texts)
        for i in range(n_batches)
    ]
    all_vecs: list[list[float]] = []
    if workers <= 1 or n_batches <= 1:
        for job in jobs:
            all_vecs.extend(_embed_job(job))
    else:
        # executor.map preserves batch order → final vectors match input order
        with ThreadPoolExecutor(max_workers=min(workers, n_batches)) as pool:
            for chunk_vecs in pool.map(_embed_job, jobs):
                all_vecs.extend(chunk_vecs)
    elapsed = time.perf_counter() - t0
    dim = len(all_vecs[0]) if all_vecs else 0
    logger.info("Embed DashScope: finished in %.2fs — vectors=%s dim=%s", elapsed, len(all_vecs), dim)
    return all_vecs


def embed_query_texts(texts: Sequence[str]) -> list[list[float]]:
    """Same as ``embed_texts`` — ``text-embedding-v4`` uses a single symmetric embedding for passage and query."""
    return embed_texts(texts)
