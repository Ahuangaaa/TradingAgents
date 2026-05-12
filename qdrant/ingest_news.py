#!/usr/bin/env python3
"""Tushare news → parallel HTML strip → parallel DeepSeek tags → build strings → parallel DashScope text-embedding-v4 → Qdrant upsert (+ optional parallel upload batches) + TTL delete."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Allow ``python qdrant/ingest_news.py`` from repo root: add this directory to path.
_QDIR = Path(__file__).resolve().parent
_REPO_ROOT = _QDIR.parent
if str(_QDIR) not in sys.path:
    sys.path.insert(0, str(_QDIR))
# ``news_llm_tags`` imports ``tradingagents.dataflows.macro_keywords`` for macro fallback.
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

# Same as cli/main.py: load repo-root .env so TUSHARE_TOKEN / LLM keys work when not exported.
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
    load_dotenv(_REPO_ROOT / ".env.enterprise", override=False)
except ImportError:
    pass

from qdrant_client.http.models import PointStruct

from news_embed import embed_texts, embedding_dim
from news_fetch import fetch_tushare_news
from news_llm_tags import tag_news_dataframe
from qdrant_io import (
    delete_points_older_than,
    ensure_collection,
    ensure_payload_indexes,
    make_client,
    upsert_batches,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ingest_news")

# Per-request HTTP logs are noisy; set INGEST_LOG_HTTP=1 to see httpx/httpcore DEBUG.
if os.getenv("INGEST_LOG_HTTP", "").strip().lower() not in ("1", "true", "yes"):
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _upsert_concurrency() -> int:
    raw = (os.getenv("INGEST_UPSERT_CONCURRENCY") or "1").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 1
    return max(1, min(16, n))


def _log_embedding_input_snapshot(df, texts: list[str]) -> None:
    """Log how tickers / industry_tags / concept_tags enter the **embedding** string (same text → DashScope).

    Payload fields in Qdrant are separate; retrieval uses ``news_qdrant_retrieval`` query text on the client side.
    """
    if len(texts) != len(df):
        logger.warning(
            "Embedding snapshot: texts length (%s) != df rows (%s) — skip preview",
            len(texts),
            len(df),
        )
        return

    n = len(df)
    with_tickers = 0
    with_ind = 0
    with_conc = 0
    for r in df.itertuples(index=False):
        if list(getattr(r, "tickers", None) or []):
            with_tickers += 1
        if list(getattr(r, "industry_tags", None) or []):
            with_ind += 1
        if list(getattr(r, "concept_tags", None) or []):
            with_conc += 1

    logger.info(
        "Embedding input stats: rows=%s | non-empty tickers=%s | industry_tags=%s | concept_tags=%s",
        n,
        with_tickers,
        with_ind,
        with_conc,
    )
    logger.info(
        "Tags append to the SAME string as title+body (lines 「标的代码:」「行业标签:」「概念主题:」) "
        "before DashScope text-embedding-v4 — one vector per row."
    )

    max_chars = int(os.getenv("INGEST_LOG_EMBED_PREVIEW_CHARS", "2200"))
    if texts:
        raw0 = texts[0]
        clipped = (
            raw0
            if len(raw0) <= max_chars
            else raw0[:max_chars] + "\n… [truncated, full len=%s]" % len(raw0)
        )
        logger.info(
            "First-row embedding text (passed to embed_texts → vector), preview:\n%s",
            clipped,
        )

    sample_env = (os.getenv("INGEST_LOG_EMBED_SAMPLE_ROWS") or "1").strip()
    try:
        extra_n = max(0, min(5, int(sample_env)))
    except ValueError:
        extra_n = 1
    if extra_n > 1 and len(texts) > 1:
        for i in range(1, min(extra_n, len(texts))):
            t = texts[i]
            clip = t if len(t) <= 800 else t[:800] + "…"
            logger.info("Embedding text preview row[%s] (truncated 800 chars):\n%s", i, clip)

    logger.info(
        "Retrieval note: ``news_qdrant_retrieval`` uses query built from ts_code+name+industry (and macro lexicon); "
        "document vectors already encode tag lines above, so queries need not repeat tags."
    )


def _embed_text_for_row(
    title: str,
    content: str,
    *,
    tickers: list[str] | None = None,
    industry_tags: list[str] | None = None,
    concept_tags: list[str] | None = None,
    max_chars: int = 3200,
) -> str:
    body = (content or "")[:max_chars]
    core = f"{title or ''}\n{body}"
    tx = [str(x).strip() for x in (tickers or []) if str(x).strip()]
    it = [str(x).strip() for x in (industry_tags or []) if str(x).strip()]
    ct = [str(x).strip() for x in (concept_tags or []) if str(x).strip()]
    bits = [core]
    if tx:
        bits.append("标的代码: " + " ".join(tx))
    if it:
        bits.append("行业标签: " + " ".join(it))
    if ct:
        bits.append("概念主题: " + " ".join(ct))
    return "\n".join(bits)


def build_points(df, vectors: list[list[float]], ingestion_date: str) -> list[PointStruct]:
    points: list[PointStruct] = []
    for row, vec in zip(df.itertuples(index=False), vectors):
        payload = {
            "title": str(getattr(row, "title", "") or ""),
            "content": str(getattr(row, "content", "") or "")[:8000],
            "pub_time": str(getattr(row, "pub_time", "") or ""),
            "pub_ts": int(getattr(row, "pub_ts", 0) or 0),
            "ingestion_date": ingestion_date,
            "source": str(getattr(row, "src", "") or ""),
            "source_type": str(getattr(row, "source_type", "") or ""),
            "url": str(getattr(row, "url", "") or ""),
            "tickers": list(getattr(row, "tickers", None) or []),
            "industry_tags": list(getattr(row, "industry_tags", None) or []),
            "concept_tags": list(getattr(row, "concept_tags", None) or []),
        }
        points.append(
            PointStruct(
                id=str(getattr(row, "stable_id")),
                vector=vec,
                payload=payload,
            )
        )
    return points


def cmd_ingest(args: argparse.Namespace) -> int:
    ingestion_date = datetime.now().strftime("%Y-%m-%d")
    collection = os.getenv("QDRANT_COLLECTION", "financial_news")
    upsert_bs = int(os.getenv("INGEST_UPSERT_BATCH_SIZE", "500"))
    t_pipeline = time.perf_counter()

    logger.info(
        "======== ingest start: days=%s collection=%r dry_run=%s skip_delete=%s html_strip_workers=%s tag_batch=%s tag_concurrency=%s embed_workers(env)=%s upsert_workers(env)=%s upsert_batch=%s ========",
        args.days,
        collection,
        args.dry_run,
        args.skip_delete,
        args.html_strip_concurrency,
        args.tag_batch,
        args.tag_concurrency,
        os.getenv("NEWS_EMBED_CONCURRENCY", "4"),
        _upsert_concurrency(),
        upsert_bs,
    )

    logger.info("[1/6] Fetching news from Tushare …")
    t0 = time.perf_counter()
    df = fetch_tushare_news(days=args.days)
    logger.info("[1/6] Tushare done in %.2fs — %s rows", time.perf_counter() - t0, len(df))
    if df.empty:
        logger.info("No rows from Tushare; exiting.")
        return 0

    logger.info(
        "[2/6][3/6] Parallel HTML strip → parallel DeepSeek tagging (see news_llm_tags for substeps) …"
    )
    t0 = time.perf_counter()
    df = tag_news_dataframe(
        df,
        rows_per_llm_call=args.tag_batch,
        tag_concurrency=args.tag_concurrency,
        html_strip_concurrency=args.html_strip_concurrency,
    )
    logger.info(
        "[2/6][3/6] Strip + tag total %.2fs — %s rows",
        time.perf_counter() - t0,
        len(df),
    )

    logger.info("[4/6] Building embedding strings (title + body + ticker/industry/concept tags per row) …")
    t0 = time.perf_counter()
    texts = [
        _embed_text_for_row(
            str(getattr(r, "title", "")),
            str(getattr(r, "content", "")),
            tickers=list(getattr(r, "tickers", None) or []),
            industry_tags=list(getattr(r, "industry_tags", None) or []),
            concept_tags=list(getattr(r, "concept_tags", None) or []),
        )
        for r in df.itertuples(index=False)
    ]
    logger.info("[4/6] Built %s strings in %.2fs", len(texts), time.perf_counter() - t0)

    _log_embedding_input_snapshot(df, texts)

    if args.dry_run:
        logger.info("[dry-run] Skipping vectorize + Qdrant. Sample first row (subset of fields):")
        sample = df.iloc[0].to_dict()
        logger.info(
            "%s",
            json.dumps(
                {
                    "stable_id": sample.get("stable_id"),
                    "title": (sample.get("title") or "")[:120],
                    "tickers": sample.get("tickers"),
                    "industry_tags": sample.get("industry_tags"),
                    "concept_tags": sample.get("concept_tags"),
                    "pub_ts": sample.get("pub_ts"),
                },
                ensure_ascii=False,
            )[:3000],
        )
        logger.info(
            "======== ingest finished (dry-run) total %.2fs ========",
            time.perf_counter() - t_pipeline,
        )
        return 0

    dim = embedding_dim()
    logger.info(
        "[5/6] Vectorizing with parallel DashScope batches — model text-embedding-v4 (see news_embed for workers) …"
    )
    t0 = time.perf_counter()
    vectors = embed_texts(texts)
    logger.info("[5/6] Vectorize done in %.2fs", time.perf_counter() - t0)
    if vectors and len(vectors[0]) != dim:
        logger.warning("Vector len %s != EMBEDDING_DIMENSIONS %s", len(vectors[0]), dim)

    uw = _upsert_concurrency()
    logger.info(
        "[6/6] Qdrant: connect → ensure collection (dim=%s) → payload indexes → build points → upsert (parallel=%s) …",
        len(vectors[0]) if vectors else dim,
        uw,
    )
    t0 = time.perf_counter()
    client = make_client()
    try:
        ensure_collection(client, collection, vector_size=len(vectors[0]) if vectors else dim)
        logger.info("[6/6] Ensuring payload keyword indexes on %r …", collection)
        ensure_payload_indexes(client, collection)
        logger.info("[6/6] Building %s Qdrant points (payload + vector) …", len(df))
        points = build_points(df, vectors, ingestion_date)
        upsert_batches(client, collection, points, batch_size=upsert_bs, max_workers=uw)
        if not args.skip_delete:
            cutoff = int((datetime.now() - timedelta(days=30)).timestamp())
            logger.info("[6/6] TTL cleanup: deleting points with pub_ts < %s", cutoff)
            delete_points_older_than(client, collection, pub_ts_lt=cutoff)
        else:
            logger.info("[6/6] TTL cleanup skipped (--skip-delete)")
    finally:
        client.close()
    logger.info(
        "======== ingest finished OK — Qdrant phase %.2fs — pipeline total %.2fs ========",
        time.perf_counter() - t0,
        time.perf_counter() - t_pipeline,
    )
    return 0


def cmd_delete_only(_: argparse.Namespace) -> int:
    collection = os.getenv("QDRANT_COLLECTION", "financial_news")
    cutoff = int((datetime.now() - timedelta(days=30)).timestamp())
    logger.info(
        "======== delete-only: collection=%r pub_ts_lt=%s (points older than ~30d) ========",
        collection,
        cutoff,
    )
    client = make_client()
    try:
        delete_points_older_than(client, collection, pub_ts_lt=cutoff)
    finally:
        client.close()
    logger.info("======== delete-only finished ========")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Ingest Tushare news into Qdrant.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="Fetch, tag, embed, upsert, optional TTL delete")
    pi.add_argument("--days", type=int, default=31, help="Lookback days for Tushare fetch")
    pi.add_argument("--dry-run", action="store_true", help="Fetch+tag only; print sample; no Qdrant/embed")
    pi.add_argument("--skip-delete", action="store_true", help="Skip delete_points_older_than")
    pi.add_argument("--tag-batch", type=int, default=15, help="Rows per LLM tagging call")
    pi.add_argument(
        "--html-strip-concurrency",
        type=int,
        default=None,
        help="Parallel local HTML strip workers (default: env NEWS_HTML_STRIP_CONCURRENCY or 8)",
    )
    pi.add_argument(
        "--tag-concurrency",
        type=int,
        default=None,
        help="Parallel LLM HTTP batches (default: env NEWS_TAG_LLM_CONCURRENCY or 4; use 1 for serial)",
    )
    pi.set_defaults(func=cmd_ingest)

    pdel = sub.add_parser("delete-only", help="Only run 30-day pub_ts cleanup")
    pdel.set_defaults(func=cmd_delete_only)

    args = p.parse_args()
    logger.info("Command: %s", " ".join(sys.argv))
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
