#!/usr/bin/env python3
"""Tushare news -> DeepSeek tags -> Jina ST embeddings (default) -> Qdrant upsert (500/batch) + TTL delete."""

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


def _embed_text_for_row(title: str, content: str, max_chars: int = 3200) -> str:
    body = (content or "")[:max_chars]
    return f"{title or ''}\n{body}"


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
        "======== ingest start: days=%s collection=%r dry_run=%s skip_delete=%s tag_batch=%s upsert_batch=%s ========",
        args.days,
        collection,
        args.dry_run,
        args.skip_delete,
        args.tag_batch,
        upsert_bs,
    )

    logger.info("[1/5] Fetching news from Tushare …")
    t0 = time.perf_counter()
    df = fetch_tushare_news(days=args.days)
    logger.info("[1/5] Tushare done in %.2fs — %s rows", time.perf_counter() - t0, len(df))
    if df.empty:
        logger.info("No rows from Tushare; exiting.")
        return 0

    logger.info("[2/5] LLM structured tagging (tickers / industry / concept) …")
    t0 = time.perf_counter()
    df = tag_news_dataframe(df, rows_per_llm_call=args.tag_batch)
    logger.info("[2/5] Tagging done in %.2fs — %s rows", time.perf_counter() - t0, len(df))

    logger.info("[3/5] Building embedding texts (title + body snippet per row) …")
    t0 = time.perf_counter()
    texts = [
        _embed_text_for_row(str(getattr(r, "title", "")), str(getattr(r, "content", "")))
        for r in df.itertuples(index=False)
    ]
    logger.info("[3/5] Built %s strings in %.2fs", len(texts), time.perf_counter() - t0)

    if args.dry_run:
        logger.info("[dry-run] Skipping vectorize + Qdrant. Sample first row (subset of fields):")
        sample = df.iloc[0].to_dict()
        logger.info(
            "%s",
            json.dumps(
                {k: sample[k] for k in ("stable_id", "title", "tickers", "pub_ts")},
                ensure_ascii=False,
            )[:2000],
        )
        logger.info(
            "======== ingest finished (dry-run) total %.2fs ========",
            time.perf_counter() - t_pipeline,
        )
        return 0

    dim = embedding_dim()
    logger.info("[4/5] Vectorizing (%s texts, embedding_dim hint=%s) …", len(texts), dim)
    t0 = time.perf_counter()
    vectors = embed_texts(texts)
    logger.info("[4/5] Vectorize done in %.2fs", time.perf_counter() - t0)
    if vectors and len(vectors[0]) != dim:
        logger.warning("Vector len %s != EMBEDDING_DIMENSIONS %s", len(vectors[0]), dim)

    logger.info(
        "[5/5] Qdrant: connect → ensure collection (dim=%s) → payload indexes → build points → upsert",
        len(vectors[0]) if vectors else dim,
    )
    t0 = time.perf_counter()
    client = make_client()
    try:
        ensure_collection(client, collection, vector_size=len(vectors[0]) if vectors else dim)
        logger.info("[5/5] Ensuring payload keyword indexes on %r …", collection)
        ensure_payload_indexes(client, collection)
        logger.info("[5/5] Building %s Qdrant points (payload + vector) …", len(df))
        points = build_points(df, vectors, ingestion_date)
        upsert_batches(client, collection, points, batch_size=upsert_bs)
        if not args.skip_delete:
            cutoff = int((datetime.now() - timedelta(days=30)).timestamp())
            logger.info("[5/5] TTL cleanup: deleting points with pub_ts < %s", cutoff)
            delete_points_older_than(client, collection, pub_ts_lt=cutoff)
        else:
            logger.info("[5/5] TTL cleanup skipped (--skip-delete)")
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
    pi.set_defaults(func=cmd_ingest)

    pdel = sub.add_parser("delete-only", help="Only run 30-day pub_ts cleanup")
    pdel.set_defaults(func=cmd_delete_only)

    args = p.parse_args()
    logger.info("Command: %s", " ".join(sys.argv))
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
