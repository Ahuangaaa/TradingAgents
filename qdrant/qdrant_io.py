"""Qdrant client helpers: collection, payload indexes, batched upsert, TTL delete."""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)

logger = logging.getLogger(__name__)


def make_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    key = (os.getenv("QDRANT_API_KEY") or "").strip()
    kwargs: dict[str, Any] = {}
    if key:
        kwargs["api_key"] = key
    logger.info("Qdrant: connecting to %s (api_key=%s)", url, "set" if key else "none")
    return QdrantClient(url, **kwargs)


def collection_names(client: QdrantClient) -> set[str]:
    return {c.name for c in client.get_collections().collections}


def ensure_collection(
    client: QdrantClient,
    name: str,
    *,
    vector_size: int,
    distance: Distance = Distance.COSINE,
) -> None:
    if name in collection_names(client):
        logger.info("Qdrant: collection %r already exists — skip create", name)
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=distance),
    )
    logger.info("Created collection %s (dim=%s)", name, vector_size)


def ensure_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    """Keyword indexes for filter-heavy fields (ignore if already exists)."""
    for field in ("tickers", "industry_tags", "concept_tags"):
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
                wait=True,
            )
            logger.info("Payload index created: %s.%s", collection_name, field)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "already exists" in msg or "duplicate" in msg or "409" in msg:
                logger.debug("Index exists or skip: %s — %s", field, exc)
            else:
                logger.warning("Payload index %s: %s", field, exc)


def upsert_batches(
    client: QdrantClient,
    collection_name: str,
    points: Iterable[PointStruct],
    *,
    batch_size: int = 500,
) -> int:
    """Upsert in chunks; returns total points written."""
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x: Iterable, **_k):  # type: ignore[no-untyped-def]
            return x

    plist = list(points)
    logger.info(
        "Qdrant: upsert starting — %s points into %r, batch_size=%s",
        len(plist),
        collection_name,
        batch_size,
    )
    total = 0
    for i in tqdm(range(0, len(plist), batch_size), desc="qdrant upsert", unit="batch"):
        batch = plist[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch, wait=True)
        total += len(batch)
    logger.info("Upserted %s points into %s", total, collection_name)
    return total


def delete_points_older_than(
    client: QdrantClient,
    collection_name: str,
    *,
    pub_ts_lt: int,
) -> None:
    """Delete points where payload ``pub_ts`` < ``pub_ts_lt`` (unix seconds)."""
    if collection_name not in collection_names(client):
        logger.warning("Collection %s does not exist; skip delete.", collection_name)
        return
    flt = Filter(
        must=[
            FieldCondition(
                key="pub_ts",
                range=Range(lt=float(pub_ts_lt)),
            )
        ]
    )
    logger.info(
        "Qdrant: deleting points where pub_ts < %s (%s)",
        pub_ts_lt,
        collection_name,
    )
    client.delete(collection_name=collection_name, points_selector=flt, wait=True)
    logger.info("Qdrant: delete filter applied on %s", collection_name)
