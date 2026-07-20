import sqlite3
import logging
import asyncio
from qdrant_client.models import Prefetch, Fusion, FusionQuery, SparseVector, Filter, FieldCondition, MatchValue
from src.retrieval.models import QueryRequest

COLLECTION_NAME = "advanced_sec_edgar_production"
logger = logging.getLogger("CoreRetrieval")

def hydrate_parents_from_sql(parent_ids: list[str]) -> list[dict]:
    """Fetch unabridged parent documents from SQLite."""
    if not parent_ids:
        return []
    
    conn = sqlite3.connect("parent_docstore.db")
    cursor = conn.cursor()
    
    # Staff-Level Realtime Harden: Force Write-Ahead Logging to prevent lockups on orchestrator gather
    cursor.execute("PRAGMA journal_mode=WAL;")
    
    placeholders = ",".join(["?"] * len(parent_ids))
    query = f"SELECT id, ticker, fiscal_year, item_number, is_table, full_text FROM parent_documents WHERE id IN ({placeholders})"
    
    cursor.execute(query, parent_ids)
    rows = cursor.fetchall()
    conn.close()
    
    fetched_ids = {row[0] for row in rows}
    missing_ids = set(parent_ids) - fetched_ids
    if missing_ids:
        logger.warning(f"Data Integrity Warning: Parent IDs missing in SQLite: {missing_ids}")
    
    hydrated = []
    for row in rows:
        hydrated.append({
            "parent_id": row[0],
            "ticker": row[1],
            "fiscal_year": row[2],
            "item_number": row[3],
            "is_table": bool(row[4]),
            "content": row[5]
        })
    return hydrated


async def execute_core_retrieval(request: QueryRequest, infra) -> list[dict]:
    """Encapsulates the isolated hybrid search, reranking, and SQL hydration pipeline."""
    must_conditions = []
    if request.ticker:
        must_conditions.append(FieldCondition(key="ticker", match=MatchValue(value=request.ticker)))
    if request.fiscal_year:
        must_conditions.append(FieldCondition(key="fiscal_year", match=MatchValue(value=request.fiscal_year)))
    if request.item_number:
        must_conditions.append(FieldCondition(key="item_number", match=MatchValue(value=request.item_number)))
    if request.table_only:
        must_conditions.append(FieldCondition(key="is_table", match=MatchValue(value=True)))

    query_filter = Filter(must=must_conditions) if must_conditions else None

    # Embed the query via async task offloading
    dense_query = await asyncio.to_thread(lambda: list(infra.dense_model.embed([request.query]))[0].tolist())
    sparse_query_result = await asyncio.to_thread(lambda: list(infra.sparse_model.embed([request.query]))[0])
    
    sparse_vector = SparseVector(
        indices=sparse_query_result.indices.tolist(),
        values=sparse_query_result.values.tolist()
    )

    prefetch_dense = Prefetch(
        query=dense_query,
        using="dense-text",
        limit=request.top_k_chunks,
        filter=query_filter
    )
    
    prefetch_sparse = Prefetch(
        query=sparse_vector,
        using="sparse-text",
        limit=request.top_k_chunks,
        filter=query_filter
    )

    search_results = await infra.qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[prefetch_dense, prefetch_sparse],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=request.top_k_chunks,
        with_payload=True
    )
    
    hits = search_results.points
    if not hits:
        return []

    chunk_texts = [hit.payload["text"] for hit in hits]
    rerank_scores = await asyncio.to_thread(lambda: list(infra.reranker.rerank(request.query, chunk_texts)))
    
    scored_hits = sorted(zip(hits, rerank_scores), key=lambda x: x[1], reverse=True)
    
    unique_parent_ids = []
    for hit, score in scored_hits:
        parent_id = hit.payload.get("parent_id")
        if parent_id and parent_id not in unique_parent_ids:
            unique_parent_ids.append(parent_id)
        if len(unique_parent_ids) >= request.top_k_parents:
            break
            
    return await asyncio.to_thread(hydrate_parents_from_sql, unique_parent_ids)