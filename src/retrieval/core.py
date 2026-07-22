import sqlite3
import logging
import asyncio
from qdrant_client.models import Prefetch, Fusion, FusionQuery, SparseVector, Filter, FieldCondition, MatchValue
from src.retrieval.models import QueryRequest

COLLECTION_NAME = "advanced_sec_edgar_production"
logger = logging.getLogger("CoreRetrieval")

def hydrate_windowed_context(unique_parent_hits: list, window_buffer: int = 1000) -> list[dict]:
    """Hydrates tables fully, but extracts only targeted surrounding windows for prose."""
    if not unique_parent_hits:
        return []
        
    # Extract IDs safely to prevent empty SQL IN clauses
    unique_parent_ids = [hit.payload.get("parent_id") for hit, _ in unique_parent_hits if hit.payload.get("parent_id")]
    if not unique_parent_ids:
        return []
        
    conn = sqlite3.connect("parent_docstore.db")
    cursor = conn.cursor()
    
    # NOTE: PRAGMA journal_mode=WAL is intentionally excluded here. 
    # It is a persistent setting applied during ingestion. Read threads must not alter it.
    
    placeholders = ",".join(["?"] * len(unique_parent_ids))
    query = f"SELECT id, full_text FROM parent_documents WHERE id IN ({placeholders})"
    cursor.execute(query, unique_parent_ids)
    
    parent_map = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()

    hydrated_blocks = []

    for hit, score in unique_parent_hits:
        payload = hit.payload
        parent_id = payload.get("parent_id")
        
        if not parent_id:
            continue
            
        full_text = parent_map.get(parent_id, "")
        is_table = payload.get("is_table", False)
        
        if is_table:
            # FIX: JIT Compilation of bloated SEC HTML into clean Markdown
            import pandas as pd
            from io import StringIO
            from bs4 import BeautifulSoup
            
            try:
                # Parse the raw HTML table stored in SQLite
                df_list = pd.read_html(StringIO(full_text))
                if df_list:
                    df = df_list[0].dropna(how='all').dropna(axis=1, how='all')
                    # Convert to strict Markdown (collapses thousands of HTML tokens to a few hundred)
                    content = df.to_markdown(index=False)
                else:
                    raise ValueError("No table found")
            except Exception:
                # Fallback: aggressively strip HTML tags if pandas fails
                content = BeautifulSoup(full_text, "lxml").get_text(separator=" | ", strip=True)
        else:
            # Prose: Find the child chunk inside the parent and expand the window
            child_text = payload.get("text", "")
            start_idx = full_text.find(child_text)
            
            if start_idx != -1:
                # Extract surrounding window safely
                window_start = max(0, start_idx - window_buffer)
                window_end = min(len(full_text), start_idx + len(child_text) + window_buffer)
                content = f"...{full_text[window_start:window_end]}..."
            else:
                content = child_text
                
        hydrated_blocks.append({
            "parent_id": parent_id,
            "ticker": payload.get("ticker"),
            "fiscal_year": payload.get("fiscal_year"),
            "item_number": payload.get("item_number"),
            "is_table": is_table,
            "content": f"[Match Confidence: {score:.2f}]\n{content}"
        })

    return hydrated_blocks

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
    
    # 5. Extract and Deduplicate by exact Parent IDs (Preserving Tuples)
    unique_parent_hits = []
    seen_parents = set()
    for hit, score in scored_hits:
        parent_id = hit.payload.get("parent_id")
        if parent_id and parent_id not in seen_parents:
            seen_parents.add(parent_id)
            unique_parent_hits.append((hit, score))
        if len(unique_parent_hits) >= request.top_k_parents:
            break
            
    # 6. SQL Hydration (Windowed)
    return await asyncio.to_thread(hydrate_windowed_context, unique_parent_hits)