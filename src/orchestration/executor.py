import asyncio
import logging
import os
import httpx
from typing import List, Dict, Any
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger("ExecutorNode")
RETRIEVAL_ENDPOINT = os.getenv("RETRIEVAL_API_URL", "http://localhost:8000/retrieve")

async def _fetch_single_retrieval(client: httpx.AsyncClient, payload: dict, semaphore: asyncio.Semaphore) -> tuple[List[dict], str]:
    """Issues HTTP request to /retrieve endpoint with concurrency bounds."""
    async with semaphore:
        try:
            response = await client.post(RETRIEVAL_ENDPOINT, json=payload, timeout=45.0)
            if response.status_code == 200:
                return response.json(), ""
            err_msg = f"HTTP {response.status_code} from /retrieve for sub-query: {payload.get('query')}"
            logger.error(err_msg)
            return [], err_msg
        except Exception as e:
            err_msg = f"Network error during retrieval for query '{payload.get('query')}': {str(e)}"
            logger.error(err_msg)
            return [], err_msg

async def _fetch_local(retriever, payload: dict, sem: asyncio.Semaphore) -> tuple[list[dict], str]:
    async with sem:
        try:
            return await retriever(payload), ""
        except Exception as e:
            msg = f"Local retrieval error for '{payload.get('query')}': {e}"
            logger.error(msg)
            return [], msg
        
def build_payloads(sub_queries: List[dict]) -> List[dict]:
    payloads = []
    is_multi_hop = len(sub_queries) > 1
    base_top_k_parents = 2 if is_multi_hop else 3
    table_top_k_parents = 3 if is_multi_hop else 5

    for sq in sub_queries:
        mode = sq.get("retrieval_mode", "HYBRID")
        common_params = {
            "query": sq["query"],
            "ticker": sq.get("ticker"),
            "fiscal_year": sq.get("fiscal_year"),
            "item_number": sq.get("item_number"),
        }
        table_payload = {**common_params, "table_only": True, "top_k_chunks": 30, "top_k_parents": table_top_k_parents}
        prose_payload = {**common_params, "table_only": False, "top_k_chunks": 20, "top_k_parents": base_top_k_parents}
        
        if mode == "TABLE":
            payloads.append(table_payload)
        elif mode == "PROSE":
            payloads.append(prose_payload)
        elif mode == "HYBRID":
            payloads.append(table_payload)
            payloads.append(prose_payload)
    return payloads        

async def execute_retrieval(state: dict, config: RunnableConfig | None = None) -> dict:
    """Executor Node: Issues HTTP requests to /retrieve and merges context blocks."""
    sub_queries = state.get("sub_queries", [])
    if not sub_queries:
        return {"retrieved_contexts": [], "retrieval_errors": []}

    retriever = (config or {}).get("configurable", {}).get("retriever")
    sem = asyncio.Semaphore(int(os.getenv("RETRIEVAL_CONCURRENCY", "4")))
    
    payloads = build_payloads(sub_queries)

    if retriever is not None:
        results = await asyncio.gather(*(_fetch_local(retriever, p, sem) for p in payloads))
    else:
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
        async with httpx.AsyncClient(limits=limits) as client:
            results = await asyncio.gather(*(_fetch_single_retrieval(client, p, sem) for p in payloads))

    best_contexts_by_parent: Dict[str, Dict[str, Any]] = {}
    retrieval_errors: List[str] = []
    
    for contexts, error in results:
        if error:
            retrieval_errors.append(error)
        for ctx in contexts:
            p_id = ctx.get("parent_id")
            if not p_id:
                continue
            if p_id not in best_contexts_by_parent or ctx.get("score", 0.0) > best_contexts_by_parent[p_id].get("score", 0.0):
                best_contexts_by_parent[p_id] = ctx

    return {
        "retrieved_contexts": list(best_contexts_by_parent.values()),
        "retrieval_errors": retrieval_errors
    }