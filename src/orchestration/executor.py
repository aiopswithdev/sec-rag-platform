import asyncio
import logging
import httpx
from typing import List, Dict, Any

logger = logging.getLogger("ExecutorNode")
RETRIEVAL_ENDPOINT = "http://localhost:8000/retrieve"

async def _fetch_single_retrieval(client: httpx.AsyncClient, payload: dict, semaphore: asyncio.Semaphore) -> tuple[List[dict], str]:
    """Issues HTTP request to /retrieve endpoint with concurrency bounds."""
    async with semaphore:
        try:
            response = await client.post(RETRIEVAL_ENDPOINT, json=payload, timeout=10.0)
            if response.status_code == 200:
                return response.json(), ""
            err_msg = f"HTTP {response.status_code} from /retrieve for sub-query: {payload.get('query')}"
            logger.error(err_msg)
            return [], err_msg
        except Exception as e:
            err_msg = f"Network error during retrieval for query '{payload.get('query')}': {str(e)}"
            logger.error(err_msg)
            return [], err_msg


async def execute_retrieval(state: dict) -> dict:
    """Executor Node: Issues HTTP requests to /retrieve and merges context blocks."""
    sub_queries = state["sub_queries"]
    semaphore = asyncio.Semaphore(2)  # Bound concurrent HTTP execution
    
    retrieved_contexts: List[Dict[str, Any]] = []
    retrieval_errors: List[str] = []
    seen_parent_ids = set()

    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = []
        
        for sq in sub_queries:
            mode = sq.get("retrieval_mode", "HYBRID")
            
            base_payload = {
                "query": sq["query"],
                "ticker": sq.get("ticker"),
                "fiscal_year": sq.get("fiscal_year"),
                "item_number": sq.get("item_number"),
                "top_k_chunks": 20,
                "top_k_parents": 3
            }
            
            if mode == "TABLE":
                p = base_payload.copy()
                p["table_only"] = True
                tasks.append(_fetch_single_retrieval(client, p, semaphore))
            elif mode == "PROSE":
                p = base_payload.copy()
                p["table_only"] = False
                tasks.append(_fetch_single_retrieval(client, p, semaphore))
            elif mode == "HYBRID":
                # Execute TWO parallel requests (one table, one prose) and merge
                p_table = base_payload.copy()
                p_table["table_only"] = True
                
                p_prose = base_payload.copy()
                p_prose["table_only"] = False
                
                tasks.append(_fetch_single_retrieval(client, p_table, semaphore))
                tasks.append(_fetch_single_retrieval(client, p_prose, semaphore))

        results = await asyncio.gather(*tasks)

    for contexts, error in results:
        if error:
            retrieval_errors.append(error)
        for ctx in contexts:
            p_id = ctx.get("parent_id")
            if p_id and p_id not in seen_parent_ids:
                seen_parent_ids.add(p_id)
                retrieved_contexts.append(ctx)

    return {
        "retrieved_contexts": retrieved_contexts,
        "retrieval_errors": retrieval_errors
    }