import logging

logger = logging.getLogger("RouterNode")

ITEM_TO_MODE = {
    "Item 1A": "PROSE",
    "Item 7": "HYBRID",
    "Item 8": "TABLE"
}

async def route_sub_queries(state: dict) -> dict:
    """Router Node: Deterministic mode assignment derived directly from item_number."""
    sub_queries = state.get("sub_queries", [])
    
    for sq in sub_queries:
        item = sq.get("item_number")
        sq["retrieval_mode"] = ITEM_TO_MODE.get(item, "HYBRID")
        
    return {"sub_queries": sub_queries}