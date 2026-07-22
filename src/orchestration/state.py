from typing import TypedDict, List, Dict, Any, Optional

class SubQuery(TypedDict):
    query: str
    ticker: Optional[str]
    fiscal_year: Optional[int]
    item_number: Optional[str]
    retrieval_mode: str  # "PROSE", "TABLE", or "HYBRID"

class AgentState(TypedDict):
    raw_query: str
    available_inventory: List[Dict[str, Any]]
    sub_queries: List[SubQuery]
    retrieved_contexts: List[Dict[str, Any]]
    retrieval_errors: List[str]
    final_answer: str
    sources: List[Dict[str, Any]]
    telemetry: Dict[str, Any]