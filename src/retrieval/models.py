from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class QueryRequest(BaseModel):
    query: str
    top_k_chunks: int = 20
    top_k_parents: int = 3
    ticker: Optional[str] = None          
    fiscal_year: Optional[int] = None      
    item_number: Optional[str] = None      
    table_only: bool = False

class RetrievedContext(BaseModel):
    parent_id: str
    ticker: str          
    fiscal_year: int      
    item_number: str
    is_table: bool
    content: str

class AskResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    telemetry: Dict[str, Any]