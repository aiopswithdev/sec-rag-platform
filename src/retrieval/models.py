from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class QueryRequest(BaseModel):
    query: str
    top_k_chunks: int = 20
    top_k_parents: int = 5
    ticker: Optional[str] = None          
    fiscal_year: Optional[int] = None      
    item_number: Optional[str] = None      
    table_only: bool = False
class AskRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
class SubQueryTrace(BaseModel):
    query: str
    ticker: Optional[str] = None
    fiscal_year: Optional[int] = None
    item_number: Optional[str] = None
    retrieval_mode: Optional[str] = None    
class OrchestratedAskResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    sub_queries: List[SubQueryTrace]
    retrieval_errors: List[str] = []
    telemetry: Dict[str, Any]
class RetrievedContext(BaseModel):
    parent_id: str
    ticker: str          
    fiscal_year: int      
    item_number: str
    is_table: bool
    score: float
    content: str

# class AskResponse(BaseModel):
#     query: str
#     answer: str
#     sources: List[Dict[str, Any]]
#     telemetry: Dict[str, Any]