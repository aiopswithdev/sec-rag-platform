import os
import sqlite3
import logging
import asyncio
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Prefetch, 
    Fusion, 
    FusionQuery, 
    SparseVector, 
    Filter, 
    FieldCondition, 
    MatchValue
)
from fastembed import TextEmbedding, SparseTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from groq import AsyncGroq

# ---------------------------------------------------------------------------
# Environment Variable Auto-Loading
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
    logging.info(">>> [ENV] Loaded environment variables from local .env file.")
except ImportError:
    logging.warning(">>> [ENV] python-dotenv not installed. Falling back to system environment variables.")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ---------------------------------------------------------------------------
# Global Infrastructure State
# ---------------------------------------------------------------------------
class Infrastructure:
    qdrant: AsyncQdrantClient = None
    dense_model: TextEmbedding = None
    sparse_model: SparseTextEmbedding = None
    reranker: TextCrossEncoder = None
    llm_client: AsyncGroq = None

infra = Infrastructure()
COLLECTION_NAME = "advanced_sec_edgar_production"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize asynchronous connections and models on startup
    logging.info(">>> [INIT] Connecting to Qdrant (Async)...")
    api_key = os.getenv("QDRANT_API_KEY")
    infra.qdrant = AsyncQdrantClient(host="localhost", port=6333, api_key=api_key, https=False)
    
    logging.info(">>> [INIT] Loading Embedding Models...")
    infra.dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    infra.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    
    logging.info(">>> [INIT] Loading Cross-Encoder (Reranker)...")
    infra.reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
    
    logging.info(">>> [INIT] Initializing Asynchronous Groq Client...")
    infra.llm_client = AsyncGroq()
    
    logging.info(">>> [READY] FastAPI Retrieval Service Online.")
    yield
    logging.info(">>> [SHUTDOWN] Closing connections.")

app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    top_k_chunks: int = 20
    top_k_parents: int = 3
    ticker: str = None          
    fiscal_year: int = None      
    item_number: str = None      
    table_only: bool = False   # NEW

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
    sources: list[dict]
    telemetry: dict
    
# ---------------------------------------------------------------------------
# Retrieval Logic
# ---------------------------------------------------------------------------
def hydrate_parents_from_sql(parent_ids: list[str]) -> list[dict]:
    """Fetch unabridged parent documents from SQLite."""
    if not parent_ids:
        return []
    
    conn = sqlite3.connect("parent_docstore.db")
    cursor = conn.cursor()
    
    placeholders = ",".join(["?"] * len(parent_ids))
    query = f"SELECT id, ticker, fiscal_year, item_number, is_table, full_text FROM parent_documents WHERE id IN ({placeholders})"
    
    cursor.execute(query, parent_ids)
    rows = cursor.fetchall()
    conn.close()
    
    fetched_ids = {row[0] for row in rows}
    missing_ids = set(parent_ids) - fetched_ids
    if missing_ids:
        logging.warning(f"Data Integrity Warning: Parent IDs missing in SQLite: {missing_ids}")
    
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

# ---------------------------------------------------------------------------
# Core Retrieval Implementation (Decoupled Shared Engine)
# ---------------------------------------------------------------------------
async def execute_core_retrieval(request: QueryRequest) -> list[dict]:
    """Encapsulates the isolated hybrid search, reranking, and SQL hydration pipeline."""
    # 1. Build Metadata Filters dynamically per request
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

    # 2. Embed the query (Offloaded to avoid blocking the event loop)
    dense_query = await asyncio.to_thread(lambda: list(infra.dense_model.embed([request.query]))[0].tolist())
    sparse_query_result = await asyncio.to_thread(lambda: list(infra.sparse_model.embed([request.query]))[0])
    
    sparse_vector = SparseVector(
        indices=sparse_query_result.indices.tolist(),
        values=sparse_query_result.values.tolist()
    )

    # 3. Hybrid Search using Qdrant Prefetch & RRF
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

    # 4. Cross-Encoder Reranking
    chunk_texts = [hit.payload["text"] for hit in hits]
    rerank_scores = await asyncio.to_thread(lambda: list(infra.reranker.rerank(request.query, chunk_texts)))
    
    scored_hits = sorted(zip(hits, rerank_scores), key=lambda x: x[1], reverse=True)
    
    # 5. Extract and Deduplicate Parent IDs
    unique_parent_ids = []
    for hit, score in scored_hits:
        parent_id = hit.payload.get("parent_id")
        if parent_id and parent_id not in unique_parent_ids:
            unique_parent_ids.append(parent_id)
        if len(unique_parent_ids) >= request.top_k_parents:
            break
            
    # 6. SQL Hydration
    return await asyncio.to_thread(hydrate_parents_from_sql, unique_parent_ids)

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.post("/retrieve", response_model=list[RetrievedContext])
async def retrieve_context(request: QueryRequest):
    try:
        return await execute_core_retrieval(request)
    except Exception as e:
        logging.error(f"Retrieval Endpoint Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error during context retrieval.")
# ---------------------------------------------------------------------------
# LLM Generation Logic
# ---------------------------------------------------------------------------
@app.post("/ask", response_model=AskResponse)
async def ask_rag(request: QueryRequest):
    start_time = time.time()
    
    # 1. Retrieval Phase with Independent Telemetry Tracking
    retrieval_start = time.time()
    try:
        retrieved_parents = await execute_core_retrieval(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database retrieval failed: {str(e)}")
    retrieval_latency = time.time() - retrieval_start

    if not retrieved_parents:
        return AskResponse(
            query=request.query,
            answer="I do not have sufficient data in the retrieved filings to answer this.",
            sources=[],
            telemetry={
                "total_pipeline_latency_seconds": round(time.time() - start_time, 3),
                "retrieval_latency_seconds": round(retrieval_latency, 3),
                "source_count": 0
            }
        )

    # 2. Context Assembly & Token Guard Phase
    import tiktoken
    
    # Initialize tiktoken encoder (using cl100k_base as a reliable, fast proxy)
    encoder = tiktoken.get_encoding("cl100k_base")
    
    TOKEN_CEILING = 8000 
    context_blocks = []
    sources = []
    total_tokens = 0
    truncated = False

    for parent in retrieved_parents:
        header = f"--- Document: {parent['ticker']} (FY {parent['fiscal_year']} - {parent['item_number']}) ---\n"
        full_text = parent['content']
        
        # Calculate base tokens
        header_tokens = len(encoder.encode(header))
        
        # Reserve 500 tokens for the system prompt, question, and JSON structural overhead
        available_tokens = TOKEN_CEILING - total_tokens - header_tokens - 500
        
        # If we have no room left for even a partial block, stop adding
        if available_tokens <= 0:
            break

        parent_token_ids = encoder.encode(full_text)
        
        # If the parent text exceeds our remaining budget, execute safe truncation
        if len(parent_token_ids) > available_tokens:
            truncated = True
            
            # Decode only the tokens that fit within the strict limit
            truncated_text = encoder.decode(parent_token_ids[:available_tokens])
            
            # Attempt smart truncation at the last period to avoid cutting mid-sentence
            last_period = truncated_text.rfind('. ')
            if last_period > len(truncated_text) // 2:
                truncated_text = truncated_text[:last_period+1]
                
            truncated_text += "\n\n[...TRUNCATED DUE TO CONTEXT LIMIT...]"
            added_tokens = len(encoder.encode(truncated_text))
        else:
            truncated_text = full_text
            added_tokens = len(parent_token_ids)

        block = header + truncated_text + "\n"
        context_blocks.append(block)
        
        sources.append({
            "ticker": parent["ticker"], 
            "fiscal_year": parent["fiscal_year"],
            "item_number": parent["item_number"],
            "parent_id": parent["parent_id"],
            "is_table": parent["is_table"]
        })
        
        total_tokens += (header_tokens + added_tokens)
        
        # If we truncated this block, the ceiling is hit and we must stop iteration
        if truncated:
            break
            
    compiled_context = "\n".join(context_blocks)

    # 3. System Prompt Engineering (Hardened with Advanced Table & Multi-Doc Comparators)
    system_prompt = (
        "You are a strict financial analyst AI. Answer the user's question using ONLY the provided SEC 10-K context.\n\n"
        "RULES:\n"
        "1. If the context does not contain the specific metric or fact requested after thorough examination, "
        "state: 'I do not have sufficient data in the retrieved filings to answer this.'\n"
        "2. DO NOT invent facts or external data. However, you MAY group related risk concepts (e.g., logistics, "
        "component sourcing, manufacturing locations) if they are explicitly mentioned.\n"
        "3. You MUST cite the source for every fact or metric: e.g., 'AAPL FY 2024 - Item 1A'.\n"
        "4. If the context includes HTML tables, extract the relevant data and present it in clean markdown tables. "
        "Do not output raw HTML tags.\n"
        "5. When the context contains multiple documents (different years or tickers), compare them "
        "explicitly. Highlight differences using bullet points or a comparison table.\n"
        "6. Before concluding, examine both prose and tables"
    )
    
    if truncated:
        system_prompt += "\nNOTE: Context was truncated for safety. Answer as best as you can with the provided fragment."

    # 4. LLM Generation Phase via Groq
    try:
        response = await infra.llm_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{compiled_context}\n\nQuestion: {request.query}"}
            ]
        )
        answer = response.choices[0].message.content
        
    except Exception as e:
        logging.error(f"LLM Generation Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"LLM Generation failed: {str(e)}")

    total_time = time.time() - start_time

    return AskResponse(
        query=request.query,
        answer=answer,
        sources=sources,
        telemetry={
            "total_pipeline_latency_seconds": round(total_time, 3),
            "retrieval_latency_seconds": round(retrieval_latency, 3),
            "source_count": len(sources)
        }
    )

if __name__ == "__main__":
    import uvicorn
    # Bind to 0.0.0.0 to make it publicly accessible outside the VPC if rules allow
    uvicorn.run(app, host="0.0.0.0", port=8000)
