import os
import time
import logging
import tiktoken
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from qdrant_client import AsyncQdrantClient
from fastembed import TextEmbedding, SparseTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from groq import AsyncGroq

from src.retrieval.models import QueryRequest, RetrievedContext, AskResponse
from src.retrieval.core import execute_core_retrieval

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class Infrastructure:
    qdrant: AsyncQdrantClient = None
    dense_model: TextEmbedding = None
    sparse_model: SparseTextEmbedding = None
    reranker: TextCrossEncoder = None
    llm_client: AsyncGroq = None

infra = Infrastructure()

@asynccontextmanager
async def lifespan(app: FastAPI):
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


@app.post("/retrieve", response_model=list[RetrievedContext])
async def retrieve_context(request: QueryRequest):
    try:
        return await execute_core_retrieval(request, infra)
    except Exception as e:
        logging.error(f"Retrieval Endpoint Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error during context retrieval.")


@app.post("/ask", response_model=AskResponse)
async def ask_rag(request: QueryRequest):
    start_time = time.time()
    
    retrieval_start = time.time()
    try:
        retrieved_parents = await execute_core_retrieval(request, infra)
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

    encoder = tiktoken.get_encoding("cl100k_base")
    TOKEN_CEILING = 8000 
    context_blocks = []
    sources = []
    total_tokens = 0
    truncated = False

    for parent in retrieved_parents:
        header = f"--- Document: {parent['ticker']} (FY {parent['fiscal_year']} - {parent['item_number']}) ---\n"
        full_text = parent['content']
        header_tokens = len(encoder.encode(header))
        available_tokens = TOKEN_CEILING - total_tokens - header_tokens - 500
        
        if available_tokens <= 0:
            break

        parent_token_ids = encoder.encode(full_text)
        
        if len(parent_token_ids) > available_tokens:
            truncated = True
            truncated_text = encoder.decode(parent_token_ids[:available_tokens])
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
        if truncated:
            break
            
    compiled_context = "\n".join(context_blocks)

    system_prompt = (
        "You are a strict financial analyst AI. Answer the user's question using ONLY the provided SEC 10-K context.\n\n"
        "RULES:\n"
        "1. If the context does not contain the specific metric or fact requested after thorough examination, "
        "state: 'I do not have sufficient data in the retrieved filings to answer this.'\n"
        "2. DO NOT invent facts or external data. However, you MAY group related risk concepts if explicitly mentioned.\n"
        "3. You MUST cite the source for every fact or metric: e.g., 'AAPL FY 2024 - Item 1A'.\n"
        "4. If the context includes HTML tables, extract the relevant data and present it in clean markdown tables.\n"
        "5. When the context contains multiple documents, compare them explicitly.\n"
        "6. Before concluding, examine both prose and tables."
    )
    
    if truncated:
        system_prompt += "\nNOTE: Context was truncated for safety. Answer as best as you can with the provided fragment."

    try:
        response = await infra.llm_client.chat.completions.create(
            model="openai/gpt-oss-120b",
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

    return AskResponse(
        query=request.query,
        answer=answer,
        sources=sources,
        telemetry={
            "total_pipeline_latency_seconds": round(time.time() - start_time, 3),
            "retrieval_latency_seconds": round(retrieval_latency, 3),
            "source_count": len(sources)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)