import os
import json
import time
import uuid
import asyncio
import logging
from contextlib import asynccontextmanager
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from fastapi import FastAPI, HTTPException, Security, status, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
from qdrant_client import AsyncQdrantClient
from fastembed import TextEmbedding, SparseTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from groq import AsyncGroq

from src.retrieval.models import QueryRequest, RetrievedContext, AskRequest, OrchestratedAskResponse
from src.retrieval.core import execute_core_retrieval
from src.orchestration.graph import orchestrator_app


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

API_KEY = os.getenv("API_KEY")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def require_api_key(key: str = Security(_api_key_header)):
    if not API_KEY:
        raise RuntimeError("API_KEY is not set — refusing to serve requests with auth disabled.")
    if key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key.")

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

ASK_TIMEOUT = float(os.getenv("ASK_TIMEOUT", "180"))
_ask_gate = asyncio.Semaphore(int(os.getenv("ASK_MAX_CONCURRENCY", "4")))

async def _local_retriever(payload: dict) -> list[dict]:
    """Adapter: executor payload dict -> in-process core retrieval."""
    return await execute_core_retrieval(QueryRequest(**payload), infra)

@app.post("/retrieve", response_model=list[RetrievedContext], dependencies=[Depends(require_api_key)])
async def retrieve_context(request: QueryRequest):
    try:
        return await execute_core_retrieval(request, infra)
    except Exception as e:
        logging.error(f"Retrieval Endpoint Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error during context retrieval.")

@app.post("/ask", response_model=OrchestratedAskResponse, dependencies=[Depends(require_api_key)])
async def ask_orchestrated(request: AskRequest):
    started = time.perf_counter()
    trace_id = str(uuid.uuid4())

    async with _ask_gate:
        try:
            final_state = await asyncio.wait_for(
                orchestrator_app.ainvoke(
                    {"raw_query": request.query},
                    config={"configurable": {"retriever": _local_retriever},
                            "run_id": trace_id},
                ),
                timeout=ASK_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise HTTPException(504, f"Orchestration exceeded {ASK_TIMEOUT}s")
        except Exception as e:
            logging.exception("Orchestration failure")
            raise HTTPException(502, f"Orchestration failed: {e}")

    node_telemetry = final_state.get("telemetry", {}) or {}
    return OrchestratedAskResponse(
        query=request.query,
        answer=final_state.get("final_answer", ""),
        sources=final_state.get("sources", []),
        sub_queries=final_state.get("sub_queries", []),
        retrieval_errors=final_state.get("retrieval_errors", []),
        telemetry={
            **node_telemetry,
            "trace_id": trace_id,
            "total_latency_seconds": round(time.perf_counter() - started, 3),
            "sub_query_count": len(final_state.get("sub_queries", [])),
            "context_block_count": len(final_state.get("retrieved_contexts", [])),
        },
    )

@app.post("/ask/stream", dependencies=[Depends(require_api_key)])
async def ask_stream(request: AskRequest):
    async def gen():
        async for chunk in orchestrator_app.astream(
            {"raw_query": request.query},
            config={"configurable": {"retriever": _local_retriever}},
        ):
            for node, update in chunk.items():
                payload = {"node": node}
                if node == "planner":   
                    payload["sub_queries"] = update.get("sub_queries", [])
                elif node == "router":  
                    payload["routes"] = [{"q": s["query"], "mode": s.get("retrieval_mode")} for s in update.get("sub_queries", [])]
                elif node == "executor":
                    payload["context_count"] = len(update.get("retrieved_contexts", []))
                elif node == "synthesis":
                    payload |= {
                        "answer": update.get("final_answer"),
                        "sources": update.get("sources", []),
                        "telemetry": update.get("telemetry", {})
                    }
                yield f"data: {json.dumps(payload)}\n\n"
        yield "data: {\"node\": \"done\"}\n\n"
        
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)