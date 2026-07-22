import asyncio
import json
import logging
from dotenv import load_dotenv

# Load environment variables from .env before importing orchestration code
load_dotenv()
from src.orchestration.graph import orchestrator_app

# Disable noisy HTTP client logs for the test output
logging.getLogger("httpx").setLevel(logging.WARNING)

async def run_sanity_check():
    # This query forces HYBRID routing (numbers + qualitative explanation)
    # and tests the Planner's ability to extract the correct years and tickers.
    complex_query = (
        "For AAPL, extract the segment net sales for Americas, Europe, and Greater China "
        "for fiscal years 2023 and 2024, and explain the currency fluctuations impacting these changes."
    )
    
    print("==================================================")
    print("🚀 INITIATING STAFF-LEVEL ORCHESTRATION TEST")
    print("==================================================")
    print(f"RAW QUERY: {complex_query}\n")
    
    initial_state = {"raw_query": complex_query}
    
    try:
        # We use astream to observe the state mutations dynamically as each node completes
        async for output in orchestrator_app.astream(initial_state):
            for node_name, state_update in output.items():
                print(f"\n[ NODE COMPLETE: {node_name.upper()} ]".ljust(50, "-"))
                
                if node_name == "planner":
                    sub_queries = state_update.get('sub_queries', [])
                    print(f"Extracted {len(sub_queries)} sub-queries targeting exact inventory:")
                    print(json.dumps(sub_queries, indent=2))
                    
                elif node_name == "router":
                    print("Ternary Routing Decisions Applied:")
                    for sq in state_update.get("sub_queries", []):
                        mode = sq.get("retrieval_mode", "UNKNOWN")
                        print(f" -> [{mode}] {sq['query']}")
                        
                elif node_name == "executor":
                    ctx = state_update.get("retrieved_contexts", [])
                    errs = state_update.get("retrieval_errors", [])
                    print(f"Successfully aggregated {len(ctx)} parent context windows via HTTP fan-out.")
                    if errs:
                        print(f"⚠️ Retrieval Errors Encountered: {errs}")
                        
                elif node_name == "synthesis":
                    print("\n==================================================")
                    print("FINAL SYNTHESIS OUTPUT")
                    print("==================================================\n")
                    print(state_update.get("final_answer"))
                    print("\n--- Telemetry ---")
                    print(json.dumps(state_update.get("telemetry"), indent=2))
                    
    except Exception as e:
        print(f"\n[X] Orchestration failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_sanity_check())