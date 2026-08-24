import asyncio
import json
import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.orchestration.graph import orchestrator_app
from evals.graders.decomposition_grader import grade_decomposition
from evals.graders.refusal_grader import grade_refusal
from evals.graders.generation_grader import grade_generation

# Import the production infrastructure and lifespan manager from the API
from src.retrieval.api import lifespan, infra
from src.retrieval.core import execute_core_retrieval
from src.retrieval.models import QueryRequest

async def _local_retriever(payload: dict) -> list[dict]:
    """Adapter: executor payload dict -> in-process core retrieval."""
    return await execute_core_retrieval(QueryRequest(**payload), infra)

def _print_debug_trace(state: dict):
    """Encapsulates verbose debug printing to keep the main loop clean."""
    print("\n--- [DEBUG: PLANNER & ROUTER OUTPUT] ---")
    for sq in state.get("sub_queries", []):
        print(f"  • Sub-Query: {sq.get('query')}")
        print(f"    Target: Ticker={sq.get('ticker')}, FY={sq.get('fiscal_year')}, Item={sq.get('item_number')}, Mode={sq.get('retrieval_mode')}")

    print("\n--- [DEBUG: EXECUTOR RETRIEVAL SUMMARY] ---")
    retrieved = state.get("retrieved_contexts", [])
    print(f"Total Context Blocks Retrieved: {len(retrieved)}")
    
    by_ticker = {}
    for ctx in retrieved:
        t = ctx.get("ticker", "UNKNOWN")
        is_tbl = "TABLE" if ctx.get("is_table") else "PROSE"
        by_ticker.setdefault(t, []).append(f"{is_tbl} (Item {ctx.get('item_number')})")
        
    for t, summary in by_ticker.items():
        print(f"  • {t}: {len(summary)} blocks -> {summary}")

    print("\n--- [DEBUG: GENERATED FINAL ANSWER] ---")
    print(state.get("final_answer", "<EMPTY ANSWER>"))
    print("-" * 40)

async def evaluate_harness(jsonl_path: str):
    print("=" * 50 + "\n⚙️ INITIATING E2E COMPONENT EVALUATION\n" + "=" * 50)
    
    with open(jsonl_path, 'r') as f:
        queries = [json.loads(line) for line in f]

    metrics = {k: [] for k in [
        "planner_recall", "planner_precision", "routing_accuracy",
        "presence_score", "groundedness_score", "citation_score", "adversarial_success"
    ]}

    # Boot up the exact same infrastructure used by the live FastAPI server
    dummy_app = FastAPI()
    async with lifespan(dummy_app):
        for case in queries:
            print(f"\n" + "=" * 50)
            print(f"🔍 EVALUATING [{case['id']}] ({case['category']})")
            print(f"Query: \"{case['query']}\"")
            print("=" * 50)
            
            # 1. Execute Orchestrator WITH the injected local retriever
            final_state = await orchestrator_app.ainvoke(
                {"raw_query": case["query"]},
                config={"configurable": {"retriever": _local_retriever}}
            )
            
            # 2. Print Trace
            _print_debug_trace(final_state)

            # 3. Grade Results
            if case["category"] == "adversarial_refusal":
                success = grade_refusal(final_state.get("final_answer", ""))
                metrics["adversarial_success"].append(1.0 if success else 0.0)
                print(f"  -> Adversarial Defense: {'PASS' if success else 'FAIL'}")
            else:
                decomp_scores = grade_decomposition(case["expected_sub_queries"], final_state.get("sub_queries", []))
                gen_scores = grade_generation(final_state.get("final_answer", ""), case["expected_facts"], final_state.get("retrieved_contexts", []))
                
                metrics["planner_recall"].append(decomp_scores["recall"])
                metrics["planner_precision"].append(decomp_scores["precision"])
                metrics["routing_accuracy"].append(decomp_scores["routing_accuracy"])
                metrics["presence_score"].append(gen_scores["presence_score"])
                metrics["groundedness_score"].append(gen_scores["groundedness_score"])
                metrics["citation_score"].append(gen_scores["citation_score"])
                
                print(f"\n  -> Decomposition Scores : Recall={decomp_scores['recall']:.2f}, Routing={decomp_scores['routing_accuracy']:.2f}")
                print(f"  -> Generation Scores    : Faithfulness={gen_scores['groundedness_score']:.2f}, Citations={gen_scores['citation_score']:.2f}")

    # Aggregation & Rollup
    avg = lambda lst: sum(lst) / len(lst) if lst else 0.0
    
    print("\n" + "=" * 50 + "\n📊 FINAL HARNESS METRICS ROLLUP\n" + "=" * 50)
    print(f"Planner Tuple Recall:       {avg(metrics['planner_recall']):.2f}")
    print(f"Planner Tuple Precision:    {avg(metrics['planner_precision']):.2f}")
    print(f"Router Accuracy:            {avg(metrics['routing_accuracy']):.2f}")
    print(f"Answer Fact Presence:       {avg(metrics['presence_score']):.2f}")
    print(f"Context Groundedness:       {avg(metrics['groundedness_score']):.2f}")
    print(f"Citation Accuracy:          {avg(metrics['citation_score']):.2f}")
    print(f"Adversarial Refusal Rate:   {avg(metrics['adversarial_success']):.2f}")
    print("=" * 50)

if __name__ == "__main__":
    golden_path = os.path.join(os.path.dirname(__file__), "golden_queries.jsonl")
    asyncio.run(evaluate_harness(golden_path))