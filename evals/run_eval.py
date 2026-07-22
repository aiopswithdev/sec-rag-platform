import asyncio
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()
# Ensure the parent directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.orchestration.graph import orchestrator_app
from evals.graders.decomposition_grader import grade_decomposition
from evals.graders.refusal_grader import grade_refusal
from evals.graders.generation_grader import grade_generation

async def evaluate_harness(jsonl_path: str):
    print("==================================================")
    print("⚙️ INITIATING DEBUG-ENABLED COMPONENT EVALUATION")
    print("==================================================")
    
    with open(jsonl_path, 'r') as f:
        queries = [json.loads(line) for line in f]

    metrics = {
        "planner_recall": [], "planner_precision": [], "routing_accuracy": [],
        "presence_score": [], "groundedness_score": [], "citation_score": [],
        "adversarial_success": []
    }

    for case in queries:
        print(f"\n==================================================")
        print(f"🔍 EVALUATING [{case['id']}] ({case['category']})")
        print(f"Query: \"{case['query']}\"")
        print("==================================================")
        
        # 1. Execute Orchestrator
        state = {"raw_query": case["query"]}
        final_state = await orchestrator_app.ainvoke(state)
        
        actual_sub_queries = final_state.get("sub_queries", [])
        final_answer = final_state.get("final_answer", "")
        retrieved_contexts = final_state.get("retrieved_contexts", [])

        # --- DEBUG INSTRUMENTATION ---
        print("\n--- [DEBUG: PLANNER & ROUTER OUTPUT] ---")
        for sq in actual_sub_queries:
            print(f"  • Sub-Query: {sq.get('query')}")
            print(f"    Target: Ticker={sq.get('ticker')}, FY={sq.get('fiscal_year')}, Item={sq.get('item_number')}, Mode={sq.get('retrieval_mode')}")

        print(f"\n--- [DEBUG: EXECUTOR RETRIEVAL SUMMARY] ---")
        print(f"Total Context Blocks Retrieved: {len(retrieved_contexts)}")
        by_ticker = {}
        for ctx in retrieved_contexts:
            t = ctx.get("ticker", "UNKNOWN")
            is_tbl = "TABLE" if ctx.get("is_table") else "PROSE"
            by_ticker.setdefault(t, []).append(f"{is_tbl} (Item {ctx.get('item_number')})")
        for t, summary in by_ticker.items():
            print(f"  • {t}: {len(summary)} blocks -> {summary}")

        print(f"\n--- [DEBUG: GENERATED FINAL ANSWER] ---")
        print(final_answer if final_answer else "<EMPTY ANSWER>")
        print("----------------------------------------")

        # 2. Grade Refusal or Generation
        if case["category"] == "adversarial_refusal":
            success = grade_refusal(final_answer)
            metrics["adversarial_success"].append(1.0 if success else 0.0)
            print(f"  -> Adversarial Defense: {'PASS' if success else 'FAIL'}")
        else:
            decomp_scores = grade_decomposition(case["expected_sub_queries"], actual_sub_queries)
            metrics["planner_recall"].append(decomp_scores["recall"])
            metrics["planner_precision"].append(decomp_scores["precision"])
            metrics["routing_accuracy"].append(decomp_scores["routing_accuracy"])

            gen_scores = grade_generation(final_answer, case["expected_facts"], retrieved_contexts)
            metrics["presence_score"].append(gen_scores["presence_score"])
            metrics["groundedness_score"].append(gen_scores["groundedness_score"])
            metrics["citation_score"].append(gen_scores["citation_score"])
            
            print(f"\n  -> Decomposition Scores : Recall={decomp_scores['recall']:.2f}, Routing={decomp_scores['routing_accuracy']:.2f}")
            print(f"  -> Generation Scores    : Faithfulness={gen_scores['groundedness_score']:.2f}, Citations={gen_scores['citation_score']:.2f}")

    # Aggregation
    def avg(lst): return sum(lst) / len(lst) if lst else 0.0
    
    print("\n==================================================")
    print("📊 FINAL HARNESS METRICS ROLLUP")
    print("==================================================")
    print(f"Planner Tuple Recall:       {avg(metrics['planner_recall']):.2f}")
    print(f"Planner Tuple Precision:    {avg(metrics['planner_precision']):.2f}")
    print(f"Router Accuracy:            {avg(metrics['routing_accuracy']):.2f}")
    print(f"Answer Fact Presence:       {avg(metrics['presence_score']):.2f}")
    print(f"Context Groundedness:       {avg(metrics['groundedness_score']):.2f}")
    print(f"Citation Accuracy:          {avg(metrics['citation_score']):.2f}")
    print(f"Adversarial Refusal Rate:   {avg(metrics['adversarial_success']):.2f}")
    print("==================================================")

if __name__ == "__main__":
    golden_path = os.path.join(os.path.dirname(__file__), "golden_queries.jsonl")
    asyncio.run(evaluate_harness(golden_path))