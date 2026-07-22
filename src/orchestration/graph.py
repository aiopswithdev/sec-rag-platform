from langgraph.graph import StateGraph, START, END
from src.orchestration.state import AgentState
from src.orchestration.planner import plan_queries
from src.orchestration.router import route_sub_queries
from src.orchestration.executor import execute_retrieval
from src.orchestration.synthesis import synthesize_answer

def build_orchestration_graph():
    """Assembles the LangGraph StateMachine topology."""
    workflow = StateGraph(AgentState)

    # 1. Register Nodes
    workflow.add_node("planner", plan_queries)
    workflow.add_node("router", route_sub_queries)
    workflow.add_node("executor", execute_retrieval)
    workflow.add_node("synthesis", synthesize_answer)

    # 2. Wire Linear Topology
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "router")
    workflow.add_edge("router", "executor")
    workflow.add_edge("executor", "synthesis")
    workflow.add_edge("synthesis", END)

    # 3. Compile Graph
    return workflow.compile()

# Instantiated graph executable
orchestrator_app = build_orchestration_graph()