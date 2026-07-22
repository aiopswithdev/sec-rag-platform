import logging
from typing import Literal
from pydantic import BaseModel, Field
import instructor
from groq import Groq

logger = logging.getLogger("RouterNode")

class RoutingDecision(BaseModel):
    retrieval_mode: Literal["PROSE", "TABLE", "HYBRID"] = Field(
        description="'TABLE' for exact financial statement metrics/numbers. 'PROSE' for qualitative commentary/risk disclosures. 'HYBRID' for questions requiring both."
    )
    reasoning: str = Field(description="Brief justification for the chosen retrieval mode.")

def route_sub_queries(state: dict) -> dict:
    """Router Node: Classifies each sub-query into PROSE, TABLE, or HYBRID mode."""
    sub_queries = state["sub_queries"]
    client = instructor.from_groq(Groq())
    
    updated_sub_queries = []
    
    for sq in sub_queries:
        prompt = (
            f"Classify the optimal retrieval mode for this sub-query:\n"
            f"Query: '{sq['query']}'\n"
            f"Target Section: {sq.get('item_number')}\n\n"
            "Examples:\n"
            "- 'net sales or total revenue numbers' -> TABLE\n"
            "- 'supply chain risk disclosures or legal proceedings' -> PROSE\n"
            "- 'what factors drove the change in operating margin' -> HYBRID"
        )
        
        try:
            decision: RoutingDecision = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                response_model=RoutingDecision,
                max_retries=2,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            sq["retrieval_mode"] = decision.retrieval_mode
            logger.info(f"Sub-query routed to [{decision.retrieval_mode}]: {sq['query']} | Reason: {decision.reasoning}")
        except Exception as e:
            logger.error(f"Router failed for sub-query '{sq['query']}': {e}")
            sq["retrieval_mode"] = "HYBRID"  # Safe fallback
            
        updated_sub_queries.append(sq)
        
    return {"sub_queries": updated_sub_queries}