import logging
from typing import Literal
from pydantic import BaseModel, Field
import instructor
from groq import Groq
import asyncio
from groq import AsyncGroq

logger = logging.getLogger("RouterNode")

class RoutingDecision(BaseModel):
    retrieval_mode: Literal["PROSE", "TABLE", "HYBRID"] = Field(
        description="'TABLE' for financial metrics/tables. 'PROSE' for qualitative text/risks. 'HYBRID' for segment performance and MD&A drivers."
    )
    reasoning: str = Field(
        description="Brief justification. CRITICAL: NEVER use double quotes (\") inside this string; use single quotes (') only."
    )

system_prompt = (
        "You are a strict SEC 10-K query router. Classify the query into 'PROSE', 'TABLE', or 'HYBRID'.\n\n"
        "CRITICAL FORMATTING RULE:\n"
        "Do NOT use double quotes (\") inside the reasoning field. Use single quotes (') only.\n\n"
        "CLASSIFICATION RULES:\n"
        "1. 'PROSE': Qualitative text, Item 1A risk factors, Item 1C cybersecurity disclosures, legal proceedings.\n"
        "2. 'TABLE': Exact financial statement metrics, total net sales, revenue, effective tax rate, provision for income taxes, balance sheet, income statement, Item 8 tables.\n"
        "3. 'HYBRID': Geographic or Product segment performance, MD&A drivers, narrative explanations of numerical changes, Item 7 growth reasons.\n\n"
        "EXAMPLES:\n"
        "- 'total net sales or total revenue' -> TABLE\n"
        "- 'effective tax rate or provision for income taxes' -> TABLE\n"
        "- 'primary cybersecurity risk factors or business risks' -> PROSE\n"
        "- 'geographic segment net sales and reasons for growth' -> HYBRID"
    )

_router_client = instructor.from_groq(AsyncGroq(), mode=instructor.Mode.JSON)  # module-level

async def _route_one(sq: dict, sem: asyncio.Semaphore) -> dict:
    sq_copy = sq.copy()
    async with sem:
        try:
            decision = await _router_client.chat.completions.create(
                model="openai/gpt-oss-20b", response_model=RoutingDecision,
                max_retries=2, temperature=0.0,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": f"Sub-Query: '{sq['query']}' | Target Section: {sq.get('item_number', 'UNKNOWN')}"}],
            )
            sq_copy["retrieval_mode"] = decision.retrieval_mode
        except Exception as e:
            logger.error(f"Router failed for '{sq['query']}': {e}")
            sq_copy["retrieval_mode"] = "HYBRID"
    return sq_copy

async def route_sub_queries(state: dict) -> dict:
    """Router Node: Classifies each sub-query into PROSE, TABLE, or HYBRID retrieval modes."""
    sub_queries = state.get("sub_queries", [])
    if not sub_queries:
        return {"sub_queries": []}

    sem = asyncio.Semaphore(4)
    updated_sub_queries = await asyncio.gather(*(_route_one(sq, sem) for sq in sub_queries))
    return {"sub_queries": list(updated_sub_queries)}