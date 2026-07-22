import sqlite3
import logging
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
import instructor
from groq import Groq

logger = logging.getLogger("PlannerNode")

class ExtractedSubQuery(BaseModel):
    query: str = Field(description="Focused search query string.")
    ticker: Optional[str] = Field(description="Exact ticker string from inventory, e.g., 'AAPL', 'MSFT'.")
    fiscal_year: Optional[int] = Field(description="Exact fiscal year integer from inventory, e.g., 2023, 2024.")
    # FIX: Strict constraint forces instructor to automatically retry if the LLM outputs "1A" instead of "Item 1A"
    item_number: Optional[Literal["Item 1A", "Item 7", "Item 8"]] = Field(
        description="Must be strictly one of 'Item 1A', 'Item 7', or 'Item 8'."
    )

class QueryPlan(BaseModel):
    sub_queries: List[ExtractedSubQuery]


def fetch_database_inventory(db_path: str = "parent_docstore.db") -> List[dict]:
    """Queries SQLite for distinct active document metadata tuples."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT ticker, fiscal_year, item_number FROM parent_documents")
        rows = cursor.fetchall()
        conn.close()
        return [{"ticker": r[0], "fiscal_year": r[1], "item_number": r[2]} for r in rows]
    except Exception as e:
        logger.error(f"Failed to fetch database inventory: {e}")
        return []


def plan_queries(state: dict) -> dict:
    """Planner Node: Decomposes raw query against strict database inventory."""
    raw_query = state["raw_query"]
    inventory = state.get("available_inventory") or fetch_database_inventory()
    
    inventory_str = "\n".join([f"- Ticker: {i['ticker']} | Year: {i['fiscal_year']} | Item: {i['item_number']}" for i in inventory])
    
    system_prompt = (
        "You are an expert enterprise financial query planner. Your job is to decompose complex user prompts "
        "into the minimum number of focused sub-queries needed to retrieve the data.\n\n"
        "### DOMAIN KNOWLEDGE (SEC FILINGS)\n"
        "- 'Item 1A' (Risk Factors): Contains qualitative business and market risks. No financial tables.\n"
        "- 'Item 7' (MD&A): Contains narrative explanations of financial results, year-over-year changes, segment performance, and macro factors (like currency fluctuations).\n"
        "- 'Item 8' (Financial Statements): Contains the raw, audited numerical tables (income statements, balance sheets).\n\n"
        "### STRICT INVENTORY RULE\n"
        "You must ONLY generate sub-queries matching the available data below:\n"
        f"{inventory_str}\n\n"
        "### DECOMPOSITION RULES\n"
        "1. DO NOT over-fragment. If a user asks for multiple regions' sales in a single year, keep it as ONE sub-query.\n"
        "2. If a query requires both numbers (Item 8/Item 7) and explanations (Item 7), generate a single sub-query targeting Item 7, as MD&A contains both.\n"
        "3. Ensure the 'query' field retains the FULL context of what the user is asking (e.g., include the requirement to explain currency impacts)."
    )

    client = instructor.from_groq(Groq())
    
    try:
        plan: QueryPlan = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            response_model=QueryPlan,
            max_retries=2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_query}
            ],
            temperature=0.0
        )
        sub_queries = [sq.model_dump() for sq in plan.sub_queries]
    except Exception as e:
        logger.error(f"Planner failed or output invalid schema: {e}")
        # Fallback to direct raw query with no filters
        sub_queries = [{
            "query": raw_query,
            "ticker": None,
            "fiscal_year": None,
            "item_number": None,
            "retrieval_mode": "HYBRID"
        }]

    return {
        "available_inventory": inventory,
        "sub_queries": sub_queries
    }