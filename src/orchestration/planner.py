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
    "You are an expert enterprise financial query planner. Your objective is to decompose complex user prompts "
    "into focused, isolated sub-queries needed to retrieve SEC 10-K data.\n\n"
    "CRITICAL FORMATTING RULE:\n"
    "NEVER use double quotes (\") inside string values. Use single quotes (') only.\n\n"
    "### DOMAIN KNOWLEDGE (SEC 10-K FILINGS)\n"
    "- Item 1A (Risk Factors): Qualitative business, market, and cybersecurity risks. No tables.\n"
    "- Item 7 (MD&A): Narrative explanations of results, segment performance, and growth drivers.\n"
    "- Item 8 (Financial Statements & Notes): Audited financial tables (Income Statement, Balance Sheet, Tax & Geographic Footnotes).\n\n"
    "### STRICT TUPLE INVENTORY RULE (NON-NEGOTIABLE)\n"
    "You MUST ONLY generate a sub-query if BOTH the ticker AND the fiscal_year exist together in the available inventory below:\n"
    f"{inventory_str}\n\n"
    "CRITICAL SUB-QUERY CONSTRAINTS:\n"
    "1. EMPTY ARRAY FOR MISSING DATA: If a ticker or fiscal year in the prompt is NOT in the inventory above, "
    "return an empty array: {\"sub_queries\": []}. Do not attempt to guess or substitute years.\n"
    "2. NO REGIONAL FRAGMENTATION: Never create separate sub-queries for individual regions (e.g., one for Americas, one for Europe). Group them into a single comprehensive search string.\n"
    "3. PURE SEMANTIC SEARCH STRINGS: DO NOT include company names (e.g., 'Apple', 'Microsoft') or years (e.g., '2024') "
    "inside the 'query' text field. Keep query strings focused purely on concepts.\n"
    "4. KEYWORD EXPANSION (CRITICAL):\n"
    "   - Geographic queries MUST use this exact string: 'United States International Americas Europe Greater China geographic segment net sales revenue'.\n"
    "   - Tax queries MUST use: 'effective tax rate provision for income taxes'.\n"
    "5. SECTION TARGETING:\n"
    "   - Geographic/segment queries: Generate TWO sub-queries per ticker (one targeting 'Item 7' for MD&A narrative, one targeting 'Item 8' for footnote tables).\n"
    "   - Tax queries: Target 'Item 8'.\n"
    "   - Risk queries: Target 'Item 1A'.\n\n"
    "FEW-SHOT EXAMPLES:\n\n"
    "Example 1 (Multi-Entity Geographic Comparison):\n"
    "User Prompt: 'Look at the geographic segment performance for both Apple and Microsoft in 2024.'\n"
    "Output: {\n"
    "  \"reasoning\": \"Decomposing geographic query into Item 7 and Item 8 for both tickers, using the expanded neutral keyword string.\",\n"
    "  \"sub_queries\": [\n"
    "    {\n"
    "      \"ticker\": \"AAPL\",\n"
    "      \"fiscal_year\": 2024,\n"
    "      \"item_number\": \"Item 7\",\n"
    "      \"query\": \"United States International Americas Europe Greater China geographic segment net sales revenue\"\n"
    "    },\n"
    "    {\n"
    "      \"ticker\": \"AAPL\",\n"
    "      \"fiscal_year\": 2024,\n"
    "      \"item_number\": \"Item 8\",\n"
    "      \"query\": \"United States International Americas Europe Greater China geographic segment net sales revenue\"\n"
    "    },\n"
    "    {\n"
    "      \"ticker\": \"MSFT\",\n"
    "      \"fiscal_year\": 2024,\n"
    "      \"item_number\": \"Item 7\",\n"
    "      \"query\": \"United States International Americas Europe Greater China geographic segment net sales revenue\"\n"
    "    },\n"
    "    {\n"
    "      \"ticker\": \"MSFT\",\n"
    "      \"fiscal_year\": 2024,\n"
    "      \"item_number\": \"Item 8\",\n"
    "      \"query\": \"United States International Americas Europe Greater China geographic segment net sales revenue\"\n"
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Example 2 (Accounting Metric Search Expansion):\n"
    "User Prompt: 'What were the exact effective tax rates for Microsoft in 2023 and 2024?'\n"
    "Output: {\n"
    "  \"reasoning\": \"Generating isolated sub-queries per year targeting Item 8 with tax footnote search expansion.\",\n"
    "  \"sub_queries\": [\n"
    "    {\n"
    "      \"ticker\": \"MSFT\",\n"
    "      \"fiscal_year\": 2023,\n"
    "      \"item_number\": \"Item 8\",\n"
    "      \"query\": \"effective tax rate provision for income taxes\"\n"
    "    },\n"
    "    {\n"
    "      \"ticker\": \"MSFT\",\n"
    "      \"fiscal_year\": 2024,\n"
    "      \"item_number\": \"Item 8\",\n"
    "      \"query\": \"effective tax rate provision for income taxes\"\n"
    "    }\n"
    "  ]\n"
    "}"
)

    client = instructor.from_groq(Groq())
    
    try:
        plan: QueryPlan = client.chat.completions.create(
            model="llama-3.1-8b-instant",
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