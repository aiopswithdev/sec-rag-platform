import os
import re
import time
import sqlite3
import logging
from functools import lru_cache
from typing import List, Optional, Literal, Set, Tuple
from pydantic import BaseModel, Field
import instructor
from groq import Groq

from dotenv import load_dotenv
load_dotenv()

from src.common.rate_limiter import planner_limiter

logger = logging.getLogger("PlannerNode")

_planner_client = instructor.from_groq(Groq(), mode=instructor.Mode.TOOLS)

class ExtractedSubQuery(BaseModel):
    query: str = Field(
        default="",
        description="Free-text search string — ONLY used when query_category is 'OTHER' "
                    "(e.g. a specific risk topic like cybersecurity or supply chain). Ignored for every other category."
    )
    query_category: Literal["TOTAL_METRIC", "GEO_NUMBERS", "GEO_DRIVERS", "TAX_RATE", "OTHER"] = Field(
        description="'TOTAL_METRIC' for a single company-wide revenue/sales figure. "
                    "'GEO_NUMBERS' for geographic segment dollar figures. "
                    "'GEO_DRIVERS' for geographic segment growth explanations. "
                    "'TAX_RATE' for effective tax rate / tax provision questions. "
                    "'OTHER' for anything else (e.g. risk factors) — in which case 'query' must contain a focused free-text search string."
    )
    ticker: Optional[str] = Field(default=None, description="Exact ticker string from inventory, e.g., 'AAPL', 'MSFT'.")
    fiscal_year: Optional[int] = Field(default=None, description="Exact fiscal year integer from inventory, e.g., 2023, 2024.")
    item_number: Optional[Literal["Item 1A", "Item 7", "Item 8"]] = Field(
        default=None, description="Must be strictly one of 'Item 1A', 'Item 7', or 'Item 8'."
    )

# Resolved by code, never retyped by the model — this is the fix.
CANONICAL_QUERIES = {
    "TOTAL_METRIC": "total net sales total revenue consolidated statements of operations",
    "GEO_NUMBERS": "United States International Americas Europe Greater China geographic segment net sales revenue",
    "GEO_DRIVERS": "reasons drivers factors for segment revenue growth decline foreign currency impact",
    "TAX_RATE": "effective tax rate reconciliation federal statutory rate provision for income taxes income tax expense",
}

class QueryPlan(BaseModel):
    reasoning: str = Field(description="Brief plan justification. Use single quotes only, never double quotes.")
    sub_queries: List[ExtractedSubQuery]

@lru_cache(maxsize=1)
def fetch_valid_company_years(db_path: str = "parent_docstore.db") -> Set[Tuple[str, int]]:
    if not os.path.exists(db_path):
        logger.error(f"Database file not found at path: {db_path}")
        return set()
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT ticker, fiscal_year FROM parent_documents")
        rows = cursor.fetchall()
        conn.close()
        return {(r[0], r[1]) for r in rows}
    except Exception as e:
        logger.error(f"Failed to fetch database inventory: {e}")
        return set()

def fetch_database_inventory(db_path: str = "parent_docstore.db") -> List[dict]:
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


def _extract_failed_generation(exc) -> Optional[str]:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        raw = (body.get("error") or {}).get("failed_generation")
        if raw:
            return raw
    m = re.search(r"'failed_generation':\s*'(.*)'\}\}", str(exc), re.DOTALL)
    return m.group(1) if m else None


def _salvage_failed_generation(exc) -> Optional[QueryPlan]:
    """gpt-oss-20b intermittently fails to produce a clean tool call under
    tool_choice='required'. Two shapes observed so far:
      A) The plan is returned as plain top-level JSON, bypassing the tool-call
         envelope entirely (400 'Tool choice is required, but model did not call a tool').
      B) The plan IS wrapped as {"name": "QueryPlan", "arguments": {...}} but has
         stray trailing tokens appended after the closing brace (400 tool_use_failed).
    Either way the correct plan is sitting right there in the error body — recover it
    instead of discarding a plan that was already right."""
    raw = _extract_failed_generation(exc)
    if not raw:
        return None
    raw = raw.strip()

    # Shape A: plain JSON at the top level.
    try:
        return QueryPlan.model_validate_json(raw)
    except Exception:
        pass

    # Shape B: wrapped in a tool-call envelope with trailing garbage — brace-balance
    # from the "arguments" object to recover just the valid JSON span.
    idx = raw.find('"arguments"')
    if idx == -1:
        return None
    start = raw.find('{', idx)
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == '{':
            depth += 1
        elif raw[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return QueryPlan.model_validate_json(raw[start:i + 1])
                except Exception as parse_err:
                    logger.warning(f"Salvage parse failed: {parse_err}")
                    return None
    return None


def plan_queries(state: dict) -> dict:
    """Planner Node: Decomposes raw query against strict database inventory."""
    raw_query = state["raw_query"]
    valid_inventory = fetch_valid_company_years()
    if not valid_inventory:
        raise RuntimeError("Parent docstore inventory is empty — check DOCSTORE_PATH / run ingestion.")

    system_prompt = (
    "You are an expert enterprise financial query planner. Your objective is to decompose complex user prompts "
    "into focused, isolated sub-queries needed to retrieve SEC 10-K data.\n\n"
    "CRITICAL FORMATTING RULE:\n"
    "NEVER use double quotes (\") inside string values. Use single quotes (') only.\n\n"
    "### DOMAIN KNOWLEDGE (SEC 10-K FILINGS)\n"
    "- Item 1A (Risk Factors): Qualitative business, market, and cybersecurity risks. No tables.\n"
    "- Item 7 (MD&A): Narrative explanations of results, segment performance, and growth drivers.\n"
    "- Item 8 (Financial Statements & Notes): Audited financial tables (Income Statement, Balance Sheet, Tax & Geographic Footnotes).\n\n"
    "CRITICAL SUB-QUERY CONSTRAINTS:\n"
    "1. If you cannot confidently identify a specific ticker and fiscal year from the user's prompt, "
    "return an empty array: {\"sub_queries\": []}. Do not guess or substitute years.\n"
    "2. NO REGIONAL FRAGMENTATION: Never create separate sub-queries for individual regions (e.g., one for Americas, one for Europe). Group them into a single comprehensive search string.\n"
    "3. PURE SEMANTIC SEARCH STRINGS: DO NOT include company names (e.g., 'Apple', 'Microsoft') or years (e.g., '2024') "
    "inside the 'query' text field. Keep query strings focused purely on concepts. Do NOT wrap the query text in quote characters.\n"
    "4. KEYWORD EXPANSION (CRITICAL):\n"
    "   - TOTAL / company-wide revenue queries MUST use: total net sales total revenue consolidated statements of operations.\n"
    "   - Geographic NUMBERS query (for Item 8): United States International Americas Europe Greater China geographic segment net sales revenue.\n"
    "   - Geographic DRIVERS query (for Item 7): reasons drivers factors for segment revenue growth decline foreign currency impact.\n"
    "   - Tax queries MUST use: effective tax rate provision for income taxes.\n"
    # planner.py, KEYWORD EXPANSION section
"   - Tax queries MUST use: effective tax rate reconciliation federal statutory rate income tax expense.\n"
    "5. SECTION TARGETING:\n"
    "   - TOTAL revenue / single income-statement metrics: generate exactly ONE sub-query per ticker-year targeting 'Item 8'. Do NOT add an Item 7 sub-query.\n"
    "   - Geographic/segment queries ONLY: generate TWO sub-queries per ticker — 'Item 7' with the DRIVERS string, 'Item 8' with the NUMBERS string.\n"
    "   - Tax queries: Target 'Item 8'.\n"
    "   - Risk queries: Target 'Item 1A'.\n\n"
    "FEW-SHOT EXAMPLES:\n\n"
    "Example 1 (Multi-Entity Geographic Comparison):\n"
    "User Prompt: 'Look at the geographic segment performance for both Apple and Microsoft in 2024.'\n"
    "Output: {\n"
    "  \"reasoning\": \"One GEO_DRIVERS sub-query (Item 7) and one GEO_NUMBERS sub-query (Item 8) per ticker.\",\n"
    "  \"sub_queries\": [\n"
    "    {\"ticker\": \"AAPL\", \"fiscal_year\": 2024, \"item_number\": \"Item 7\", \"query_category\": \"GEO_DRIVERS\", \"query\": \"\"},\n"
    "    {\"ticker\": \"AAPL\", \"fiscal_year\": 2024, \"item_number\": \"Item 8\", \"query_category\": \"GEO_NUMBERS\", \"query\": \"\"},\n"
    "    {\"ticker\": \"MSFT\", \"fiscal_year\": 2024, \"item_number\": \"Item 7\", \"query_category\": \"GEO_DRIVERS\", \"query\": \"\"},\n"
    "    {\"ticker\": \"MSFT\", \"fiscal_year\": 2024, \"item_number\": \"Item 8\", \"query_category\": \"GEO_NUMBERS\", \"query\": \"\"}\n"
    "  ]\n"
    "}\n\n"

    "Example 2 (Tax Rate):\n"
    "User Prompt: 'What were the exact effective tax rates for Microsoft in 2023 and 2024?'\n"
    "Output: {\n"
    "  \"reasoning\": \"Two isolated TAX_RATE sub-queries, one per fiscal year, targeting Item 8.\",\n"
    "  \"sub_queries\": [\n"
    "    {\"ticker\": \"MSFT\", \"fiscal_year\": 2023, \"item_number\": \"Item 8\", \"query_category\": \"TAX_RATE\", \"query\": \"\"},\n"
    "    {\"ticker\": \"MSFT\", \"fiscal_year\": 2024, \"item_number\": \"Item 8\", \"query_category\": \"TAX_RATE\", \"query\": \"\"}\n"
    "  ]\n"
    "}\n\n"

    "Example 3 (Single Total Metric):\n"
    "User Prompt: 'Extract Apple total net sales exactly as reported in their 2023 financial statements.'\n"
    "Output: {\n"
    "  \"reasoning\": \"Single company-wide metric — one TOTAL_METRIC sub-query, Item 8 only.\",\n"
    "  \"sub_queries\": [\n"
    "    {\"ticker\": \"AAPL\", \"fiscal_year\": 2023, \"item_number\": \"Item 8\", \"query_category\": \"TOTAL_METRIC\", \"query\": \"\"}\n"
    "  ]\n"
    "}\n\n"

    "Example 4 (Risk Factors — free text, since the topic varies):\n"
    "User Prompt: 'Compare cybersecurity risk factors for Apple and Microsoft in 2024.'\n"
    "Output: {\n"
    "  \"reasoning\": \"Risk topic varies per question, so query_category is OTHER with a focused free-text string.\",\n"
    "  \"sub_queries\": [\n"
    "    {\"ticker\": \"AAPL\", \"fiscal_year\": 2024, \"item_number\": \"Item 1A\", \"query_category\": \"OTHER\", \"query\": \"cybersecurity risk factors\"},\n"
    "    {\"ticker\": \"MSFT\", \"fiscal_year\": 2024, \"item_number\": \"Item 1A\", \"query_category\": \"OTHER\", \"query\": \"cybersecurity risk factors\"}\n"
    "  ]\n"
    "}"
    )

    estimated_tokens = (len(system_prompt) + len(raw_query)) // 4 + 300
    planner_limiter.acquire(estimated_tokens)

    plan = None
    last_exc = None
    for attempt in range(3):
        try:
            plan = _planner_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                response_model=QueryPlan,
                max_retries=2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": raw_query}
                ],
                temperature=0.0
            )
            break
        except Exception as e:
            last_exc = e
            logger.warning(f"Planner attempt {attempt + 1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(1.5)

    if plan is None:
        plan = _salvage_failed_generation(last_exc)
        if plan is not None:
            logger.warning(f"Recovered planner output from malformed tool call ({len(plan.sub_queries)} sub-queries)")
        else:
            logger.error(f"Planner failed after 3 attempts: {last_exc}")
            return {
                "sub_queries": [],
                "retrieval_errors": [f"Planner unavailable: {str(last_exc)[:200]}"]
            }

    # --- DETERMINISTIC TUPLE GUARD --- (runs on salvaged output too, never bypassed)
    validated_queries = []
    for sq in plan.sub_queries:
        if (sq.ticker, sq.fiscal_year) in valid_inventory:
            sq_dict = sq.model_dump()
            canonical = CANONICAL_QUERIES.get(sq.query_category)
            sq_dict["query"] = canonical if canonical else sq_dict["query"].strip(" '\"")
            validated_queries.append(sq_dict)
        else:
            logger.warning(f"Tuple Guard dropped out-of-bounds query: {sq.ticker} FY{sq.fiscal_year}")

    return {"sub_queries": validated_queries}