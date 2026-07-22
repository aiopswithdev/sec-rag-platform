import time
import logging
import tiktoken
from collections import defaultdict
from groq import Groq

logger = logging.getLogger("SynthesisNode")

def synthesize_answer(state: dict) -> dict:
    """
    Synthesis Node: Consumes raw contexts, applies per-entity balanced token allocation
    with a hybrid context diet (guaranteeing both tables and prose narrative per entity),
    and generates the final answer.
    """
    start_time = time.time()
    raw_query = state["raw_query"]
    contexts = state.get("retrieved_contexts", [])
    errors = state.get("retrieval_errors", [])

    if not contexts:
        err_context = f"\n(Note: Retrieval issues occurred: {'; '.join(errors)})" if errors else ""
        return {
            "final_answer": f"I do not have sufficient data in the retrieved filings to answer this.{err_context}",
            "sources": [],
            "telemetry": {"latency_seconds": round(time.time() - start_time, 3), "source_count": 0}
        }

    # --- Per-Entity Balanced Token Allocation ---
    encoder = tiktoken.get_encoding("cl100k_base")
    TOTAL_TOKEN_CEILING = 8000
    SAFETY_MARGIN = 500  # Reserve tokens for system prompt and formatting overhead
    
    # 1. Group contexts by ticker
    grouped_contexts = defaultdict(list)
    for ctx in contexts:
        ticker = ctx.get("ticker", "UNKNOWN")
        grouped_contexts[ticker].append(ctx)

    unique_tickers = sorted(list(grouped_contexts.keys()))
    num_tickers = max(len(unique_tickers), 1)

    # 2. Divide token budget equally across entities
    per_entity_ceiling = (TOTAL_TOKEN_CEILING - SAFETY_MARGIN) // num_tickers

    context_blocks = []
    sources = []
    any_truncated = False

    # 3. Pack contexts per ticker using the Hybrid Context Diet Strategy
    for ticker in unique_tickers:
        entity_contexts = grouped_contexts[ticker]
        
        # Separate table vs prose contexts
        tables = [c for c in entity_contexts if c.get("is_table", False)]
        prose = [c for c in entity_contexts if not c.get("is_table", False)]
        
        # Build diet-ordered contexts: Guarantee 1 Table + up to 2 Prose blocks first
        diet_ordered_contexts = []
        
        if tables:
            diet_ordered_contexts.append(tables.pop(0))
            
        for _ in range(2):
            if prose:
                diet_ordered_contexts.append(prose.pop(0))
                
        # Append remaining tables and prose for greedy fill
        diet_ordered_contexts.extend(tables)
        diet_ordered_contexts.extend(prose)
        
        entity_tokens = 0

        for parent in diet_ordered_contexts:
            raw_item = str(parent.get("item_number", ""))
            item_str = raw_item if raw_item.startswith("Item") else f"Item {raw_item}"
            header = f"--- Document: {parent['ticker']} (FY {parent['fiscal_year']} - {item_str}) ---\n"
            full_text = parent.get("content", "")
            
            header_tokens = len(encoder.encode(header))
            available_tokens = per_entity_ceiling - entity_tokens - header_tokens

            # Skip block if available allocation is too small to fit meaningful text
            if available_tokens <= 50:
                continue

            parent_token_ids = encoder.encode(full_text)

            if len(parent_token_ids) > available_tokens:
                any_truncated = True
                truncated_text = encoder.decode(parent_token_ids[:available_tokens])

                # Sentence-boundary safe cut
                last_period = truncated_text.rfind('. ')
                if last_period > len(truncated_text) // 2:
                    truncated_text = truncated_text[:last_period + 1]

                truncated_text += "\n\n[...TRUNCATED DUE TO PER-ENTITY CONTEXT LIMIT...]"
                added_tokens = len(encoder.encode(truncated_text))
                is_curr_truncated = True
            else:
                truncated_text = full_text
                added_tokens = len(parent_token_ids)
                is_curr_truncated = False

            block = header + truncated_text + "\n"
            context_blocks.append(block)

            sources.append({
                "ticker": parent["ticker"],
                "fiscal_year": parent["fiscal_year"],
                "item_number": parent["item_number"],
                "parent_id": parent["parent_id"],
                "is_table": parent["is_table"]
            })

            entity_tokens += (header_tokens + added_tokens)
            
            # Stop filling for this ticker once ceiling is reached or truncated
            if is_curr_truncated or entity_tokens >= per_entity_ceiling:
                break

    compiled_context = "\n".join(context_blocks)

    system_prompt = (
        "You are a strict financial analyst AI. Answer using ONLY the provided SEC 10-K context.\n\n"
        "RULES:\n"
        "1. If the context does not contain the specific metric or fact requested, state: "
        "'I do not have sufficient data in the retrieved filings to answer this.'\n"
        "2. DO NOT invent facts or external data.\n"
        "3. CITE YOUR SOURCES strictly at the end of EVERY sentence or bullet using: [Ticker FY Year - Item Number] "
        "(e.g., [AAPL FY 2024 - Item 1A] or [MSFT FY 2023 - Item 8]).\n"
        "4. When comparing or discussing multiple companies, you MUST provide a dedicated, equally detailed section "
        "or bullet list for EACH company. DO NOT combine companies into single merged sentences, and DO NOT abbreviate "
        "or omit details for one company in favor of another.\n"
        "5. Present financial table data using clean markdown tables."
    )

    if errors:
        system_prompt += f"\nNOTE: Some data could not be retrieved due to system errors: {'; '.join(errors)}"
    if any_truncated:
        system_prompt += "\nNOTE: Context was truncated for safety. Answer as best as you can with the provided fragment."

    llm_client = Groq()
    try:
        response = llm_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{compiled_context}\n\nQuestion: {raw_query}"}
            ]
        )
        answer = response.choices[0].message.content
    except Exception as e:
        logger.error(f"Synthesis generation error: {e}")
        answer = f"Error during final synthesis generation: {str(e)}"

    return {
        "final_answer": answer,
        "sources": sources,
        "telemetry": {
            "total_latency_seconds": round(time.time() - start_time, 3),
            "source_count": len(sources),
            "retrieval_errors_count": len(errors)
        }
    }