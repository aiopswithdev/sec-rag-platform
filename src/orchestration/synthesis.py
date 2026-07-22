import time
import logging
import tiktoken
from groq import Groq

logger = logging.getLogger("SynthesisNode")

def synthesize_answer(state: dict) -> dict:
    """Synthesis Node: Consumes raw contexts, applies guaranteed truncation, and generates final answer."""
    start_time = time.time()
    raw_query = state["raw_query"]
    contexts = state.get("retrieved_contexts", [])
    # SORT: Place tables (is_table == True) at the front of the list
    # True evaluates to 1, False to 0. We sort by `not x['is_table']` to put False (0) first.
    contexts = sorted(contexts, key=lambda x: not x.get("is_table", False))
    errors = state.get("retrieval_errors", [])
    
    if not contexts:
        err_context = f"\n(Note: Retrieval issues occurred: {'; '.join(errors)})" if errors else ""
        return {
            "final_answer": f"I do not have sufficient data in the retrieved filings to answer this.{err_context}",
            "sources": [],
            "telemetry": {"latency_seconds": round(time.time() - start_time, 3), "source_count": 0}
        }

    # --- PORTED FROM api.py: Guaranteed-First-Parent Truncation Algorithm ---
    encoder = tiktoken.get_encoding("cl100k_base")
    TOKEN_CEILING = 8000
    
    context_blocks = []
    sources = []
    total_tokens = 0
    truncated = False

    for parent in contexts:
        header = f"--- Document: {parent['ticker']} (FY {parent['fiscal_year']} - {parent['item_number']}) ---\n"
        full_text = parent['content']
        
        header_tokens = len(encoder.encode(header))
        available_tokens = TOKEN_CEILING - total_tokens - header_tokens - 500
        
        if available_tokens <= 0:
            break

        parent_token_ids = encoder.encode(full_text)
        
        if len(parent_token_ids) > available_tokens:
            truncated = True
            truncated_text = encoder.decode(parent_token_ids[:available_tokens])
            
            # Sentence-boundary safe cut
            last_period = truncated_text.rfind('. ')
            if last_period > len(truncated_text) // 2:
                truncated_text = truncated_text[:last_period + 1]
                
            truncated_text += "\n\n[...TRUNCATED DUE TO CONTEXT LIMIT...]"
            added_tokens = len(encoder.encode(truncated_text))
        else:
            truncated_text = full_text
            added_tokens = len(parent_token_ids)

        block = header + truncated_text + "\n"
        context_blocks.append(block)
        
        sources.append({
            "ticker": parent["ticker"],
            "fiscal_year": parent["fiscal_year"],
            "item_number": parent["item_number"],
            "parent_id": parent["parent_id"],
            "is_table": parent["is_table"]
        })
        
        total_tokens += (header_tokens + added_tokens)
        if truncated:
            break

    compiled_context = "\n".join(context_blocks)

    system_prompt = (
        "You are a strict financial analyst AI. Answer the user's question using ONLY the provided SEC 10-K context.\n\n"
        "RULES:\n"
        "1. If the context does not contain the specific metric or fact requested, state: "
        "'I do not have sufficient data in the retrieved filings to answer this.'\n"
        "2. DO NOT invent facts or external data.\n"
        "3. You MUST cite the source for every fact or metric: e.g., 'AAPL FY 2024 - Item 1A'.\n"
        "4. Present financial table data using clean markdown tables.\n"
        "5. Compare multiple documents explicitly using bullet points or comparison tables."
    )
    
    if errors:
        system_prompt += f"\nNOTE: Some data could not be retrieved due to system errors: {'; '.join(errors)}"
    if truncated:
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