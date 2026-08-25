import os
import time
import logging
import tiktoken
from collections import defaultdict
from groq import Groq

from src.common.rate_limiter import synthesis_limiter

logger = logging.getLogger("SynthesisNode")

_synthesis_client = Groq()

TOTAL_TOKEN_CEILING = int(os.getenv("SYNTHESIS_TOKEN_CEILING", "5000"))
SAFETY_MARGIN = int(os.getenv("SYNTHESIS_SAFETY_MARGIN", "1800"))
MAX_COMPLETION_TOKENS = int(os.getenv("SYNTHESIS_MAX_COMPLETION_TOKENS", "1400"))
MAX_BLOCK_SHARE = float(os.getenv("SYNTHESIS_MAX_BLOCK_SHARE", "0.45"))
REFUSAL_PHRASE = "I do not have sufficient data in the retrieved filings"

SYSTEM_PROMPT = (
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
    "5. Present financial table data using clean markdown tables.\n"
    "6. DERIVED METRICS: You MAY perform simple arithmetic (ratios, percentages, year-over-year changes) using "
    "values explicitly present in the context. This is NOT inventing data. When you do, show the inputs and the "
    "computation, then cite the source of the input values. For example, an effective tax rate may be computed as "
    "provision for income taxes divided by income before income taxes when both figures appear in the context."
)


def _pack_context(contexts: list, encoder) -> tuple[str, list, bool]:
    grouped_contexts = defaultdict(list)
    for ctx in contexts:
        key = (ctx.get("ticker", "UNKNOWN"), ctx.get("fiscal_year", 0))
        grouped_contexts[key].append(ctx)

    group_keys = sorted(grouped_contexts.keys())
    num_groups = max(len(group_keys), 1)
    per_entity_ceiling = max((TOTAL_TOKEN_CEILING - SAFETY_MARGIN) // num_groups, 400)
    block_ceiling = max(int(per_entity_ceiling * MAX_BLOCK_SHARE), 250)

    context_blocks = []
    sources = []
    any_truncated = False

    for key in group_keys:
        entity_contexts = grouped_contexts[key]

        tables = [c for c in entity_contexts if c.get("is_table", False)]
        prose = [c for c in entity_contexts if not c.get("is_table", False)]
        tables.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        prose.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        diet_ordered_contexts = []
        if tables:
            diet_ordered_contexts.append(tables.pop(0))
        for _ in range(2):
            if prose:
                diet_ordered_contexts.append(prose.pop(0))
        diet_ordered_contexts.extend(tables)
        diet_ordered_contexts.extend(prose)

        entity_tokens = 0

        for parent in diet_ordered_contexts:
            remaining = per_entity_ceiling - entity_tokens
            if remaining <= 80:
                break

            item_str = str(parent.get("item_number", "UNKNOWN"))
            header = f"--- Document: {parent['ticker']} (FY {parent['fiscal_year']} - {item_str}) ---\n"
            full_text = parent.get("content", "")

            header_tokens = len(encoder.encode(header))
            available_tokens = min(remaining - header_tokens, block_ceiling)
            if available_tokens <= 50:
                continue

            parent_token_ids = encoder.encode(full_text)

            if len(parent_token_ids) > available_tokens:
                any_truncated = True
                truncated_text = encoder.decode(parent_token_ids[:available_tokens])
                last_period = truncated_text.rfind('. ')
                if last_period > len(truncated_text) // 2:
                    truncated_text = truncated_text[:last_period + 1]
                truncated_text += "\n\n[...TRUNCATED DUE TO PER-ENTITY CONTEXT LIMIT...]"
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
            entity_tokens += (header_tokens + added_tokens)

    return "\n".join(context_blocks), sources, any_truncated


def _call_synthesis(system_prompt: str, user_msg: str, encoder) -> str:
    """Single synthesis call with TPM pacing, usage reconciliation, and resilience to the
    stale-connection failures that follow long limiter sleeps."""
    estimated = len(encoder.encode(system_prompt)) + len(encoder.encode(user_msg)) + MAX_COMPLETION_TOKENS
    handle = synthesis_limiter.acquire(estimated)

    last_exc = None
    for attempt in range(3):
        try:
            response = _synthesis_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                temperature=0.0,
                max_tokens=MAX_COMPLETION_TOKENS,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ]
            )
            choice = response.choices[0]
            logger.info(f"Synthesis finish_reason={choice.finish_reason} usage={response.usage}")
            if getattr(response, "usage", None):
                synthesis_limiter.reconcile(handle, response.usage.total_tokens)
            return choice.message.content
        except Exception as e:
            last_exc = e
            logger.warning(f"Synthesis call attempt {attempt + 1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(3.0 * (attempt + 1))  # long enough for the HTTP pool to recycle dead sockets
    raise last_exc


def synthesize_answer(state: dict) -> dict:
    start_time = time.time()
    raw_query = state["raw_query"]
    contexts = state.get("retrieved_contexts", [])
    errors = state.get("retrieval_errors", [])

    if not contexts:
        err_context = f"\n(Note: Retrieval issues occurred: {'; '.join(errors)})" if errors else ""
        return {
            "final_answer": f"I do not have sufficient data in the retrieved filings to answer this.{err_context}",
            "sources": [],
            "compiled_context": "",
            "telemetry": {"synthesis_latency_seconds": round(time.time() - start_time, 3), "source_count": 0}
        }

    encoder = tiktoken.get_encoding("cl100k_base")
    compiled_context, sources, any_truncated = _pack_context(contexts, encoder)

    system_prompt = SYSTEM_PROMPT
    if errors:
        system_prompt += f"\nNOTE: Some data could not be retrieved due to system errors: {'; '.join(errors)}"
    if any_truncated:
        system_prompt += "\nNOTE: Context was truncated for safety. Answer as best as you can with the provided fragment."

    user_msg = f"Context:\n{compiled_context}\n\nQuestion: {raw_query}"

    try:
        answer = _call_synthesis(system_prompt, user_msg, encoder)
    except Exception as e:
        logger.error(f"Synthesis generation error: {e}")
        answer = f"Error during final synthesis generation: {str(e)}"

    # A refusal while holding non-empty context is suspicious — try once more.
    # ISOLATED: a failed re-ask must never destroy the valid answer we already have.
    if answer and REFUSAL_PHRASE.lower() in answer.lower() and compiled_context:
        logger.warning("Refusal despite non-empty context — re-asking once with an extraction nudge")
        nudge_msg = user_msg + (
            "\n\nIMPORTANT: The context above DOES contain retrieved SEC filing data. Examine every table row "
            "and sentence carefully before concluding data is missing. If the exact metric is not stated verbatim "
            "but can be computed from values that ARE present, compute it and show your work."
        )
        try:
            retry_answer = _call_synthesis(system_prompt, nudge_msg, encoder)
            if retry_answer and REFUSAL_PHRASE.lower() not in retry_answer.lower():
                logger.info("Re-ask succeeded — replacing refusal with extracted answer")
                answer = retry_answer
            else:
                logger.info("Re-ask still refused — keeping original refusal (likely a genuine retrieval gap)")
        except Exception as e:
            logger.warning(f"Re-ask failed ({e}) — keeping original answer")

    return {
        "final_answer": answer,
        "sources": sources,
        "compiled_context": compiled_context,
        "telemetry": {
            "synthesis_latency_seconds": round(time.time() - start_time, 3),
            "source_count": len(sources),
            "retrieval_errors_count": len(errors)
        }
    }