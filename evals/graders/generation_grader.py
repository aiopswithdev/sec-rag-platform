import logging
from typing import List, Dict
from pydantic import BaseModel, Field
import instructor
from groq import Groq
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

logger = logging.getLogger("GenerationGrader")

class FactEvaluation(BaseModel):
    is_present_in_answer: bool = Field(description="Does the final answer explicitly contain the expected value or claim? Must be boolean true or false.")
    is_grounded_in_context: bool = Field(description="Is the claim strictly supported by the provided context? Must be boolean true or false.")
    is_correctly_cited: bool = Field(description="Does the answer explicitly cite the expected source item? Must be boolean true or false.")

class AnswerEvaluation(BaseModel):
    evaluations: List[FactEvaluation]


# Exponential backoff decorator to gracefully handle temporary HTTP 429 rate limits
@retry(
    wait=wait_exponential(multiplier=2, min=2, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(Exception)
)
def _call_judge_with_backoff(client, prompt: str, system_prompt: str):
    return client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # 70B has significantly higher TPM limits on Groq than 120B
        response_model=AnswerEvaluation,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )


def grade_generation(answer: str, expected_facts: List[Dict], contexts: List[Dict]) -> Dict[str, float]:
    """Uses an LLM to judge specific claim extraction and citation fidelity."""
    if not expected_facts:
        return {"presence_score": 1.0, "groundedness_score": 1.0, "citation_score": 1.0}

    client = instructor.from_groq(Groq(), mode=instructor.Mode.JSON)
    
    # Format context for the judge (Cap at 2000 chars per context block to keep token payload < 2500)
    context_str = ""
    for idx, ctx in enumerate(contexts):
        item_str = str(ctx.get('item_number', ''))
        item_display = item_str if item_str.startswith("Item") else f"Item {item_str}"
        
        context_str += f"--- Source {idx + 1} ({ctx.get('ticker')} FY{ctx.get('fiscal_year')} {item_display}) ---\n"
        context_str += ctx.get("content", "")[:2000] + "...\n\n"

    prompt = (
        f"Evaluate the generated answer against the expected facts.\n\n"
        f"RETRIEVED CONTEXT:\n{context_str}\n\n"
        f"GENERATED ANSWER:\n{answer}\n\n"
        f"EXPECTED FACTS TO CHECK:\n"
    )
    for fact in expected_facts:
        prompt += f"- Claim: {fact['claim']} | Expected Value: {fact['value']} | Expected Source: {fact['source_item']}\n"

    system_prompt = (
        "You are an objective, strict scoring system.\n"
        "You MUST evaluate each expected fact and output a JSON object containing an 'evaluations' array.\n"
        "Each object in 'evaluations' MUST strictly contain ONLY these three boolean fields:\n"
        "- 'is_present_in_answer' (true/false)\n"
        "- 'is_grounded_in_context' (true/false)\n"
        "- 'is_correctly_cited' (true/false)\n\n"
        "DO NOT output fields like 'expected', 'actual', or 'result'."
    )

    try:
        eval_result: AnswerEvaluation = _call_judge_with_backoff(client, prompt, system_prompt)
        
        results = eval_result.evaluations
        if not results:
            return {"presence_score": 0.0, "groundedness_score": 0.0, "citation_score": 0.0}
            
        presence = sum(1 for e in results if e.is_present_in_answer) / len(results)
        grounded = sum(1 for e in results if e.is_grounded_in_context) / len(results)
        citation = sum(1 for e in results if e.is_correctly_cited) / len(results)
        
        return {"presence_score": presence, "groundedness_score": grounded, "citation_score": citation}
        
    except Exception as e:
        logger.error(f"Generation grader failed after retries: {e}")
        return {"presence_score": 0.0, "groundedness_score": 0.0, "citation_score": 0.0}