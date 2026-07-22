import logging
from typing import List, Dict
from pydantic import BaseModel, Field
import instructor
from groq import Groq

logger = logging.getLogger("GenerationGrader")

class FactEvaluation(BaseModel):
    is_present_in_answer: bool = Field(description="Does the final answer explicitly contain the expected value/claim?")
    is_grounded_in_context: bool = Field(description="Is the claim strictly derived from the provided retrieved context?")
    is_correctly_cited: bool = Field(description="Does the answer explicitly cite the expected source item (e.g., Item 7, Item 1A)?")

class AnswerEvaluation(BaseModel):
    evaluations: List[FactEvaluation]

def grade_generation(answer: str, expected_facts: List[Dict], contexts: List[Dict]) -> Dict[str, float]:
    """Uses a heavy LLM to judge specific claim extraction and citation fidelity."""
    if not expected_facts:
        return {"presence_score": 1.0, "groundedness_score": 1.0, "citation_score": 1.0}

    # client = instructor.from_groq(Groq())
    # FIX: Use instructor.Mode.JSON to prevent HTTP 400 server-side tool validation crashes
    client = instructor.from_groq(Groq(), mode=instructor.Mode.JSON)
    
    # Format context for the judge
    context_str = ""
    for idx, ctx in enumerate(contexts):
        item_str = ctx.get('item_number', '')
        item_display = item_str if item_str.startswith("Item") else f"Item {item_str}"
        
        context_str += f"--- Source {idx + 1} ({ctx.get('ticker')} FY{ctx.get('fiscal_year')} {item_display}) ---\n"
        # FIX: Pass the full context block without 1000-character truncation
        context_str += ctx.get("content", "") + "\n\n"

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
        eval_result: AnswerEvaluation = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            response_model=AnswerEvaluation,
            max_retries=3,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        
        results = eval_result.evaluations
        if not results:
            return {"presence_score": 0.0, "groundedness_score": 0.0, "citation_score": 0.0}
            
        presence = sum(1 for e in results if e.is_present_in_answer) / len(results)
        grounded = sum(1 for e in results if e.is_grounded_in_context) / len(results)
        citation = sum(1 for e in results if e.is_correctly_cited) / len(results)
        
        return {"presence_score": presence, "groundedness_score": grounded, "citation_score": citation}
        
    except Exception as e:
        logger.error(f"Generation grader failed: {e}")
        return {"presence_score": 0.0, "groundedness_score": 0.0, "citation_score": 0.0}