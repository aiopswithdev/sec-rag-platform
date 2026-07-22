def grade_refusal(answer: str) -> bool:
    """Deterministically checks if the LLM triggered the strict refusal guardrail."""
    target_phrase = "I do not have sufficient data in the retrieved filings"
    return target_phrase.lower() in answer.lower()