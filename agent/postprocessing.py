"""
Post-processing functions for agent responses.
"""
from .client import client
from .config import OPENAI_MODEL
from .prompts import get_rewrite_list_prompt, get_combine_multi_intent_responses_prompt
from .utils import _extract_bulleted_items, rewrite_long_lists_locally, parse_llm_json


def combine_multi_intent_responses(responses: dict[str, str], query: str) -> str:
    """
    Combine multiple response fragments into one coherent answer using GPT.
    """
    if not responses:
        return ""
    if len(responses) == 1:
        _, value = next(iter(responses.items()))
        return value

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": get_combine_multi_intent_responses_prompt(responses=responses, query=query)}],
            temperature=0,
        )
        combined = (resp.choices[0].message.content or "").strip()
        if combined:
            return combined
    except Exception as e:
        print(f"Warning: Response combination failed: {e}")

    # Simple fallback: join with spaces
    return " ".join(responses.values())


def rewrite_long_node_lists_with_gpt(text: str) -> str:
    """
    Rewrite long bulleted lists in the response into concise sentences.
    Falls back to local rewriting if GPT fails.
    """
    _, items, _, _ = _extract_bulleted_items(text)
    if len(items) < 4:
        return text
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": get_rewrite_list_prompt(text)}],
            temperature=0,
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        if rewritten:
            return rewritten
    except Exception:
        pass
    return rewrite_long_lists_locally(text, max_per_sentence=2, min_trigger=4)


# =============================================================================
# Data-query structured response parsing
# =============================================================================

_FALLBACK_DATA_QUERY_RESPONSE = {
    "message": "Sorry, something went wrong looking into that. Could you ask again?",
    "highlighted_ids": [],
}


def parse_data_query_response(raw: str) -> dict:
    """Parse the data-query tool loop's final structured-output text into a dict."""
    return parse_llm_json(raw, fallback=dict(_FALLBACK_DATA_QUERY_RESPONSE))

