"""
Intent classification for user queries.
Merges intent classification and deictic detection into a single LLM call.
"""
from .client import client
from .config import OPENAI_MODEL_CLASSIFIER
from .prompts import get_intent_classification_prompt, INTENT_CLASSIFIER_SYSTEM_PROMPT
from .utils import parse_llm_json


def classify_query(user_query: str, has_image: bool = False) -> dict:
    """
    Classify user intent AND detect deictic references in a single call.

    Returns:
        dict with keys:
            - intent: one of the valid intent labels
            - has_deictic: boolean
    """
    resp = client.chat.completions.create(
        model=OPENAI_MODEL_CLASSIFIER,
        messages=[
            {"role": "system", "content": INTENT_CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": get_intent_classification_prompt(user_query)},
        ],
        temperature=0,
    )

    raw = (resp.choices[0].message.content or "").strip()

    result = parse_llm_json(raw, fallback={"intent": "general_question", "has_deictic": False})

    # Validate intent
    valid_intents = {
        "load_chart", "chart_overview", "image_analysis",
        "touch_interaction", "trend", "operations",
        "data_analysis", "general_question",
    }

    intent = result.get("intent", "general_question")
    if intent not in valid_intents:
        # Try to match partial
        for v in valid_intents:
            if v in intent.lower():
                intent = v
                break
        else:
            intent = "general_question"

    has_deictic = result.get("has_deictic") is True

    return {
        "intent": intent,
        "has_deictic": has_deictic,
    }


# Keep old functions for backwards compatibility (they now use the merged call)
def classify_intent(user_query: str, has_image: bool = False) -> str:
    """Legacy function - returns just the intent."""
    return classify_query(user_query, has_image)["intent"]


def detect_deictic_reference(user_query: str) -> bool:
    """Legacy function - returns just the deictic flag."""
    return classify_query(user_query)["has_deictic"]
