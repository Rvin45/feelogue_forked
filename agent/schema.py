
# Formatted output for ChatGPT response
INTENT_SCHEMA={
  "type": "object",
  "properties": {
    "intents": {
      "type": "array",
      "description": "List of detected user intents, ordered by priority (most important first)",
      "items": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string",
            "enum": [
              "load_chart",
              "image_analysis",
              "chart_overview",
              "touch_interaction",
              "data_analysis",
              "trend",
              "operations",
              "general_question"
            ],
            "description": "The classified intent type"
          },
          "query": {
            "type": "string",
            "description": "A fully self-contained query specific to this intent"
          }
        },
        "required": ["type", "query"],
        "additionalProperties": False
      },
      "minItems": 1
    },
    "has_deictic": {
      "type": "boolean",
      "description": "True if the query includes explicit deictic references like 'this', 'that', or touched elements"
    }
  },
  "required": ["intents", "has_deictic"],
  "additionalProperties": False
}

EVALUATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "result": {
            "type": "string",
            "enum": ["answered", "unanswered", "intent_error"],
            "description": (
                "answered: response fully addressed the user's question. "
                "unanswered: response is evasive, off-topic, or missing key information. "
                "intent_error: wrong handler was used (e.g., visual question answered without "
                "image_analysis, or data question answered with load_chart only)."
            ),
        },
        "feedback": {
            "type": "string",
            "description": (
                "Brief explanation of the verdict. On intent_error: which handler was wrong "
                "and what should have been used. On unanswered: what aspect was not addressed."
            ),
        },
        "followup_question": {
            "type": "string",
            "description": (
                "A short follow-up question the user could ask when result is 'unanswered'. "
                "Return an empty string when result is 'answered' or 'intent_error'."
            ),
        },
    },
    "required": ["result", "feedback", "followup_question"],
    "additionalProperties": False,
}