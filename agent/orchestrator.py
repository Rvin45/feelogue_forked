"""
Thin entry point for user requests.
All routing, handling, and post-processing now lives inside graph.py.
"""
import json

from .graph import graph
from .context import get_current_config


def process_user_request(user_input: str) -> dict:
    """Parse an incoming MQTT payload and invoke the LangGraph graph."""
    data = json.loads(user_input)
    if "user_request_for_agent" not in data:
        return {
            "response": "Invalid payload received.",
            "rtd_command": None,
            "nodes": None,
            "referents": None,
            "followup_stage": False,
        }

    user_request = data["user_request_for_agent"]
    transcript_data = user_request.get("transcript", {})
    user_query = (
        transcript_data.get("text_transcript")
        or transcript_data.get("transcript")
        or ""
    ).strip()

    state_patch = {
        "user_query": user_query,
        "touchdata": user_request.get("touchdata") or {},
        "highlighted_context": user_request.get("highlighted_context") or {},
    }

    result = graph.invoke(state_patch, get_current_config())

    return {
        "response": result.get("final_response", ""),
        "rtd_command": result.get("rtd_command"),
        "nodes": result.get("nodes") or None,
        "referents": {
            "touch_used": result.get("touch_used", False),
            "highlight_used": result.get("highlight_used", False),
            "touch_nodes": result.get("touch_nodes") or {},
            "highlight_nodes": result.get("highlight_nodes") or {},
        },
        "followup_stage": result.get("followup_stage", False),
    }
