"""
Operations handling for chart manipulation (zoom, reset, pan, filter).

Target resolution (data lookups, ordinals, touch/highlight-based points,
direction/quadrant phrasing) happens in the operations tool loop (graph.py's
operations_node + _run_tool_loop, bound to _llm_ops with
response_format=OPERATIONS_SCHEMA), which also produces the spoken message
itself. This module parses that loop's final JSON output, and provides a
deterministic backup acknowledgment for when the model's own message is
missing or empty.
"""
from .utils import parse_llm_json

_FALLBACK_OPERATION_RESPONSE = {
    "operation": None,
    "target": None,
    "factor": None,
    "clarification_needed": True,
    "message": "Sorry, I didn't catch what chart operation you wanted. Could you rephrase that?",
}


def parse_operation_response(raw: str) -> dict:
    """Parse the operations tool loop's final structured-output text into a dict."""
    return parse_llm_json(raw, fallback=dict(_FALLBACK_OPERATION_RESPONSE))


def build_operation_ack(rtd_cmd: dict) -> str:
    """
    Deterministic backup acknowledgment, built straight from the resolved
    rtd_command. Used only when the model's own `message` is missing/empty.
    """
    if not rtd_cmd or not isinstance(rtd_cmd, dict):
        return "I couldn't understand that operation."

    op = rtd_cmd.get("operation")
    target = rtd_cmd.get("target")
    factor = rtd_cmd.get("factor")

    if not op:
        return "I couldn't determine what operation you wanted."

    factor_str = f" by {factor}%" if factor else ""
    has_target = isinstance(target, list) and len(target) > 0
    is_range = has_target and len(target) >= 2

    if op == "zoom":
        if is_range:
            return f"Zooming to {target[0]} through {target[1]}{factor_str}."
        if has_target:
            return f"Zooming to {target[0]}{factor_str}."
        return f"Zooming in{factor_str}."

    if op == "pan":
        if has_target:
            return f"Panning to {target[0]}{factor_str}."
        return f"Panning{factor_str}."

    if op == "filter":
        if is_range:
            return f"Filtering to {target[0]} through {target[1]}."
        if has_target:
            return f"Filtering to {target[0]}."
        return "Filtering the data."

    if op == "reset":
        return "Resetting the view."

    return f"Performing {op}."
