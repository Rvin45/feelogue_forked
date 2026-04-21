"""
Operations handling for chart manipulation (zoom, pan, layer switch).
"""
import re

from .client import client
from .config import OPENAI_MODEL
from .prompts import OPERATIONS_SYSTEM_PROMPT, get_operations_extraction_prompt
from .touch_context import pick_best_referent_node
from .utils import _normalize_text, _extract_pan_numeric_factor, parse_llm_json


def build_operations_rtd_command(
    user_query: str,
    touch_context: dict | None,
    highlighted_context: dict | None,
    x_values: list | None = None,
) -> dict:
    """
    Uses GPT to extract operation commands:
      operation: zoom | pan | layer_switch
      target: list[str] for explicit targets or anchored target for relative/deictic
      factor: int percent if explicitly given, else null
    """
    touch_context = touch_context or {}
    highlighted_context = highlighted_context or {}

    def _norm_placeholder_to_dict(x):
        return x if isinstance(x, dict) else {}

    def _query_needs_anchor(q: str, op: str) -> bool:
        q = (q or "").lower()
        if op not in {"pan", "zoom"}:
            return False
        cues = ["next", "previous", "prev", "prior", "this", "that", "here", "there",
                "these", "those", "selected", "touched", "highlighted"]
        return any(c in q for c in cues)

    def _extract_relative(q: str):
        q = (q or "").lower()
        if "next" in q:
            return "next"
        if re.search(r"\b(previous|prev|prior)\b", q):
            return "previous"
        if re.search(r"\b(this|that|these|those|selected|touched|highlighted|current|here)\b", q):
            return "this"
        return None

    touch_context = _norm_placeholder_to_dict(touch_context)
    highlighted_context = _norm_placeholder_to_dict(highlighted_context)

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": OPERATIONS_SYSTEM_PROMPT},
            {"role": "user", "content": get_operations_extraction_prompt(user_query, x_values)},
        ],
        temperature=0,
    )

    raw = (resp.choices[0].message.content or "").strip()

    result = parse_llm_json(raw, fallback={"operation": None, "target": None, "factor": None})

    op = result.get("operation")

    # Handle deictic/relative references
    if op and _query_needs_anchor(user_query, op):
        rel = _extract_relative(user_query)
        referent = pick_best_referent_node(touch_context, highlighted_context)

        if referent:
            result["target"] = [{
                "relative": rel or "this",
                "anchor": referent.get("node_values", {}),
                "source": referent.get("source"),
            }]
        elif rel:
            result["target"] = [{"relative": rel}]

    # Extract numeric factor for pan if not already set
    if op == "pan" and result.get("factor") is None:
        factor = _extract_pan_numeric_factor(user_query)
        if factor:
            result["factor"] = factor

    return result


def resolve_operation_targets_to_values(
    user_query: str,
    rtd_cmd: dict,
    df,
    x_col: str,
    y_col: str,
) -> dict:
    """
    Resolve operation targets to actual data values.
    """

    if not rtd_cmd or not isinstance(rtd_cmd, dict):
        return rtd_cmd

    op = rtd_cmd.get("operation")
    targets = rtd_cmd.get("target")

    if not op or not targets:
        return rtd_cmd

    # For layer_switch, targets are layer names, not data values
    if op == "layer_switch":
        return rtd_cmd

    # For zoom/pan with explicit data targets, resolve them
    if df is not None and x_col and isinstance(targets, list):
        resolved = []
        for t in targets:
            if isinstance(t, dict):
                # Already anchored
                resolved.append(t)
            elif isinstance(t, str):
                # Try to match against x-axis values
                t_norm = _normalize_text(t)
                matched = False
                for idx, row in df.iterrows():
                    x_val = row.get(x_col)
                    if x_val is not None:
                        if _normalize_text(str(x_val)) == t_norm:
                            resolved.append({
                                "x": x_val,
                                "y": row.get(y_col),
                                "index": idx,
                            })
                            matched = True
                            break
                if not matched:
                    # Keep original if no match
                    resolved.append(t)
            else:
                resolved.append(t)

        rtd_cmd["target"] = resolved

    return rtd_cmd


def build_operation_ack(rtd_cmd: dict) -> str:
    """
    Build an acknowledgment message for an operation.
    """
    if not rtd_cmd or not isinstance(rtd_cmd, dict):
        return "I couldn't understand that operation."

    op = rtd_cmd.get("operation")
    target = rtd_cmd.get("target")
    factor = rtd_cmd.get("factor")

    if not op:
        return "I couldn't determine what operation you wanted."

    if op == "zoom":
        if target:
            if isinstance(target, list) and len(target) > 0:
                t = target[0]
                if isinstance(t, dict) and "x" in t:
                    return f"Zooming to {t['x']}."
                elif isinstance(t, dict) and "relative" in t:
                    return f"Zooming to the {t['relative']} point."
                else:
                    return f"Zooming to {t}."
        return "Zooming in."

    elif op == "pan":
        direction = ""
        if target and isinstance(target, list):
            for t in target:
                if isinstance(t, str) and t.lower() in ("left", "right", "up", "down"):
                    direction = t.lower()
                    break

        factor_str = f" by {factor}%" if factor else ""

        if direction:
            return f"Panning {direction}{factor_str}."
        return f"Panning{factor_str}."

    elif op == "layer_switch":
        if target and isinstance(target, list) and len(target) > 0:
            return f"Switching to {target[0]} view."
        return "Switching layer."

    return f"Performing {op}."
