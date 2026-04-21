"""
Touch and highlight context handling.
"""
from typing import Tuple, List, Dict

# Minimum probability threshold for touch detection
TOUCH_PROBABILITY_THRESHOLD = 0.2


def collect_touch_nodes(touchdata: dict) -> Tuple[List[str], Dict]:
    """
    Extract touched nodes from touch data.

    Returns:
        Tuple of (human-readable info list, nodes dict)
    """
    info, nodes_out = [], {}
    if not isinstance(touchdata, dict):
        return info, nodes_out

    for side in ["left_touch", "right_touch"]:
        block = touchdata.get(side)
        if not isinstance(block, dict):
            continue

        nodes = block.get("nodes", {}) or {}
        for node_id, node in nodes.items():
            if not isinstance(node, dict):
                continue
            if float(node.get("probability", 0) or 0) < TOUCH_PROBABILITY_THRESHOLD:
                continue

            # Categorize node type
            if "data-mark" in node_id:
                t = "Data Value"
            elif "x-axis" in node_id:
                t = "X-Axis Markup"
            elif "y-axis" in node_id:
                t = "Y-Axis Markup"
            else:
                t = "Unknown"

            nv = node.get("node_values", {}) or {}
            info.append(
                f"{side} - {t}: [{', '.join(map(str, nv.keys()))}], "
                f"Values: [{', '.join(map(str, nv.values()))}]"
            )
            nodes_out[node_id] = node

    return info, nodes_out


def collect_highlight_nodes(highlighted_context: dict) -> Tuple[List[str], Dict]:
    """
    Extract highlighted nodes from highlight context.

    Returns:
        Tuple of (human-readable info list, nodes dict)
    """
    info, nodes_out = [], {}
    if not isinstance(highlighted_context, dict):
        return info, nodes_out

    nodes = highlighted_context.get("nodes") or {}
    if not isinstance(nodes, dict):
        return info, nodes_out

    for node_id, node in nodes.items():
        if not isinstance(node, dict):
            continue
        if float(node.get("probability", 0) or 0) < TOUCH_PROBABILITY_THRESHOLD:
            continue

        if "data-mark" in node_id:
            t = "Data Value"
        elif "x-axis" in node_id:
            t = "X-Axis Markup"
        elif "y-axis" in node_id:
            t = "Y-Axis Markup"
        else:
            t = "Unknown"

        nv = node.get("node_values", {}) or {}
        info.append(
            f"Highlighted - {t}: [{', '.join(map(str, nv.keys()))}], "
            f"Values: [{', '.join(map(str, nv.values()))}]"
        )
        nodes_out[node_id] = node

    return info, nodes_out


def _pick_best_node_values(nodes: dict) -> dict | None:
    """
    Given a flat {node_id: node} dict (already extracted by collect_*),
    return the node_values of the highest-probability node, or None.
    """
    best_prob, best_nv = -1, None
    for node in nodes.values():
        if not isinstance(node, dict):
            continue
        try:
            prob = float(node.get("probability", 0) or 0)
        except Exception:
            prob = 0
        nv = node.get("node_values") or {}
        if nv and prob > best_prob:
            best_prob, best_nv = prob, nv
    return best_nv


def pick_best_referent_node(touch_ctx: dict, highlight_ctx: dict) -> dict | None:
    """
    Pick the best referent node from touch or highlight context.
    Prioritizes by probability.
    """
    candidates = []

    if isinstance(touch_ctx, dict):
        for side in ("left_touch", "right_touch"):
            block = touch_ctx.get(side)
            if isinstance(block, dict):
                nodes = block.get("nodes") or {}
                if isinstance(nodes, dict):
                    for node_id, node in nodes.items():
                        if isinstance(node, dict):
                            prob = float(node.get("probability", 0) or 0)
                            nv = node.get("node_values") or {}
                            if prob >= TOUCH_PROBABILITY_THRESHOLD and isinstance(nv, dict) and nv:
                                candidates.append({
                                    "source": f"touch:{side}",
                                    "node_id": node_id,
                                    "node_values": nv,
                                    "probability": prob,
                                })

    if isinstance(highlight_ctx, dict):
        nodes = highlight_ctx.get("nodes") or {}
        if isinstance(nodes, dict):
            for node_id, node in nodes.items():
                if isinstance(node, dict):
                    prob = float(node.get("probability", 0) or 0)
                    nv = node.get("node_values") or {}
                    if prob >= TOUCH_PROBABILITY_THRESHOLD and isinstance(nv, dict) and nv:
                        candidates.append({
                            "source": "highlight",
                            "node_id": node_id,
                            "node_values": nv,
                            "probability": prob,
                        })

    if not candidates:
        return None
    candidates.sort(key=lambda c: c["probability"], reverse=True)
    return candidates[0]
