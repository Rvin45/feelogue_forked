"""
Main orchestrator for processing user requests.
"""

import json
import time

import pandas as pd

from .chart_loader import analyze_user_intent_with_context
from .client import client
from .config import OPENAI_MODEL
from .context import (
    agent_context,
    clear_followup,
    ensure_df_headers_in_context,
    get_graph_config,
    get_xy_cols,
    mark_followup,
)
from .graph import graph
from .intent import classify_query
from .operations import (
    build_operation_ack,
    build_operations_rtd_command,
    resolve_operation_targets_to_values,
)
from .postprocessing import extract_highlighted_data_points, rewrite_long_node_lists_with_gpt
from .utils import strip_markdown
from .prompts import (
    CHART_OVERVIEW_SYSTEM_PROMPT,
    IMAGE_ANALYSIS_SYSTEM_PROMPT,
    get_chart_overview_prompt,
    get_system_prompt,
)
from .touch_context import collect_highlight_nodes, collect_touch_nodes, _pick_best_node_values


# =============================================================================
# Response builder
# =============================================================================

def _make_result(
    response: str,
    rtd_command=None,
    nodes=None,
    followup_stage: bool = False,
    touch_used: bool = False,
    highlight_used: bool = False,
    touch_nodes=None,
    highlight_nodes=None,
) -> dict:
    """Build the standard response dict sent back to mqtt_handler."""
    return {
        "response": response,
        "rtd_command": rtd_command,
        "nodes": nodes,
        "referents": {
            "touch_used": touch_used,
            "highlight_used": highlight_used,
            "touch_nodes": touch_nodes,
            "highlight_nodes": highlight_nodes,
        },
        "followup_stage": followup_stage,
    }


# =============================================================================
# Intent handlers
# =============================================================================

def _handle_load_chart(user_query: str) -> dict:
    """Resolve and load a chart by name, with disambiguation if needed."""
    analysis = analyze_user_intent_with_context(user_query, agent_context)

    if analysis.get("followup_stage", False):
        mark_followup("load_chart")
    else:
        clear_followup()

    return _make_result(
        response=analysis["response"],
        rtd_command=analysis.get("rtd_command"),
        followup_stage=agent_context.get("followup_stage", False),
    )


def _handle_image_analysis(user_query: str, base64_image: str | None) -> dict:
    """Analyze the current chart image using multimodal LLM."""
    if not base64_image:
        clear_followup()
        return _make_result(
            response="I don't have an image of the current chart to analyze."
        )

    print("Processing multimodal image analysis...")
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": IMAGE_ANALYSIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_query},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            },
        ],
        temperature=0,
    )
    clear_followup()
    return _make_result(response=response.choices[0].message.content)


def _handle_operations(user_query: str, touchdata: dict, highlighted_context: dict) -> dict:
    """Extract and resolve a chart operation (zoom, pan, layer switch)."""
    df = agent_context.get("df")
    x_col, y_col = get_xy_cols()

    x_values = None
    if df is not None and x_col and x_col in df.columns:
        x_values = df[x_col].astype(str).tolist()

    rtd_cmd = build_operations_rtd_command(
        user_query=user_query,
        touch_context=touchdata,
        highlighted_context=highlighted_context,
        x_values=x_values,
    )
    rtd_cmd = resolve_operation_targets_to_values(
        user_query=user_query,
        rtd_cmd=rtd_cmd,
        df=df,
        x_col=x_col,
        y_col=y_col,
    )

    clear_followup()
    return _make_result(
        response=build_operation_ack(rtd_cmd),
        rtd_command=rtd_cmd,
    )


def _handle_chart_overview() -> dict:
    """Generate a spoken overview of the current chart."""
    pre_built = agent_context.get("chart_overview")

    if pre_built:
        if isinstance(pre_built, dict):
            parts = []
            if pre_built.get("title"):
                parts.append(pre_built["title"] + ".")
            if pre_built.get("description"):
                parts.append(pre_built["description"])
            for series_info in pre_built.get("series", []):
                if isinstance(series_info, dict):
                    name = series_info.get("name", "")
                    desc = series_info.get("description", "")
                    if name and desc:
                        parts.append(f"{name}: {desc}")
                    elif desc:
                        parts.append(desc)
                elif isinstance(series_info, str):
                    parts.append(series_info)
            response = " ".join(parts) if parts else str(pre_built)
        else:
            response = str(pre_built)

        clear_followup()
        return _make_result(response=response)

    # No pre-built overview — generate from column names via LLM
    chart_type = agent_context.get("chart_type")
    df_cols = agent_context.get("df_columns", [])

    if not chart_type or not df_cols:
        clear_followup()
        return _make_result(
            response="I don't have a chart loaded yet. Please load a chart first."
        )

    x_col = agent_context.get("x_field") or df_cols[0]
    y_col = agent_context.get("y_field") or (df_cols[1] if len(df_cols) >= 2 else "y-axis")
    color_field = agent_context.get("color_field")

    try:
        overview_response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": CHART_OVERVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": get_chart_overview_prompt(
                    x_col, y_col, chart_type, color_col=color_field
                )},
            ],
            temperature=0,
        )
        response = overview_response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Warning: GPT overview fallback due to: {e}")
        response = f"This {chart_type} chart shows how {y_col} changes with respect to {x_col}."

    clear_followup()
    return _make_result(response=response)


def _handle_data_query(user_query: str, touchdata: dict, highlighted_context: dict) -> dict:
    """
    Handle data_analysis, trend, touch_interaction, and general_question intents.

    Enriches the query with any touch/highlight context, streams it through
    LangGraph, then post-processes the response and extracts highlighted nodes.
    """
    # --- Touch & highlight referent enrichment ---
    touch_info, touch_nodes = collect_touch_nodes(touchdata)
    highlight_info, highlight_nodes = collect_highlight_nodes(highlighted_context)

    use_touch = len(touch_nodes) > 0
    use_highlight = len(highlight_nodes) > 0

    referent_parts = []
    if use_touch:
        referent_parts.extend(touch_info)
    if use_highlight:
        referent_parts.extend(highlight_info)

    enriched_query = (
        f"{user_query} ({'; '.join(referent_parts)})" if referent_parts else user_query
    )

    # Persist best referent for future deictic ops ("zoom here", "pan here")
    if use_touch:
        best_nv = _pick_best_node_values(touch_nodes)
        if best_nv:
            agent_context["last_touch_node_values"] = best_nv
            agent_context["last_referent_node_values"] = best_nv
    elif use_highlight:
        best_nv = _pick_best_node_values(highlight_nodes)
        if best_nv:
            agent_context["last_referent_node_values"] = best_nv

    # --- LangGraph stream ---
    df = agent_context.get("df")
    df_cols = agent_context.get("df_columns", [])
    head = df.head(6).to_dict(orient="records") if isinstance(df, pd.DataFrame) else []
    tail = df.tail(6).to_dict(orient="records") if isinstance(df, pd.DataFrame) else []

    df_context = {
        "columns": df_cols,
        "x_field": agent_context.get("x_field"),
        "y_field": agent_context.get("y_field"),
        "head": head,
        "tail": tail,
    }

    if not agent_context.get("graph_thread_initialized"):
        messages = [
            {"role": "system", "content": get_system_prompt(json.dumps(df_context))},
            {"role": "user", "content": enriched_query},
        ]
        agent_context["graph_thread_initialized"] = True
    else:
        messages = [{"role": "user", "content": enriched_query}]

    start_time = time.time()
    events = graph.stream(
        {"messages": messages},
        get_graph_config(),
        stream_mode="values",
    )

    response_message = "No response generated."
    for event in events:
        if "messages" in event and event["messages"]:
            last_msg = event["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
                response_message = last_msg.content

    print(f"Streaming completed in {time.time() - start_time:.2f} seconds.")

    # --- Post-processing ---
    response_message = strip_markdown(response_message)
    rewritten = rewrite_long_node_lists_with_gpt(response_message)
    if rewritten != response_message:
        print("Rewrote long list into sentences.")
        response_message = rewritten

    x_col = agent_context.get("x_field")
    y_col = agent_context.get("y_field")
    color_field = agent_context.get("color_field")
    extracted_nodes = extract_highlighted_data_points(
        response_message, df, x_col, y_col, color_col=color_field
    )
    print("Extracted nodes:", extracted_nodes)

    clear_followup()
    return _make_result(
        response=response_message,
        nodes=extracted_nodes,
        touch_used=use_touch,
        highlight_used=use_highlight,
        touch_nodes=touch_nodes if use_touch else None,
        highlight_nodes=highlight_nodes if use_highlight else None,
    )


# =============================================================================
# Entry point
# =============================================================================

def process_user_request(user_input: str) -> dict:
    """
    Parse an incoming user request and route it to the appropriate handler.
    """
    data = json.loads(user_input)

    if "user_request_for_agent" not in data:
        return _make_result(response="Invalid payload received.")

    user_request = data["user_request_for_agent"]
    transcript_data = user_request.get("transcript", {})
    touchdata = user_request.get("touchdata", {})
    highlighted_context = user_request.get("highlighted_context") or {}

    user_query = transcript_data.get(
        "text_transcript", transcript_data.get("transcript", "") or ""
    ).strip()

    ensure_df_headers_in_context()

    base64_image = agent_context.get("image_data")
    classification = classify_query(user_query, has_image=bool(base64_image))
    intent = classification["intent"]

    # Override intent if mid-disambiguation on chart loading
    if agent_context.get("followup_stage") and agent_context.get("followup_topic") == "load_chart":
        intent = "load_chart"

    agent_context["last_intent"] = intent
    print(f"Detected Intent: {intent} | Deictic: {classification['has_deictic']}")

    if intent == "load_chart":
        return _handle_load_chart(user_query)

    if intent == "image_analysis":
        return _handle_image_analysis(user_query, base64_image)

    if intent == "operations":
        return _handle_operations(user_query, touchdata, highlighted_context)

    if intent == "chart_overview":
        return _handle_chart_overview()

    # data_analysis, trend, touch_interaction, general_question
    return _handle_data_query(user_query, touchdata, highlighted_context)
