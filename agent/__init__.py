"""
RTD Conversational Agent

A multimodal agent for accessible data visualization on refreshable tactile displays.
Combines touch and speech interaction for exploring charts and data.
"""

from .context import (
    agent_context,
    clear_followup,
    mark_followup,
    set_dataframe,
    ensure_df_headers_in_context,
    get_xy_cols,
    get_graph_config,
    update_dataframe_from_layer,
)

from .intent import classify_query, classify_intent, detect_deictic_reference

from .touch_context import (
    collect_touch_nodes,
    collect_highlight_nodes,
    pick_best_referent_node,
)

from .operations import (
    build_operations_rtd_command,
    resolve_operation_targets_to_values,
    build_operation_ack,
)

from .data_query import csv_query_tool

from .chart_loader import analyze_user_intent_with_context

from .postprocessing import (
    rewrite_long_node_lists_with_gpt,
    extract_highlighted_data_points,
)

from .graph import graph

from .orchestrator import process_user_request

from .mqtt_handler import run, create_mqtt_client, publish_message

__version__ = "0.1.0"

__all__ = [
    # Context
    "agent_context",
    "clear_followup",
    "mark_followup",
    "set_dataframe",
    "ensure_df_headers_in_context",
    "get_xy_cols",
    "get_graph_config",
    "update_dataframe_from_layer",
    # Intent
    "classify_query",
    "classify_intent",
    "detect_deictic_reference",
    # Touch
    "collect_touch_nodes",
    "collect_highlight_nodes",
    "pick_best_referent_node",
    # Operations
    "build_operations_rtd_command",
    "resolve_operation_targets_to_values",
    "build_operation_ack",
    # Data
    "csv_query_tool",
    # Chart loading
    "analyze_user_intent_with_context",
    # Post-processing
    "rewrite_long_node_lists_with_gpt",
    "extract_highlighted_data_points",
    # Graph
    "graph",
    # Orchestrator
    "process_user_request",
    # MQTT
    "run",
    "create_mqtt_client",
    "publish_message",
]
