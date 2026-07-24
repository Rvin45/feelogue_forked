"""
RTD Conversational Agent

A multimodal agent for accessible data visualization on refreshable tactile displays.
Combines touch and speech interaction for exploring charts and data.
"""

from .context import (
    get_df,
    get_current_config,
    update_dataframe_from_layer,
)

from .intent import classify_query, classify_intent, detect_deictic_reference

from .touch_context import (
    collect_touch_nodes,
    collect_highlight_nodes,
)

from .operations import (
    parse_operation_response,
    build_operation_ack,
)

from .data_query import csv_query_tool

from .chart_loader import analyze_user_intent_with_context

from .postprocessing import (
    rewrite_long_node_lists_with_gpt,
    parse_data_query_response,
)

from .graph import graph

from .orchestrator import process_user_request

from .mqtt_handler import run, create_mqtt_client, publish_message

__version__ = "0.1.0"

__all__ = [
    # Context
    "get_df",
    "get_current_config",
    "update_dataframe_from_layer",
    # Intent
    "classify_query",
    "classify_intent",
    "detect_deictic_reference",
    # Touch
    "collect_touch_nodes",
    "collect_highlight_nodes",
    # Operations
    "parse_operation_response",
    "build_operation_ack",
    # Data
    "csv_query_tool",
    # Chart loading
    "analyze_user_intent_with_context",
    # Post-processing
    "rewrite_long_node_lists_with_gpt",
    "parse_data_query_response",
    # Graph
    "graph",
    # Orchestrator
    "process_user_request",
    # MQTT
    "run",
    "create_mqtt_client",
    "publish_message",
]
