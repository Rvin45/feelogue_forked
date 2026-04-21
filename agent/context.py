"""
Shared agent context and state management.
"""
from typing import Optional

# Global context dict - holds runtime state
agent_context: dict = {}


def clear_followup():
    """Clear follow-up conversation state."""
    agent_context.pop("followup_stage", None)
    agent_context.pop("followup_topic", None)
    agent_context.pop("pending_chart_options", None)


def mark_followup(topic: Optional[str] = None):
    """Mark that we're in a follow-up conversation."""
    agent_context["followup_stage"] = True
    if topic:
        agent_context["followup_topic"] = topic


def set_dataframe(df):
    """Register a pandas DataFrame in context."""
    import pandas as pd
    agent_context["df"] = df
    agent_context["df_columns"] = list(df.columns)
    print(f"DataFrame registered with columns: {agent_context['df_columns']}")


def ensure_df_headers_in_context():
    """Ensure df_columns is up to date with the current DataFrame."""
    df = agent_context.get("df")
    if df is not None and hasattr(df, "columns"):
        cols = list(df.columns)
        if cols != agent_context.get("df_columns"):
            agent_context["df_columns"] = cols
    else:
        agent_context["df_columns"] = []


def register_dataset_columns(cols):
    """Manually register column names."""
    agent_context["df_columns"] = list(cols)


def get_xy_cols() -> tuple:
    """Get x and y column names from context."""
    x = agent_context.get("x_field") or agent_context.get("first_column")
    y = agent_context.get("y_field") or agent_context.get("selected_column")
    return x, y


def get_graph_config() -> dict:
    """Get LangGraph config with thread ID."""
    return {
        "configurable": {"thread_id": agent_context.get("graph_thread_id", "default-1")},
        "recursion_limit": 25,
    }


def update_dataframe_from_layer(msg: dict):
    """
    Update agent context from a layer_data_update message.
    Builds the runtime DataFrame and sets chart metadata.
    """
    import pandas as pd

    layer_name = msg.get("layer_name", "unnamed")
    chart_type = msg.get("chart_type", "line")
    data_points = msg.get("data_points") or msg.get("data") or []

    if not data_points:
        print(f"Warning: No data points in layer update for '{layer_name}'")
        return

    # Read field names from the message first (Unity sends these)
    x_field = msg.get("x_field")
    y_field = msg.get("y_field")

    # Fall back to guessing from data point keys if not provided
    if not x_field or not y_field:
        sample = data_points[0]
        keys = list(sample.keys())
        if not x_field:
            for k in keys:
                kl = k.lower()
                if kl in ("x", "date", "time", "year", "quarter", "month", "period"):
                    x_field = k
                    break
            if not x_field and len(keys) >= 1:
                x_field = keys[0]
        if not y_field:
            for k in keys:
                kl = k.lower()
                if kl in ("y", "value", "amount", "count", "rate"):
                    y_field = k
                    break
            if not y_field and len(keys) >= 2:
                y_field = keys[1]

    df = pd.DataFrame(data_points)

    agent_context["df"] = df
    agent_context["df_columns"] = list(df.columns)
    agent_context["x_field"] = x_field
    agent_context["y_field"] = y_field
    agent_context["first_column"] = x_field
    agent_context["selected_column"] = y_field
    agent_context["chart_type"] = chart_type
    agent_context["active_layer"] = layer_name
    agent_context["color_field"] = msg.get("series_field")  # None when absent
    agent_context["dataset_version"] = agent_context.get("dataset_version", 0) + 1
    old_thread_id = agent_context.get("graph_thread_id")
    agent_context["graph_thread_id"] = f"{layer_name}-{agent_context['dataset_version']}"
    agent_context["graph_thread_initialized"] = False

    # Free memory for the old thread's LangGraph checkpoints
    if old_thread_id:
        from .graph import clear_graph_thread
        clear_graph_thread(old_thread_id)

    print(f"DataFrame updated: {len(df)} rows, columns: {list(df.columns)}")
