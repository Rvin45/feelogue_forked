"""
Runtime state shared across modules.

agent_context has been replaced by AgentState in graph.py.
This module now holds only the live DataFrame (excluded from LangGraph
checkpoints because pandas objects can't be serialized by MemorySaver)
and the current thread ID used to address the graph checkpoint.
"""
import pandas as pd

# Module-level DataFrame ref -- never put into AgentState
_df: pd.DataFrame | None = None

# Current LangGraph thread ID -- updated whenever a new layer loads
_current_thread_id: str = "default-1"


def get_df() -> pd.DataFrame | None:
    return _df


def get_current_config() -> dict:
    return {
        "configurable": {"thread_id": _current_thread_id},
        "recursion_limit": 30,
    }


def update_dataframe_from_layer(msg: dict) -> dict:
    """
    Build a DataFrame from a layer_data_update MQTT message.
    Updates the module-level df ref and returns a serializable metadata
    patch suitable for graph.update_state().
    """
    global _df, _current_thread_id

    layer_name = msg.get("layer_name", "unnamed")
    chart_type = msg.get("chart_type", "line")
    data_points = msg.get("data_points") or msg.get("data") or []

    if not data_points:
        print(f"Warning: No data points in layer update for '{layer_name}'")
        return {}

    x_field = msg.get("x_field")
    y_field = msg.get("y_field")

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
    _df = df

    # Bump thread ID so the new dataset gets a fresh conversation context
    from .graph import graph, clear_graph_thread
    old_thread_id = _current_thread_id
    new_config = get_current_config()
    # Derive version from the new thread name we're about to set
    import re
    m = re.search(r"-(\d+)$", old_thread_id)
    old_version = int(m.group(1)) if m else 0
    new_version = old_version + 1
    _current_thread_id = f"{layer_name}-{new_version}"

    if old_thread_id and old_thread_id != _current_thread_id:
        clear_graph_thread(old_thread_id)

    metadata_patch = {
        "x_field": x_field,
        "y_field": y_field,
        "color_field": msg.get("series_field"),
        "df_columns": list(df.columns),
        "chart_type": chart_type,
        "active_layer": layer_name,
        "dataset_version": new_version,
    }

    # Push metadata into the new thread's checkpoint so it's available on first invoke
    graph.update_state(get_current_config(), metadata_patch)

    print(f"DataFrame updated: {len(df)} rows, columns: {list(df.columns)}")
    return metadata_patch
