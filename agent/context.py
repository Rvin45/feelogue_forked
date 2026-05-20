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

# Module-level image ref -- survives thread ID rotations (image arrives via rtd_data_for_agent
# before or independently of layer_data_update, so we can't rely on the checkpoint alone)
_image_data: str | None = None
_image_format: str | None = None

# Current LangGraph thread ID -- updated only when the active layer changes
_current_thread_id: str = "default-1"

# Track which layer owns the current thread so same-layer data updates
# don't rotate the thread (and lose conversation history).
_current_layer_name: str | None = None
_dataset_version: int = 0

# Chart metadata index -- survives thread rotations (boot message arrives once,
# before any layer loads, so the thread it lands in gets cleared on first layer switch).
_chart_metadata_index: dict | None = None


def get_df() -> pd.DataFrame | None:
    return _df


def set_image_data(image_data: str | None, image_format: str | None = "png") -> None:
    """Store the latest chart image so it survives thread ID rotations."""
    global _image_data, _image_format
    _image_data = image_data
    _image_format = image_format or "png"


def set_chart_metadata_index(index: dict) -> None:
    """Store the chart catalog so it survives thread rotations."""
    global _chart_metadata_index
    _chart_metadata_index = index


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

    Thread rotation only happens when the active layer *name* changes.
    Updates to the same layer reuse the existing thread so conversation
    history is preserved across data refreshes.
    """
    global _df, _current_thread_id, _current_layer_name, _dataset_version

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

    from .graph import graph, clear_graph_thread

    _dataset_version += 1
    is_new_layer = (_current_layer_name != layer_name)

    if is_new_layer:
        old_thread_id = _current_thread_id
        _current_thread_id = f"{layer_name}-{_dataset_version}"
        _current_layer_name = layer_name
        if old_thread_id and old_thread_id != _current_thread_id:
            clear_graph_thread(old_thread_id)
        print(f"[context] Layer changed '{old_thread_id}' -> '{_current_thread_id}' (new thread)")
    else:
        print(f"[context] Same layer '{layer_name}' updated (thread kept: '{_current_thread_id}')")

    metadata_patch = {
        "x_field": x_field,
        "y_field": y_field,
        "color_field": msg.get("series_field"),
        "df_columns": list(df.columns),
        "chart_type": chart_type,
        "active_layer": layer_name,
        "dataset_version": _dataset_version,
    }

    # Carry image data forward -- it may have arrived before this message.
    if _image_data:
        metadata_patch["image_data"] = _image_data
        metadata_patch["image_format"] = _image_format

    # Always carry the chart catalog forward -- the boot message lands in the
    # initial thread which gets cleared on the first layer switch.
    if _chart_metadata_index:
        metadata_patch["chart_metadata_index"] = _chart_metadata_index

    graph.update_state(get_current_config(), metadata_patch)

    print(f"DataFrame updated: {len(df)} rows, columns: {list(df.columns)}")
    return metadata_patch
