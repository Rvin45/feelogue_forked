"""
Data query tool using pandas DataFrame agent.
"""
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.tools import tool

from .config import OPENAI_MODEL_ANALYSIS
from .prompts import get_data_query_prefix

# LLM for CSV/data queries - temperature=0 for reliable tool-call compliance
csv_llm = ChatOpenAI(model=OPENAI_MODEL_ANALYSIS, temperature=0, stop=None)

# Side-channel written by data_query_node before the tool loop runs.
# Holds scalar state fields (x_field, y_field, etc.) so csv_query_tool can
# read them without receiving parameters (tool signature is fixed to query: str).
_state_ref: dict = {}


def update_state_ref(patch: dict) -> None:
    """Called by data_query_node to give csv_query_tool current field metadata."""
    _state_ref.update(patch)


# Cache for the pandas agent executor -- rebuilt only when dataset or columns change.
# dataset_version from _state_ref is the cache key: it increments on every
# layer_data_update so a new DataFrame always gets a fresh executor.
_cached_executor = None
_cached_version = None
_cached_df_id = None
_cached_columns = None


def _get_executor(df, selected_data, columns_to_use: list):
    """Return a pandas agent executor, reusing the cached one when nothing changed."""
    global _cached_executor, _cached_version, _cached_df_id, _cached_columns

    version = _state_ref.get("dataset_version")
    df_id = id(df)
    cols = tuple(columns_to_use)

    if (
        _cached_executor is None
        or version != _cached_version
        or df_id != _cached_df_id
        or cols != _cached_columns
    ):
        color_field = _state_ref.get("color_field")
        df_columns = _state_ref.get("df_columns", [])
        print(f"Building pandas agent executor (version={version}, columns={cols})")
        _cached_executor = create_pandas_dataframe_agent(
            csv_llm,
            selected_data,
            verbose=True,
            allow_dangerous_code=True,
            agent_type="openai-tools",
            prefix=get_data_query_prefix(color_field, df_columns, df),
            max_iterations=6,
            agent_executor_kwargs={"handle_parsing_errors": True},
        )
        _cached_version = version
        _cached_df_id = df_id
        _cached_columns = cols
    else:
        print("Reusing cached pandas agent executor")

    return _cached_executor


@tool
def csv_query_tool(query: str) -> str:
    """
    Handles general queries on the currently loaded chart data.
    Reads the DataFrame from context.get_df(), filters to relevant columns,
    and delegates to a pandas DataFrame agent with python_repl_ast.
    """
    try:
        from .context import get_df
        df = get_df()
        if df is None or df.empty:
            return (
                "I don't have any chart data loaded right now. "
                "Please load a chart first, and then ask your question again."
            )

        x_field = _state_ref.get("x_field") or (df.columns[0] if len(df.columns) > 0 else None)
        y_field = _state_ref.get("y_field") or (df.columns[-1] if len(df.columns) > 1 else None)
        chart_type = (_state_ref.get("chart_type") or "").lower()
        color_field = _state_ref.get("color_field")
        second_column = _state_ref.get("second_column")

        if chart_type in {"bar", "line"} and x_field and y_field and {x_field, y_field}.issubset(df.columns):
            columns_to_use = [x_field, y_field]
            if color_field and color_field in df.columns:
                columns_to_use.append(color_field)
        elif chart_type == "scatter" and second_column and second_column in df.columns:
            columns_to_use = [x_field, y_field, second_column]
        else:
            columns_to_use = list(df.columns)

        if "visible" in df.columns and not df["visible"].all() and "visible" not in columns_to_use:
            columns_to_use.append("visible")

        selected_data = df[columns_to_use]
        executor = _get_executor(df, selected_data, columns_to_use)

        chat_history = _state_ref.get("messages") or []
        result = executor.invoke({"input": query, "chat_history": chat_history})
        if isinstance(result, str):
            return result
        output = result.get("output")
        if output:
            return output
        return "I wasn't able to get a result for that query."

    except Exception as e:
        print(f"csv_query_tool error: {type(e).__name__}: {e}")
        return f"I encountered an error while processing your query: {str(e)}"
