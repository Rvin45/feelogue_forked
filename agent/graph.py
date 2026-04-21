"""
LangGraph setup for conversational data analysis.
"""
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from .config import OPENAI_MODEL_ANALYSIS
from .data_query import csv_query_tool


class State(TypedDict):
    """Graph state - a list of messages."""
    messages: Annotated[list, add_messages]


# Create main LLM with tool awareness
main_llm = ChatOpenAI(model=OPENAI_MODEL_ANALYSIS, temperature=0.5)

# Define tools
tools = [csv_query_tool]

# Bind tools to LLM
llm_with_tools = main_llm.bind_tools(tools)


def chatbot(state: State):
    """Main chatbot node - processes messages and optionally calls tools."""
    new_message = llm_with_tools.invoke(state["messages"])
    return {"messages": [new_message]}


# Build the graph
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))

# Add edges
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")

# Compile with memory checkpointer
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


def clear_graph_thread(thread_id: str) -> None:
    """
    Remove all checkpoint data for a thread from MemorySaver.
    Call this when rotating to a new graph_thread_id so old
    conversation history doesn't accumulate in memory indefinitely.
    """
    to_delete = [
        k for k in memory.storage
        if isinstance(k, tuple) and len(k) > 0 and k[0] == thread_id
    ]
    for k in to_delete:
        del memory.storage[k]
    if to_delete:
        print(f"Cleared {len(to_delete)} checkpoint(s) for graph thread '{thread_id}'")
