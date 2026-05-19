import asyncio
import ollama
from langchain_core.callbacks import adispatch_custom_event
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from lib.state import QueryState
from data_loader import embed_texts
from vector_db import QdrantStorage

MAX_REWRITES = 2
_client = ollama.AsyncClient()


async def classify_intent_node(state: QueryState) -> dict:
    resp = await _client.chat(model="llama3.2", messages=[{
        "role": "user",
        "content": (
            "Classify the user message as 'rag' (needs document lookup) or 'chitchat' "
            "(greeting, small talk, opinion, no documents needed).\n"
            "Reply with only one word: 'rag' or 'chitchat'.\n\n"
            f"Message: {state['question']}"
        ),
    }])
    raw = resp["message"]["content"].strip().lower()
    intent = "chitchat" if "chitchat" in raw else "rag"
    return {"intent": intent}


def route_after_classify(state: QueryState) -> str:
    return state.get("intent", "rag")


async def chitchat_node(state: QueryState) -> dict:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Reply conversationally."},
        *(state.get("history") or []),
        {"role": "user", "content": state["question"]},
    ]
    full: list[str] = []
    async for chunk in await _client.chat(model="llama3.2", messages=messages, stream=True):
        token = chunk["message"]["content"]
        if token:
            full.append(token)
            await adispatch_custom_event("token", token)
    await adispatch_custom_event("final_meta", {
        "sources": [],
        "rewrite_count": state.get("rewrite_count", 0),
    })
    return {"answer": "".join(full), "sources": []}


async def retrieve_node(state: QueryState) -> dict:
    query_vec = (await asyncio.to_thread(embed_texts, [state["question"]]))[0]
    result = QdrantStorage().search(query_vec, state.get("top_k", 5))
    return {"contexts": result["contexts"], "sources": result["sources"]}


async def grade_docs_node(state: QueryState) -> dict:
    question = state["question"]
    relevant_contexts: list[str] = []
    relevant_sources: list[str] = []
    for i, ctx in enumerate(state["contexts"]):
        resp = await _client.chat(model="llama3.2", messages=[{
            "role": "user",
            "content": (
                f"Is this document relevant to the question? Answer only 'yes' or 'no'.\n\n"
                f"Question: {question}\n\nDocument: {ctx[:600]}"
            ),
        }])
        if "yes" in resp["message"]["content"].lower():
            relevant_contexts.append(ctx)
            if i < len(state.get("sources", [])):
                relevant_sources.append(state["sources"][i])
    return {"relevant_contexts": relevant_contexts, "relevant_sources": relevant_sources}


def route_after_grading(state: QueryState) -> str:
    if not state.get("relevant_contexts") and state.get("rewrite_count", 0) < MAX_REWRITES:
        return "rewrite"
    return "generate"


async def rewrite_query_node(state: QueryState) -> dict:
    resp = await _client.chat(model="llama3.2", messages=[{
        "role": "user",
        "content": (
            "Rewrite this query to retrieve more relevant documents.\n"
            f"Original: {state.get('original_question', state['question'])}\n"
            f"Current: {state['question']}\n"
            "Return only the rewritten query, nothing else."
        ),
    }])
    return {
        "question": resp["message"]["content"].strip(),
        "rewrite_count": state.get("rewrite_count", 0) + 1,
    }


async def generate_node(state: QueryState) -> dict:
    contexts = state.get("relevant_contexts") or state.get("contexts", [])
    sources = state.get("relevant_sources") or state.get("sources", [])
    context_block = "\n\n".join(f"- {c}" for c in contexts)
    original_q = state.get("original_question", state["question"])
    prompt = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {original_q}\n"
        "Answer concisely using the context above."
    )
    messages = [
        {"role": "system", "content": "You answer questions using only the provided context."},
        *(state.get("history") or []),
        {"role": "user", "content": prompt},
    ]
    full: list[str] = []
    async for chunk in await _client.chat(model="llama3.2", messages=messages, stream=True):
        token = chunk["message"]["content"]
        if token:
            full.append(token)
            await adispatch_custom_event("token", token)
    await adispatch_custom_event("final_meta", {
        "sources": sources,
        "rewrite_count": state.get("rewrite_count", 0),
    })
    return {"answer": "".join(full), "sources": sources}


def build_query_graph(checkpointer: MemorySaver):
    g = StateGraph(QueryState)
    g.add_node("classify_intent", classify_intent_node)
    g.add_node("chitchat", chitchat_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade_docs", grade_docs_node)
    g.add_node("rewrite_query", rewrite_query_node)
    g.add_node("generate", generate_node)
    g.add_edge(START, "classify_intent")
    g.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {"chitchat": "chitchat", "rag": "retrieve"},
    )
    g.add_edge("chitchat", END)
    g.add_edge("retrieve", "grade_docs")
    g.add_conditional_edges(
        "grade_docs",
        route_after_grading,
        {"rewrite": "rewrite_query", "generate": "generate"},
    )
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("generate", END)
    return g.compile(checkpointer=checkpointer)
