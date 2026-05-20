import asyncio
import logging
import ollama
from langchain_core.callbacks import adispatch_custom_event
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from lib.state import QueryState
from data_loader import embed_texts
from vector_db import QdrantStorage

logger = logging.getLogger(__name__)

MAX_REWRITES = 2
_client = ollama.AsyncClient()


async def classify_intent_node(state: QueryState) -> dict:
    """LangGraph node: classify the question as 'rag' or 'chitchat' using llama3.2."""
    logger.info("Node classify_intent: question='%s'", state["question"][:100])
    await adispatch_custom_event("status", "Classifying intent...")
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
    logger.info("Intent classified as '%s'", intent)
    return {"intent": intent}


def route_after_classify(state: QueryState) -> str:
    """Route to 'chitchat' or 'rag' branch based on classified intent."""
    return state.get("intent", "rag")


async def chitchat_node(state: QueryState) -> dict:
    """LangGraph node: handle small talk with streaming tokens; skips document retrieval."""
    logger.info("Node chitchat: question='%s'", state["question"][:100])
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
    logger.info("Node chitchat complete: %d tokens", len(full))
    await adispatch_custom_event("final_meta", {
        "source_refs": [],
        "rewrite_count": state.get("rewrite_count", 0),
    })
    return {"answer": "".join(full), "source_refs": []}


async def retrieve_node(state: QueryState) -> dict:
    """LangGraph node: embed the current question and search Qdrant for top-k contexts."""
    logger.info("Node retrieve: question='%s' top_k=%d", state["question"][:100], state.get("top_k", 5))
    await adispatch_custom_event("status", "Searching documents...")
    query_vec = (await asyncio.to_thread(embed_texts, [state["question"]]))[0]
    result = QdrantStorage().search(query_vec, state.get("top_k", 5))
    logger.info("Node retrieve complete: %d contexts found", len(result["contexts"]))
    return {"contexts": result["contexts"], "source_refs": result["source_refs"]}


async def grade_docs_node(state: QueryState) -> dict:
    """LangGraph node: grade all retrieved chunks for relevance in parallel using llama3.2."""
    logger.info("Node grade_docs: grading %d chunks", len(state["contexts"]))
    await adispatch_custom_event("status", f"Grading {len(state['contexts'])} chunks...")
    question = state.get("original_question", state["question"])

    async def _grade(ctx: str) -> bool:
        """Ask llama3.2 whether a context chunk is relevant to the original question."""
        resp = await _client.chat(model="llama3.2", messages=[{
            "role": "user",
            "content": (
                f"Is this document relevant to the question? Answer only 'yes' or 'no'.\n\n"
                f"Question: {question}\n\nDocument: {ctx[:600]}"
            ),
        }])
        return "yes" in resp["message"]["content"].lower()

    results = await asyncio.gather(*[_grade(ctx) for ctx in state["contexts"]])

    relevant_contexts: list[str] = []
    for i, (ctx, is_relevant) in enumerate(zip(state["contexts"], results)):
        if is_relevant:
            relevant_contexts.append(ctx)
    logger.info("Node grade_docs complete: %d/%d chunks relevant", len(relevant_contexts), len(state["contexts"]))
    return {"relevant_contexts": relevant_contexts}


def route_after_grading(state: QueryState) -> str:
    """Route to 'rewrite' if no relevant docs remain and rewrites aren't exhausted, else 'generate'."""
    if not state.get("relevant_contexts") and state.get("rewrite_count", 0) < MAX_REWRITES:
        return "rewrite"
    return "generate"


async def rewrite_query_node(state: QueryState) -> dict:
    """LangGraph node: rewrite the current question using llama3.2 to improve retrieval."""
    logger.info("Node rewrite_query: attempt %d, original='%s'", state.get("rewrite_count", 0) + 1, state.get("original_question", state["question"])[:100])
    await adispatch_custom_event("status", f"Rewriting query (attempt {state.get('rewrite_count', 0) + 1})...")
    resp = await _client.chat(model="llama3.2", messages=[{
        "role": "user",
        "content": (
            "Rewrite this query to retrieve more relevant documents.\n"
            f"Original: {state.get('original_question', state['question'])}\n"
            f"Current: {state['question']}\n"
            "Return only the rewritten query, nothing else."
        ),
    }])
    new_question = resp["message"]["content"].strip()
    logger.info("Node rewrite_query complete: new_question='%s'", new_question[:100])
    return {
        "question": new_question,
        "rewrite_count": state.get("rewrite_count", 0) + 1,
    }


async def generate_node(state: QueryState) -> dict:
    """LangGraph node: generate a grounded answer from relevant contexts with streaming tokens."""
    contexts = state.get("relevant_contexts") or state.get("contexts", [])
    logger.info("Node generate: %d contexts, question='%s'", len(contexts), state.get("original_question", state["question"])[:100])
    await adispatch_custom_event("status", "Generating answer...")
    source_refs = state.get("source_refs", [])
    context_block = "\n\n".join(f"- {c}" for c in contexts)
    original_q = state.get("original_question", state["question"])
    prompt = (
        "Answer the question based on the context below. "
        "If the context does not contain enough information, say so clearly.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {original_q}"
    )
    messages = [
        {"role": "system", "content": (
            "You are a document QA assistant. "
            "Answer using the provided context as your primary source. "
            "You may use general knowledge to reason and add helpful context, "
            "but prioritize and ground your answer in the provided context."
        )},
        *(state.get("history") or []),
        {"role": "user", "content": prompt},
    ]
    full: list[str] = []
    async for chunk in await _client.chat(model="llama3.2", messages=messages, stream=True):
        token = chunk["message"]["content"]
        if token:
            full.append(token)
            await adispatch_custom_event("token", token)
    logger.info("Node generate complete: %d tokens", len(full))
    await adispatch_custom_event("final_meta", {
        "source_refs": source_refs,
        "rewrite_count": state.get("rewrite_count", 0),
    })
    return {"answer": "".join(full), "source_refs": source_refs}


def build_query_graph(checkpointer: MemorySaver):
    """Build and compile the multi-node query graph with intent routing, grading, and rewrite loops."""
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
