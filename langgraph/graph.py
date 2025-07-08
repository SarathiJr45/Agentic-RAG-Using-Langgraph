
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_core.messages import BaseMessage
from agents.planner_agent import plan_user_query
from agents.retriver_agent import load_vector_store, retrieve_context
from agents.websearch_agent import web_search
from agents.qa_agent import answer_from_context
from sentence_transformers import SentenceTransformer

index, chunks = load_vector_store()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def planner_node(state):
    user_query = state["query"]
    
    if isinstance(user_query, BaseMessage):
        user_query = user_query.content
        
    refined = plan_user_query(user_query)
    return {"refined_query": refined}

def retriever_node(state):
    query = state["refined_query"]
    
    if isinstance(query, BaseMessage):
        query = query.content
        
    context = retrieve_context(query, index, chunks, embed_model)
    return {"context": context}

def qa_node(state):
    query = state["refined_query"]
    
    if isinstance(query, BaseMessage):
        query = query.content
        
    context = state.get("context", "")
    answer = answer_from_context(context, query)

    if "[The context does not contain that information.]" in answer:  # Check trigger
        return {"needs_web": True}
    
    return {"answer": answer, "needs_web": False}

def websearch_node(state):
    query = state["refined_query"]
    
    if isinstance(query, BaseMessage):
        query = query.content
        
    context = web_search(query)
    return {"context": context}

# --- Graph Schema ---


class AgenticRAGState(TypedDict):
    query: str
    refined_query: str
    context: str
    answer: str
    needs_web: bool

builder = StateGraph(AgenticRAGState)


builder.add_node("planner", planner_node)
builder.add_node("retriever", retriever_node)
builder.add_node("qa", qa_node)
builder.add_node("websearch", websearch_node)

builder.set_entry_point("planner")
builder.add_edge("planner", "retriever")
# builder.add_edge("retriever","websearch")
builder.add_edge("retriever", "qa")

# Conditional edge: fallback to web if QA fails
builder.add_conditional_edges("qa", lambda state: state["needs_web"], {
    True: "websearch",
    False: END
})

# Websearch leads to 2nd QA pass
builder.add_edge("websearch", "qa")

builder.set_finish_point("qa")

graph = builder.compile()


