# agents/qa_agent.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from llm.groq_llama import get_llama_llm
from langchain_core.prompts import ChatPromptTemplate

# Load the LLaMA LLM (via langchain-groq)
llm = get_llama_llm()

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research assistant. Always answer based on the provided context."),
    ("user", """Context:
{context}

Question:
{question}

Answer the question in a clear and concise manner, only using the context. If the answer isn't found, say "The context does not contain that information." """)
])

# Chain: Prompt -> LLM
qa_chain = prompt | llm

# Main callable QA function
def answer_from_context(context: str, question: str) -> str:
    return qa_chain.invoke({
        "context": context,
        "question": question
    }).content


if __name__ == "__main__":
    from agents.retriver_agent import load_vector_store, retrieve_context
    from sentence_transformers import SentenceTransformer

    # Load retriever index & model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, chunks = load_vector_store()

    # User query
    question = "What are the policies available?"

    # Retrieve relevant context
    context = retrieve_context(question, index, chunks, model, k=4)

    # Get answer from QA agent
    answer = answer_from_context(context, question)
    print("Answer:\n", answer)
