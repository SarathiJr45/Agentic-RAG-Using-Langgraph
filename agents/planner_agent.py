import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.prompts import ChatPromptTemplate
from llm.groq_llama import get_llama_llm

from dotenv import load_dotenv

load_dotenv()

llm = get_llama_llm()

# Define output structure


# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a planner agent in a RAG system. Given a user's question, Break it down in to a topic and a refined query."),
    ("user", """User query: {question}
""")
])

# Chain = Prompt → LLM → JSON Parser
chain = prompt | llm 

def plan_user_query(question: str) -> dict:
    return chain.invoke({"question": question})
  
plan_user_query("explain llm and ai agents on its working").content


# if __name__ == "__main__":
#     result = plan_user_query("What were the causes of the French Revolution?")
#     print(result)
