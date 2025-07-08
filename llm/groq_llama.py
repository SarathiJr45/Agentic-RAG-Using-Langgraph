from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv(override=True)

def get_llama_llm(model: str = "llama3-8b-8192", temperature: float = 0.2):
    return ChatGroq(
        model=model,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature,
    )

# llm = get_llama_llm()

# messages = [
#     (
#         "system",
#         "You are a helpful assistant that translates English to French. Translate the user sentence.",
#     ),
#     ("human", "I love programming."),
# ]
# ai_msg = llm.invoke(messages).content
# print(ai_msg)