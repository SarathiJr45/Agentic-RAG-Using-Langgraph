# agents/websearch_agent.py

from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the Tavily tool with API key
tavily_tool = TavilySearch(
    api_key=os.getenv("TAVILY_API_KEY"),
    max_results=5,
    topic="general"
)

def web_search(query: str) -> str:
    result = tavily_tool.invoke({"query": query})
    
    # Extract only the content from each search result
    content_only = []
    
    if 'results' in result:
        for item in result['results']:
            if 'content' in item and item['content']:
                content_only.append(item['content'])
    
    # Join all content pieces with separators
    return "\n\n\n".join(content_only)

# Test run
if __name__ == "__main__":
    query = "What happened at the last Wimbledon?"
    
    print("🔍 Content Only Result:\n")
    print(web_search(query))
    