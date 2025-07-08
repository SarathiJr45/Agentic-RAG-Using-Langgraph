from langgraph.graph import graph

while True:
    q = input("\n❓ Ask me something (or 'exit'): ")
    if q.lower() in ["exit", "quit"]:
        break
    result = graph.invoke({"query": q})
    print("\n✅ Answer:\n", result.get("answer", "No answer"))
