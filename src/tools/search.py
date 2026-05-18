from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo. Use this to find current information on any topic."""
    from ddgs import DDGS

    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return "No results found."
        output = []
        for i, r in enumerate(results, 1):
            output.append(f"{i}. {r['title']}\n   {r['href']}\n   {r['body']}")
        return "\n\n".join(output)
    except Exception as e:
        return f"Search error: {e}"
