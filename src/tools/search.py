import logging

from langchain_core.tools import tool

from src.utils.retry import retry_with_backoff

logger = logging.getLogger("agent")


@retry_with_backoff(max_retries=3, base_delay=1.0)
def _search_ddgs(query: str) -> list[dict]:
    from ddgs import DDGS

    return DDGS().text(query, max_results=5)


@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo. Use this to find current information on any topic."""
    try:
        results = _search_ddgs(query)
        if not results:
            return "No results found."
        output = []
        for i, r in enumerate(results, 1):
            output.append(f"{i}. {r['title']}\n   {r['href']}\n   {r['body']}")
        return "\n\n".join(output)
    except (ConnectionError, TimeoutError, OSError) as e:
        logger.error("web_search failed for query=%r: %s", query, e, exc_info=True)
        return f"Search error after retries: {type(e).__name__}: {e}"
    except Exception as e:
        logger.error(
            "web_search unexpected error for query=%r: %s", query, e, exc_info=True
        )
        return f"Search error: {type(e).__name__}: {e}"
