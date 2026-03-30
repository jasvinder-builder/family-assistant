from tavily import TavilyClient
from config import settings

_client = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        _client = TavilyClient(api_key=settings.tavily_api_key)
    return _client


def search_web(query: str, max_results: int = 5) -> list:
    results = _get_client().search(query=query, max_results=max_results)
    return results.get("results", [])


def search_images(query: str) -> list[str]:
    """Returns a list of image URLs."""
    results = _get_client().search(
        query=query,
        max_results=5,
        include_images=True,
    )
    return results.get("images", [])
