import logging
import time
from tavily import TavilyClient
from config import settings

logger = logging.getLogger(__name__)

_client = None
_RETRIES = 2
_RETRY_DELAY = 1.5  # seconds between attempts


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        _client = TavilyClient(api_key=settings.tavily_api_key)
    return _client


def _search_with_retry(query: str, **kwargs) -> dict:
    last_exc = None
    for attempt in range(1, _RETRIES + 1):
        try:
            return _get_client().search(query=query, **kwargs)
        except Exception as e:
            last_exc = e
            logger.warning("Tavily attempt %d/%d failed: %s", attempt, _RETRIES, e)
            if attempt < _RETRIES:
                time.sleep(_RETRY_DELAY)
    raise last_exc


def search_web(query: str, max_results: int = 5) -> list:
    results = _search_with_retry(query=query, max_results=max_results)
    return results.get("results", [])


def search_images(query: str) -> list[str]:
    results = _search_with_retry(query=query, max_results=5, include_images=True)
    return results.get("images", [])
