import os, json
from tavily import TavilyClient  # assuming this is how the SDK names it
from typing import List, Dict

# (Optional) read from env
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "your_api_key_here").strip()
print(TAVILY_API_KEY)
client = TavilyClient(api_key=TAVILY_API_KEY)


def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    resp = client.search(query, max_results=max_results, include_raw_content=True)
    results = []
    preview = []
    for r in resp.get("results", []):
        result = {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "raw_content": r.get("raw_content", "")
        }
        results.append(result)
        result = result.copy()
        result["raw_content"] = result["raw_content"][:300] if result["raw_content"] else "EMPTY"
        preview.append(result)

    print("[WEB SEARCH] Search results:\n" + json.dumps(preview, indent=4, sort_keys=True))
    del preview
    return results


print(web_search("ukrain war 2025"))
