import requests
import json
from typing import List, Dict
import os
from tavily import TavilyClient  # Tavily SDK
from datetime import datetime


def static_filter(results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove empty, duplicate, or obviously irrelevant results."""
    seen = set()
    filtered = []
    for r in results:
        title = (r.get("title") or "").strip().lower()
        url = (r.get("url") or "").strip()

        if not r.get("raw_content"):  # empty content
            continue
        if "buy" in title or "course" in title or "company" in title:
            continue
        if (title, url) in seen:  # duplicate
            continue

        seen.add((title, url))
        filtered.append(r)
    return filtered


class OllamaResearchAgent:
    def __init__(self, model="gpt-oss:20b", url="http://10.0.139.104:11434/api/chat", use_llm_filter=False):
        self.client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", "your_api_key_here"))
        self.model = model
        self.url = url
        self.use_llm_filter = use_llm_filter
        self.SYSTEM_PROMPT = f"""You are a research assistant.
The current date is: {datetime.now().strftime("%d.%m.%Y")}.

When answering:
1. If you need external information, respond ONLY with a JSON tool call in this exact format:
   {{
     "name": "web_search",
     "query": "<your search query>"
   }}

   - The value of "name" must ALWAYS be exactly "web_search".
   - Do NOT output any other fields, roles, or text.

2. When search results are provided (JSON with title/snippet/url):
   - Carefully read them.
   - Write a clear academic-style summary (2–4 paragraphs).
   - Ground all claims in the provided results.
   - Cite sources inline using (Title, URL).
   - Always include the source title + hyperlink inline in parentheses.
   - Do NOT fabricate references. Only use titles + URLs from the given results.
   - Do NOT output JSON unless explicitly asked.
   - Your final answer must be natural language prose, not a list of citations.

Your goal: deliver a concise research-style overview with correct references.
        """
        print(f"[Agent] Initialized with model='{model}' url='{url}'")

    def __str__(self):
        return "OllamaResearchAgent"

    def chat(self, messages: List[Dict]) -> dict:
        print(f"[Agent] Sending {len(messages)} messages to Ollama...")
        resp = requests.post(
            self.url,
            headers={"Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.3,
                "stream": False,
            },
        )
        resp.raise_for_status()
        print("[Agent] Response received from Ollama")
        return resp.json()

    def ask(self, question: str) -> str:
        print(f"[Agent] New question: {question.strip()}")
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]

        while True:
            print("[Agent] Sending messages to LLM...")
            response = self.chat(messages)
            print("DEBUG", json.dumps(response, indent=2, sort_keys=True))
            msg = response["message"]

            if "tool_calls" in msg and msg["tool_calls"]:
                tool_call = msg["tool_calls"][0]
                func = tool_call["function"]["name"]
                args = tool_call["function"]["arguments"]

                if func == "web_search":
                    query = args.get("query")
                    print(f"[Agent] → Detected web_search request: {query}")
                    results = self.web_search(query)

                    messages.append({"role": "assistant", "content": f"[tool_call: websearch] {query}"})
                    messages.append({"role": "tool", "content": json.dumps(results, indent=2)})
                    continue

            content = msg.get("content", "")
            if content.strip():
                print("[Agent] Final answer detected, returning result.")
                return content

            print("[Agent] No usable output, returning raw response.")
            return str(response)

    def web_search(self, query: str, max_results: int = 8) -> List[Dict[str, str]]:
        resp = self.client.search(query, max_results=max_results, include_raw_content=True)
        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "raw_content": r.get("raw_content", "")
            }
            for r in resp.get("results", [])
        ]

        # static filtering
        results = static_filter(results)

        # optional LLM-assisted filtering
        if self.use_llm_filter:
            results = self.llm_relevance_check(query, results)

        # Debug preview
        preview = [
            {
                "title": r["title"],
                "url": r["url"],
                "raw_content": r["raw_content"][:300] if r["raw_content"] else "EMPTY"
            }
            for r in results
        ]
        print("[WEB SEARCH] Search results (filtered):\n" + json.dumps(preview, indent=4, sort_keys=True))
        return results

    def llm_relevance_check(self, question: str, results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Optional LLM filter: ask the model if each result is relevant."""
        kept = []
        for r in results:
            prompt = f"""You are evaluating a search result for relevance.

    Question: {question}

    Result:
    Title: {r.get('title')}
    URL: {r.get('url')}
    Snippet: {r.get('raw_content')[:300]}

    Is this result relevant to answering the question?
    Answer with YES or NO only.
    """
            resp = self.chat([{"role": "system", "content": prompt}])
            print(f"[Agent] Filtering search results [{r.get("title")}]")
            decision = resp.get("message", {}).get("content", "").strip().upper()
            if decision.startswith("YES"):
                kept.append(r)
        return kept
