import requests
import json
from typing import List, Dict
import os
from tavily import TavilyClient  # assuming this is how the SDK names it


class SimpleOllamaResearchAgent:
    def __init__(self, model="gpt-oss:20b", url="http://10.0.139.104:11434/api/chat"):
        self.model = model
        self.url = url
        self.client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", "your_api_key_here"))
        self.SYSTEM_PROMPT = """You are a research assistant.

        When answering:
        1. If you need external information, respond ONLY with a tool call:
           {"name": "web_search", "query": "<your search query>"}

        2. When search results are provided (JSON with title/snippet/url):
           - Carefully read them.
           - Write a clear academic-style summary (2–4 paragraphs).
           - Ground all claims in the provided results.
           - Cite sources inline using (Title, URL).
           - Do NOT fabricate references. Only use titles + URLs from the given results.
           - Do NOT output JSON unless explicitly asked.
           - Your final answer must be natural language prose, not a list of citations.

        Your goal: deliver a concise research-style overview with correct references."""
        print(f"[Agent] Initialized with model='{model}' url='{url}'")

    def chat(self, messages: List[Dict]) -> dict:
        """
        Send messages to Ollama and return the raw JSON response.
        """
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

            # --- handle new tool_call field ---
            if "tool_calls" in msg and msg["tool_calls"]:
                tool_call = msg["tool_calls"][0]
                func = tool_call["function"]["name"]
                args = tool_call["function"]["arguments"]

                if func == "web_search":
                    query = args.get("query")
                    print(f"[Agent] → Detected web_search request: {query}")
                    results = self.web_search(query)

                    # Append the tool request + result
                    messages.append({"role": "assistant", "content": f"[tool_call: websearch] {query}"})
                    messages.append({"role": "tool", "content": json.dumps(results, indent=2)})
                    continue

            # --- fall back to plain text ---
            content = msg.get("content", "")
            if content.strip():
                print("[Agent] Final answer detected, returning result.")
                return content

            print("[Agent] No usable output, returning raw response.")
            return str(response)

    def web_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        resp = self.client.search(query, max_results=max_results, include_raw_content=True)
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
