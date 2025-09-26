import requests
import json
from typing import List, Dict
import os
from tavily import TavilyClient  # assuming this is how the SDK names it

# WEBSEARCH TOOL
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "your_api_key_here").strip()
client = TavilyClient(api_key=TAVILY_API_KEY)

# ---------------- CONFIG ----------------
OLLAMA_URL = "http://10.0.139.104:11434/api/chat"
OLLAMA_MODEL = "gpt-oss:20b"


# ---------------- TOOL: Web Search ----------------
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


# ---------------- AGENT ----------------
class OllamaResearchAgent:
    def __init__(self, model=OLLAMA_MODEL, url=OLLAMA_URL):
        self.model = model
        self.url = url
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
                "temperature": 0.7,
                "stream": False,
            },
        )
        resp.raise_for_status()
        print("[Agent] Response received from Ollama")
        return resp.json()

    def ask(self, question: str) -> str:
        print(f"[Agent] New question: {question.strip()}")
        messages = [
            {"role": "system", "content":
                """You are a research assistant.
If you need external information, respond ONLY with in tool_calls:
  {"name": "web_search", "query": "<your search query>"}

When search results are provided (JSON with title/snippet/url), read them carefully.
- Summarize the key findings.
- Filter out irrelevant or off-topic results.
- Provide references (use titles + URLs).
- Write in clear academic style."""
             },
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
                    results = web_search(query)

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


# ---------------- USAGE ----------------
if __name__ == "__main__":
    agent = OllamaResearchAgent()

    user_request = """
    Summarize the latest breakthroughs in quantum error correction
    and superconducting qubits research. Provide references if possible.
    """

    print("[Main] Asking agent...")
    answer = agent.ask(user_request)
    print("\n=== Agent Answer ===\n")
    print(answer)
