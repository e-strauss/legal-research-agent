import json
import os
from typing import List, Dict
from datetime import datetime
from tavily import TavilyClient

from .llm import LLMClient


def static_filter(results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove empty, duplicate, or obviously irrelevant results."""
    seen = set()
    filtered = []
    for r in results:
        title = (r.get("title") or "").strip().lower()
        url = (r.get("url") or "").strip()

        if not r.get("raw_content"):
            continue
        if "buy" in title or "course" in title or "company" in title:
            continue
        if (title, url) in seen:
            continue

        seen.add((title, url))
        filtered.append(r)
    return filtered


class ResearchAgent:
    def __init__(self, model="gpt-oss:20b", use_llm_filter=False):
        self.model = model
        self.use_llm_filter = use_llm_filter
        self.llm = LLMClient(default_model=model)
        self.client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", "your_api_key_here"))

        self.SYSTEM_PROMPT = (
            f"You are a research assistant.\n"
            f"The current date is: {datetime.now().strftime('%d.%m.%Y')}.\n\n"
            f"When answering:\n"
            f"1. If you need external information, call the tool named `web_search` "
            f"with a single argument `query` describing what to search for. "
            f"You can do multiple rounds of searching, but keep in mind that you should also come to an answer at some point."
            f"Do not invent data or answer without the search if you’re unsure.\n\n"
            f"2. When search results are provided:\n"
            f"   - Summarize in 2–4 paragraphs.\n"
            f"   - Cite sources inline as (Title, URL).\n"
            f"   - Do not fabricate references.\n"
            f"   - Return natural language prose, not JSON.\n\n"
            f"Your goal: deliver a concise, referenced research summary."
        )

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": (
                        "Search the web for information relevant to the user's query. "
                        "Use this when the user asks for recent information, factual data, or news."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query or question to look up on the web."
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

        print(f"[Agent] Initialized with model='{model}'")

    def __str__(self):
        return "ResearchAgent"

    def chat(self, messages: List[Dict], reasoning="low", tools=None) -> dict:
        """Route chat messages to the LLM client."""
        print(f"[Agent] Sending {len(messages)} messages to model '{self.model}'")
        return self.llm.query(messages, model=self.model, reasoning=reasoning, tools=tools)

    def ask(self, question: str) -> str:
        print(f"[Agent] New question: {question.strip()}")
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        while True:
            response, messages = self.chat(messages, reasoning="high", tools=self.tools)
            msg = response["message"]

            if msg.thinking:
                print('[Agent] Thinking: ', response.message.thinking)
            if msg.content:
                print('[Agent] Content: ', response.message.content)

            if "tool_calls" in msg and msg["tool_calls"]:
                for tool_call in msg["tool_calls"]:
                    func = tool_call["function"]["name"]
                    args = tool_call["function"]["arguments"]
                    tool_id = tool_call["function"].get("id")

                    if func == "web_search":
                        query = args.get("query")
                        print(f"[Agent] → Detected web_search request: {query}")
                        results = json.dumps(self.web_search(query), indent=2)
                        print(f"[Agent] WebSearch results: {results}")

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": results,
                        })
                continue

            content = msg.get("content", "")
            if content.strip():
                return content
            return str(response)

    def web_search(self, query: str, max_results: int = 8) -> List[Dict[str, str]]:
        resp = self.client.search(query, max_results=max_results, include_raw_content=True)
        results = [
            {"title": r.get("title", ""), "url": r.get("url", ""), "raw_content": r.get("raw_content", "")}
            for r in resp.get("results", [])
        ]

        results = static_filter(results)
        if self.use_llm_filter:
            results = self.llm_relevance_check(query, results)

        return results

    def llm_relevance_check(self, question: str, results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        kept = []
        for r in results:
            prompt = (
                "You are evaluating a web search result and should decide if this result is relevant to answering the question\n\n"
                f"Question: {question}\n\n"
                "Web search result:\n"
                f"Title: {r.get('title')}\n"
                f"URL: {r.get('url')}\n"
                f"Snippet: {r.get('raw_content')[:300]}\n\n"
                "Is this result relevant to answering the question?\n"
                "Answer with YES or NO only."
            )

            resp, _ = self.chat([{"role": "system", "content": prompt}])
            decision = resp.get("message", {}).get("content", "").strip().upper()
            if decision.startswith("YES"):
                prompt = (f"You are processing a web search result. For the given user "
                          f"question: {question}\n"
                          "Write a short summary for the following search result in context of the given question. "
                          "Answer directly with the summary, do not leave out key points."
                          f"This is the search result: \n{r.get('raw_content')}")
                old_length = len(r.get('raw_content'))
                print(f"[WebSearch] Writing summary for result of length: {old_length}")
                resp, _ = self.chat([{"role": "system", "content": prompt}])
                summary = resp["message"]["content"]
                new_length = len(summary)
                print(f"[WebSearch] Summary length: {new_length} [Reduction: {old_length // new_length}x]")
                r["raw_content"] = summary
                kept.append(r)
        return kept
