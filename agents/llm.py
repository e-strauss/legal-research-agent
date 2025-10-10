
import json
import openai
import requests
import os

from typing import List, Dict

ollama_url = os.getenv("OLLAMA_URL", "http://10.0.139.104:11434/api/chat")


def _query_openai(messages: List[Dict], model: str, tools: List[Dict] = None) -> (Dict, List):
    """Call the OpenAI API, handling both text and tool calls."""
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        tools=tools,
    )

    message = response.choices[0].message
    messages.append(message)
    # If the model called a tool (function)
    if hasattr(message, "tool_calls") and message.tool_calls:
        return {
            "message": {
                "role": message.role,
                "tool_calls": [
                    {
                        "function": {
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": json.loads(tc.function.arguments),
                        }
                    }
                    for tc in message.tool_calls
                ],
            }
        }, messages

    return {
        "message": {
            "role": message.role,
            "content": message.content,
        }
    }, messages



def _query_ollama(messages: List[Dict], model: str, reasoning: str, tools: List[Dict] = None) -> Dict:
    """Call a local Ollama instance."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "stream": False,
        "reasoning_effort": reasoning,
    }

    resp = requests.post(
        ollama_url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    messages.append(resp.json()["message"])
    return resp.json(), messages


class LLMClient:
    """Unified interface for interacting with different LLM providers."""

    def __init__(self, default_model: str = "gpt-oss:20b"):
        self.default_model = default_model
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def query(self, messages: List[Dict], model: str = None, reasoning: str = "low", tools: List[Dict] = None) -> (Dict, List):
        """Dispatch query to the correct LLM backend based on model name."""
        model = model or self.default_model

        if model.startswith("gpt-oss") or "ollama" in model.lower():
            return _query_ollama(messages, model, reasoning, tools)
        elif model.startswith("gpt-") or "openai" in model.lower():
            return _query_openai(messages, model, tools)
        else:
            raise ValueError(f"Unsupported model provider for '{model}'")
