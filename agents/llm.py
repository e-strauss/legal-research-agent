import json
import openai
from ollama import chat

from typing import List, Dict


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



def _query_ollama(messages: List[Dict], model: str, reasoning: str, tools: List[Dict] = None, thinking=False) -> Dict:
    """Call a local Ollama instance."""
    resp = chat(
        model=model,
        messages=messages,
        stream=False,
        tools=tools,
        think=thinking,
        options={"temperature": 0.4}
    )

    messages.append(resp.message)
    return resp, messages


class LLMClient:
    """Unified interface for interacting with different LLM providers."""

    def __init__(self, default_model: str = "gpt-oss:20b"):
        self.default_model = default_model

    def query(self, messages: List[Dict], model: str = None, thinking=False, reasoning: str = "low", tools: List[Dict] = None) -> (Dict, List):
        """Dispatch query to the correct LLM backend based on model name."""
        model = model or self.default_model

        if model.startswith("gpt-oss") or "ollama" in model.lower():
            return _query_ollama(messages, model, reasoning, tools, thinking=thinking)
        elif model.startswith("gpt-") or "openai" in model.lower():
            return _query_openai(messages, model, tools)
        else:
            raise ValueError(f"Unsupported model provider for '{model}'")
