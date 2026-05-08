"""Minimal OpenAI-compatible chat client (works with OpenAI, Ollama /v1, LM Studio, etc.)."""

from __future__ import annotations

from dataclasses import dataclass

import requests  # type: ignore[import-untyped]


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class LLMClient:
    """Thin wrapper around OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 900,
        timeout: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def chat(self, messages: list[ChatMessage]) -> str:
        """Send messages to the configured model and return assistant text."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        response = requests.post(
            url, headers=headers, json=payload, timeout=self.timeout
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"LLM request failed [{response.status_code}]: {response.text[:500]}"
            )
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as err:
            raise RuntimeError(f"Unexpected LLM response shape: {data}") from err
