"""Thin wrapper around the Groq OpenAI-compatible chat completions API."""

import json
import re

import requests

from . import config


class LLMError(RuntimeError):
    """Raised when the LLM cannot be configured or reached."""


class GroqLLM:
    """Minimal Groq chat client (no SDK dependency)."""

    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.model = model or config.GROQ_MODEL
        if not self.api_key:
            raise LLMError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )

    def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        import time
        max_retries = 4
        backoff_seconds = 2.0
        
        for attempt in range(max_retries):
            response = requests.post(
                config.GROQ_API_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "temperature": temperature,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=config.REQUEST_TIMEOUT,
            )
            
            # If hit rate limit (429), wait and retry
            if response.status_code == 429 and attempt < max_retries - 1:
                # Get retry-after header if provided, else use backoff
                retry_after = response.headers.get("Retry-After")
                sleep_time = float(retry_after) if (retry_after and retry_after.isdigit()) else backoff_seconds
                time.sleep(sleep_time)
                backoff_seconds *= 2.0
                continue
                
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]


def extract_json(text: str) -> dict:
    """Extract the first JSON object found in an LLM response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(match.group(0))
