"""Authoritative organization LLM client integration for Fitzy Sales Agent.

The Agent depends on the small ``LLMClient`` protocol rather than a specific
model vendor. This keeps Fitzy deployable against Groq, OpenAI, or a local
OpenAI-compatible model server without changing the orchestration layer.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import BaseModel, SecretStr, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    """Minimal interface required by Fitzy's runtime."""

    @property
    def configured(self) -> bool:
        """Return whether an API key or provider setting is active."""

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_message: str,
        response_model: type[T],
    ) -> T:
        """Generate and validate a structured response."""

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_message: str,
    ) -> str:
        """Generate a natural-language response."""


class LLMConfigurationError(RuntimeError):
    """Raised when the configured LLM provider is missing required settings."""


class LLMResponseError(RuntimeError):
    """Raised when an LLM response cannot be normalized or validated."""


class OpenAICompatibleLLMClient:
    """Authoritative OpenAI-compatible LLM client used by Fitzy runtime.

    Supports Groq, OpenAI, and local OpenAI-compatible endpoints. Uses shared
    AsyncClient connections when provided and handles structured outputs reliably.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | SecretStr | None = None,
        model: str | None = None,
        timeout_seconds: float = 45.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = (
            base_url
            or os.getenv("CLOTHING_AGENT_LLM_API_BASE")
            or "https://api.groq.com/openai/v1"
        ).rstrip("/")

        resolved_key = (
            api_key.get_secret_value()
            if isinstance(api_key, SecretStr)
            else api_key
        )
        if not resolved_key:
            resolved_key = (
                os.getenv("CLOTHING_AGENT_LLM_API_KEY")
                or os.getenv("GROQ_API_KEY")
                or os.getenv("OPENAI_API_KEY")
            )
        self._api_key = resolved_key

        self._model = (
            model
            or os.getenv("CLOTHING_AGENT_LLM_MODEL")
            or "openai/gpt-oss-120b"
        )
        self._timeout = timeout_seconds
        self._http_client = http_client

    @property
    def configured(self) -> bool:
        """Return True if an API key is available."""
        return bool(self._api_key and self._api_key.strip())

    def _validate_configuration(self) -> None:
        if not self.configured:
            raise LLMConfigurationError("LLM API key is not configured (CLOTHING_AGENT_LLM_API_KEY or GROQ_API_KEY required)")
        if not self._model:
            raise LLMConfigurationError("LLM model is not configured")

    async def _chat(self, *, system_prompt: str, user_message: str, json_mode: bool) -> str:
        self._validate_configuration()
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.1,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self._base_url}/chat/completions"

        if self._http_client is not None:
            response = await self._http_client.post(url, headers=headers, json=payload, timeout=self._timeout)
            return self._extract_content(response)
        else:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
                return self._extract_content(response)

    def _extract_content(self, response: httpx.Response) -> str:
        if response.is_error:
            raise LLMResponseError(f"LLM HTTP {response.status_code}: {response.text[:500]}")
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
            return str(content) if content is not None else ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError("LLM response did not contain valid completion content") from exc

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_message: str,
        response_model: type[T],
    ) -> T:
        """Generate JSON, extract the first JSON object if needed, and validate it via Pydantic."""
        raw = await self._chat(system_prompt=system_prompt, user_message=user_message, json_mode=True)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                raise LLMResponseError(f"LLM did not return a JSON object: {raw[:200]}") from None
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise LLMResponseError(f"LLM returned malformed JSON: {raw[:200]}") from exc

        try:
            return response_model.model_validate(payload)
        except ValidationError as exc:
            raise LLMResponseError(f"LLM structured output validation failed: {exc}") from exc

    async def generate_text(self, *, system_prompt: str, user_message: str) -> str:
        """Generate plain natural-language text."""
        return await self._chat(system_prompt=system_prompt, user_message=user_message, json_mode=False)


class FakeLLMClient:
    """Deterministic test double used by Fitzy test suites."""

    def __init__(self, structured_response: BaseModel, text_response: str) -> None:
        self.structured_response = structured_response
        self.text_response = text_response
        self.structured_calls: list[tuple[str, str]] = []
        self.text_calls: list[tuple[str, str]] = []

    @property
    def configured(self) -> bool:
        return True

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_message: str,
        response_model: type[T],
    ) -> T:
        self.structured_calls.append((system_prompt, user_message))
        return response_model.model_validate(self.structured_response.model_dump())

    async def generate_text(self, *, system_prompt: str, user_message: str) -> str:
        self.text_calls.append((system_prompt, user_message))
        return self.text_response
