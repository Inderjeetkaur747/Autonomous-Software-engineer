"""A thin, safe wrapper around the Groq SDK.

Features: retries with backoff, JSON mode, Pydantic validation,
self-correction loop, latency and token tracking.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import groq
from groq import Groq
from pydantic import BaseModel, ValidationError

from swe_agent.config import Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """The LLM call failed (network, rate limit, bad key...)."""


class StructuredOutputError(LLMError):
    """The model never produced output that matches the schema."""


@dataclass
class LLMCallRecord:
    """One API call. Phase 16 will export these as metrics."""

    purpose: str
    model: str
    attempt: int
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    ok: bool


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` if the model added it anyway."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        cleaned = cleaned.rstrip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    return cleaned.strip()


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Exponential backoff: 2s, 4s, 8s... Respect the 'retry-after' header if present."""
    delay = float(min(2**attempt, 20))
    response = getattr(exc, "response", None)
    if response is not None:
        header = response.headers.get("retry-after")
        if header:
            try:
                delay = min(float(header), 60.0)
            except ValueError:
                pass
    return delay


class GroqLLM:
    def __init__(
        self,
        settings: Settings | None = None,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings or get_settings()
        self._sleep = sleep  # injected so tests do not really wait
        self.calls: list[LLMCallRecord] = []

        if client is not None:
            self._client = client  # used by tests (a fake client)
        else:
            key = self._settings.groq_api_key.get_secret_value()
            if not key:
                raise LLMError("GROQ_API_KEY is empty. Add it to your .env file.")
            # max_retries=0: we do our own retries so we can log and count them.
            self._client = Groq(
                api_key=key,
                timeout=self._settings.llm_timeout_seconds,
                max_retries=0,
            )

    # ------------------------------------------------------------------ stats
    @property
    def total_tokens(self) -> int:
        return sum(c.prompt_tokens + c.completion_tokens for c in self.calls)

    def _record(
        self, purpose: str, attempt: int, started: float, response: Any | None
    ) -> None:
        usage = getattr(response, "usage", None)
        record = LLMCallRecord(
            purpose=purpose,
            model=self._settings.groq_model,
            attempt=attempt,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ok=response is not None,
        )
        self.calls.append(record)
        logger.info(
            "llm_call purpose=%s attempt=%d ok=%s latency_ms=%s tokens=%d+%d",
            record.purpose,
            record.attempt,
            record.ok,
            record.latency_ms,
            record.prompt_tokens,
            record.completion_tokens,
        )

    # ------------------------------------------------------------- low level
    def _request(self, kwargs: dict[str, Any], *, purpose: str) -> Any:
        """Call the API with retries and backoff. Returns the raw response."""
        max_attempts = self._settings.llm_max_retries
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            started = time.perf_counter()
            try:
                response = self._client.chat.completions.create(**kwargs)
            except groq.AuthenticationError as exc:
                raise LLMError("Groq rejected the API key (401). Check GROQ_API_KEY.") from exc
            except (groq.APIConnectionError, groq.APIStatusError) as exc:
                status = getattr(exc, "status_code", None)  # None = network problem
                retryable = status is None or status == 429 or status >= 500
                self._record(purpose, attempt, started, None)
                if not retryable:
                    raise LLMError(f"Groq API error {status}: {exc}") from exc
                last_error = exc
                if attempt < max_attempts:
                    delay = _retry_delay(exc, attempt)
                    logger.warning(
                        "Groq call failed (status=%s). Retry %d/%d in %.1fs",
                        status,
                        attempt,
                        max_attempts,
                        delay,
                    )
                    self._sleep(delay)
                continue

            self._record(purpose, attempt, started, response)
            return response

        raise LLMError(f"Groq call failed after {max_attempts} attempts: {last_error}")

    def _chat(self, messages: list[dict[str, str]], *, purpose: str, json_mode: bool) -> str:
        kwargs: dict[str, Any] = {
            "model": self._settings.groq_model,
            "messages": messages,
            "temperature": self._settings.llm_temperature,
            "max_tokens": self._settings.llm_max_output_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._request(kwargs, purpose=purpose)
        return response.choices[0].message.content or ""

    def chat_tools(
        self, messages: list[dict[str, Any]],
        tools: list[dict[str, Any]], *, purpose: str = "explore"
    ) -> ChatTurn:
        """One step of a tool-calling conversation. The model may ask for tools."""
        kwargs: dict[str, Any] = {
            "model": self._settings.groq_model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": self._settings.llm_temperature,
            "max_tokens": self._settings.llm_max_output_tokens,
        }
        response = self._request(kwargs, purpose=purpose)
        message = response.choices[0].message
        requests = [
            ToolCallRequest(
                id=str(call.id),
                name=call.function.name,
                arguments=call.function.arguments or "{}",
            )
            for call in (getattr(message, "tool_calls", None) or [])
        ]
        return ChatTurn(content=message.content or "", tool_calls=requests)

    # ------------------------------------------------------------ public API
    def generate_text(self, *, system: str, user: str, purpose: str = "text") -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self._chat(messages, purpose=purpose, json_mode=False)

    def generate_structured(
        self,
        schema: type[T],
        *,
        system: str,
        user: str,
        purpose: str = "structured",
        max_validation_retries: int = 2,
    ) -> T:
        """Ask for JSON, validate with Pydantic, self-correct if invalid."""
        schema_text = json.dumps(schema.model_json_schema(), indent=2)
        full_system = (
            f"{system}\n\n"
            "Respond with ONLY a valid JSON object that matches this JSON Schema. "
            "No markdown, no explanation, no extra keys.\n\n"
            f"JSON Schema:\n{schema_text}"
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user},
        ]

        last_error = ""
        for attempt in range(1, max_validation_retries + 2):
            raw = self._chat(messages, purpose=purpose, json_mode=True)
            try:
                return schema.model_validate_json(_strip_code_fences(raw))
            except ValidationError as exc:
                last_error = str(exc)[:1500]
                logger.warning(
                    "Output failed validation (attempt %d): %s", attempt, last_error[:200]
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your JSON was invalid for the schema. Errors:\n"
                            f"{last_error}\n\nReturn the corrected JSON only."
                        ),
                    }
                )

        raise StructuredOutputError(
            f"Model did not return valid {schema.__name__} after "
            f"{max_validation_retries + 1} attempts. Last error: {last_error}"
        )

@dataclass
class ToolCallRequest:
    """One tool the model wants to use. `arguments` is the raw JSON text."""

    id: str
    name: str
    arguments: str


@dataclass
class ChatTurn:
    """The model's answer in a tool-calling chat: some text and/or tool requests."""

    content: str
    tool_calls: list[ToolCallRequest]

    def to_message(self) -> dict[str, Any]:
        """The assistant message to append to the conversation."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                }
                for c in self.tool_calls
            ]
        return message