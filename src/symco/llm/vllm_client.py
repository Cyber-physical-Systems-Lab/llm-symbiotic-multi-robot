"""Minimal OpenAI-compatible chat client for a local vLLM server."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any
from urllib import error, request


@dataclass(slots=True)
class VLLMConfig:
    """Configuration for a local OpenAI-compatible vLLM endpoint."""

    base_url: str
    model: str
    api_key: str = "EMPTY"
    timeout_sec: int = 120
    temperature: float = 0.1
    max_tokens: int = 512


def load_vllm_config_from_env() -> VLLMConfig:
    """Load vLLM configuration from environment variables."""
    base_url = os.environ.get("SYMCO_VLLM_BASE_URL", "").strip()
    model = os.environ.get("SYMCO_VLLM_MODEL", "").strip()
    if not base_url:
        raise ValueError("Missing required environment variable: SYMCO_VLLM_BASE_URL")
    if not model:
        raise ValueError("Missing required environment variable: SYMCO_VLLM_MODEL")

    api_key = os.environ.get("SYMCO_VLLM_API_KEY", "EMPTY")
    timeout_sec = _parse_int_env("SYMCO_VLLM_TIMEOUT_SEC", default=120)
    temperature = _parse_float_env("SYMCO_VLLM_TEMPERATURE", default=0.1)
    max_tokens = _parse_int_env("SYMCO_VLLM_MAX_TOKENS", default=512)

    return VLLMConfig(
        base_url=base_url.rstrip("/"),
        model=model,
        api_key=api_key,
        timeout_sec=timeout_sec,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def extract_first_json_object(text: str) -> dict:
    """Extract and parse the first JSON object found in a text blob."""
    stripped = text.strip()
    unfenced = _strip_code_fences(stripped)

    for candidate_text in (unfenced, stripped):
        if not candidate_text:
            continue
        try:
            parsed = json.loads(candidate_text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    text = unfenced
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output: missing '{'")

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Found JSON-like object but failed to parse it: {exc}") from exc
                if not isinstance(parsed, dict):
                    raise ValueError("Parsed JSON value is not an object")
                return parsed

    raise ValueError("No complete JSON object found in model output: unmatched braces")


class VLLMChatClient:
    """Small chat-completions client for an OpenAI-compatible vLLM server."""

    def __init__(self, config: VLLMConfig):
        self.config = config

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Send a chat completion request and parse the first JSON object returned."""
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        try:
            content = self._chat_content(payload)
        except RuntimeError as exc:
            payload.pop("response_format", None)
            content = self._chat_content(payload, allow_json_mode_retry=False, original_error=exc)

        if not isinstance(content, str):
            raise ValueError("Invalid vLLM response format: assistant content is not a string")

        print("RAW_ASSISTANT_CONTENT:")
        print(content)

        try:
            return extract_first_json_object(content)
        except ValueError as exc:
            repair_content = self._chat_content(
                {
                    "model": self.config.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Convert the following content into one valid JSON object only. "
                                "Do not add explanation."
                            ),
                        },
                        {"role": "user", "content": content},
                    ],
                    "temperature": 0.0,
                    "max_tokens": self.config.max_tokens,
                    "response_format": {"type": "json_object"},
                }
            )
            print("RAW_ASSISTANT_CONTENT:")
            print(repair_content)
            try:
                return extract_first_json_object(repair_content)
            except ValueError as repair_exc:
                raise ValueError(
                    "Assistant content does not contain a valid JSON object after repair retry: "
                    f"{repair_exc}"
                ) from exc

    def _chat_content(
        self,
        payload: dict[str, Any],
        allow_json_mode_retry: bool = True,
        original_error: Exception | None = None,
    ) -> str:
        """Send a chat completion request and return assistant text content."""
        url = f"{self.config.base_url}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        req = request.Request(url=url, data=body, headers=headers, method="POST")

        try:
            with request.urlopen(req, timeout=self.config.timeout_sec) as response:
                raw_bytes = response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if allow_json_mode_retry and "response_format" in payload:
                raise RuntimeError(
                    f"vLLM HTTP request failed with status {exc.code}: {detail}"
                ) from exc
            if original_error is not None:
                raise RuntimeError(
                    f"vLLM HTTP request failed after retry without response_format: {exc.code}: {detail}; "
                    f"original error: {original_error}"
                ) from exc
            raise RuntimeError(
                f"vLLM HTTP request failed with status {exc.code}: {detail}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"vLLM HTTP request failed: {exc.reason}") from exc
        except Exception as exc:
            raise RuntimeError(f"Unexpected error during vLLM request: {exc}") from exc

        try:
            response_json = json.loads(raw_bytes.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"vLLM response is not valid JSON: {exc}") from exc

        try:
            content = response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "Invalid vLLM response format: expected choices[0].message.content"
            ) from exc
        if not isinstance(content, str):
            raise ValueError("Invalid vLLM response format: assistant content is not a string")
        return content


def _parse_int_env(name: str, default: int) -> int:
    """Parse an integer environment variable with a default."""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an int, got {value!r}") from exc


def _parse_float_env(name: str, default: float) -> float:
    """Parse a float environment variable with a default."""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got {value!r}") from exc


def _strip_code_fences(text: str) -> str:
    """Strip one outer triple-backtick fence block when present."""
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    if not lines[-1].strip().startswith("```"):
        return text
    return "\n".join(lines[1:-1]).strip()


if __name__ == "__main__":
    client = VLLMChatClient(load_vllm_config_from_env())
    result = client.chat_json(
        system_prompt='Return JSON only.',
        user_prompt='Reply with {"ok": true}',
    )
    print(result)
