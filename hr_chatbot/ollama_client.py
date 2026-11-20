"""Small HTTP client for embeddings and chat: OpenAI-compatible (vLLM) or Ollama."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class OllamaClientError(Exception):
    """Raised when the Ollama server cannot satisfy a request."""

    def __init__(self, code: str, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class OllamaClient:
    """Chat via ``/v1/chat/completions``; embeddings via ``/v1/embeddings``.

    `embedding_base_url` bo'sh bo'lsa embedding Ollama'ning native API'siga
    (`host`) ketadi. vLLM va Ollama'dagi bge-m3 bir xil vektor beradi, shuning
    uchun avval Ollama'da qurilgan vektor bazasi o'zgarishsiz ishlaydi.
    """

    host: str
    timeout_seconds: float
    chat_base_url: str = ""
    api_key: str = ""
    embedding_base_url: str = ""

    def embed(self, *, model: str, texts: list[str]) -> list[list[float]]:
        """Create embeddings for a batch of texts."""

        if not texts:
            return []
        if self.embedding_base_url:
            return self._embed_openai(model=model, texts=texts)
        try:
            payload = {"model": model, "input": texts}
            parsed = self._post_json(self._ollama_url("/api/embed"), payload)
            embeddings = parsed.get("embeddings")
            if isinstance(embeddings, list) and len(embeddings) == len(texts):
                return [_coerce_embedding(item) for item in embeddings]
        except OllamaClientError as exc:
            if exc.code not in {"OLLAMA_HTTP_404", "OLLAMA_CONNECTION_ERROR"}:
                raise

        embeddings = []
        for text in texts:
            legacy = self._post_json(self._ollama_url("/api/embeddings"), {"model": model, "prompt": text})
            embedding = legacy.get("embedding")
            if not isinstance(embedding, list):
                raise OllamaClientError("OLLAMA_INVALID_EMBEDDING_RESPONSE", "Ollama embedding response is invalid.")
            embeddings.append(_coerce_embedding(embedding))
        return embeddings

    def _embed_openai(self, *, model: str, texts: list[str]) -> list[list[float]]:
        parsed = self._post_json(
            _v1_url(self.embedding_base_url, "/embeddings"),
            {"model": model, "input": texts},
            self._auth_headers(),
        )
        data = parsed.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise OllamaClientError("OLLAMA_INVALID_EMBEDDING_RESPONSE", "Embedding response is invalid.")
        items = sorted(data, key=lambda item: item.get("index", 0) if isinstance(item, dict) else 0)
        return [_coerce_embedding(item.get("embedding") if isinstance(item, dict) else None) for item in items]

    def chat(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        num_ctx: int,
        num_predict: int,
        temperature: float,
    ) -> str:
        """Generate a grounded answer with an OpenAI-compatible chat endpoint.

        `num_ctx` is kept for API compatibility; vLLM fixes the context size server-side.
        """

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "max_tokens": num_predict,
            "temperature": temperature,
            "top_p": 0.9,
        }
        chat_base = self.chat_base_url or f"{self.host.rstrip('/')}/v1"
        parsed = self._post_json(_v1_url(chat_base, "/chat/completions"), payload, self._auth_headers())
        choices = parsed.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                answer = message["content"].strip()
                if answer:
                    return answer
        raise OllamaClientError("OLLAMA_INVALID_CHAT_RESPONSE", "Chat response is missing assistant content.")

    def _ollama_url(self, path: str) -> str:
        return f"{self.host.rstrip('/')}{path}"

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _post_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", **(extra_headers or {})}
        if "ngrok" in endpoint.lower():
            headers["ngrok-skip-browser-warning"] = "true"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if int(exc.code) == 404:
                raise OllamaClientError(
                    "OLLAMA_HTTP_404",
                    f"Ollama endpoint not found: {endpoint}",
                    status_code=502,
                ) from exc
            raise OllamaClientError(
                "OLLAMA_HTTP_ERROR",
                f"Ollama returned HTTP {exc.code}: {body[:500]}",
                status_code=502,
            ) from exc
        except Exception as exc:
            raise OllamaClientError(
                "OLLAMA_CONNECTION_ERROR",
                f"Ollama request failed: {type(exc).__name__}: {exc}",
                status_code=502,
            ) from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise OllamaClientError("OLLAMA_INVALID_JSON", "Ollama returned non-JSON response.", status_code=502) from exc
        if not isinstance(parsed, dict):
            raise OllamaClientError("OLLAMA_INVALID_JSON", "Ollama response is not a JSON object.", status_code=502)
        return parsed


def _v1_url(base_url: str, endpoint: str) -> str:
    """`https://host`, `https://host/v1` yoki to'liq URL dan `.../v1{endpoint}` yasaydi."""

    base = base_url.strip().rstrip("/")
    if base.endswith(endpoint):
        return base
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return f"{base}{endpoint}"


def _coerce_embedding(value: Any) -> list[float]:
    if not isinstance(value, list):
        raise OllamaClientError("OLLAMA_INVALID_EMBEDDING_RESPONSE", "Embedding item is not a list.")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise OllamaClientError("OLLAMA_INVALID_EMBEDDING_RESPONSE", "Embedding contains non-numeric values.") from exc
