"""Ollama embedding provider implementation."""

import json
from typing import Any

import requests

from ..base import BaseEmbeddingProvider, EmbeddingResult
from ..config import EmbeddingConfig
from ..provider import register_embedding_provider

# Known embedding models and their dimensions
OLLAMA_EMBEDDING_MODELS: dict[str, int] = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
    "bge-m3": 1024,
    "bge-large": 1024,
}


@register_embedding_provider("ollama")
class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """
    Ollama embedding provider supporting both old and new API versions.

    - New API (0.1.26+): /api/embed with "input" parameter
    - Old API: /api/embeddings with "prompt" parameter

    Supported models: nomic-embed-text, mxbai-embed-large, all-minilm, etc.
    See: https://ollama.ai/library for available embedding models.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """Initialize Ollama embedding provider."""
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"
        self._session = requests.Session()
        # Get dimensions from known models or use config override
        model_base = config.model.split(":")[0]
        self._dimensions: int = config.dimensions or OLLAMA_EMBEDDING_MODELS.get(model_base, 768)
        # Will be set after first successful request
        self._use_legacy_api: bool | None = None

    @property
    def dimensions(self) -> int:
        """Return embedding dimensions for the current model."""
        return self._dimensions

    def _try_embed_request(
        self, endpoint: str, payload: dict[str, Any], timeout: int = 60
    ) -> requests.Response | None:
        """Try an embed request to a specific endpoint."""
        try:
            response = self._session.post(
                f"{self.base_url}{endpoint}",
                json=payload,
                timeout=timeout,
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return None
            raise

    def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text."""
        # Try new API first if we haven't determined which to use
        if self._use_legacy_api is None or self._use_legacy_api is False:
            # New API: /api/embed with "input"
            payload = {"model": self.model, "input": text}
            try:
                response = self._try_embed_request("/api/embed", payload)
                if response is not None:
                    self._use_legacy_api = False
                    return self._parse_embed_response(response.json())
            except requests.exceptions.ConnectionError as e:
                raise ConnectionError(
                    f"Cannot connect to Ollama at {self.base_url}. Make sure Ollama is running."
                ) from e

        # Try legacy API: /api/embeddings with "prompt"
        if self._use_legacy_api is None or self._use_legacy_api is True:
            payload = {"model": self.model, "prompt": text}
            try:
                response = self._try_embed_request("/api/embeddings", payload)
                if response is not None:
                    self._use_legacy_api = True
                    return self._parse_embeddings_response(response.json())
            except requests.exceptions.ConnectionError as e:
                raise ConnectionError(
                    f"Cannot connect to Ollama at {self.base_url}. Make sure Ollama is running."
                ) from e

        raise RuntimeError(
            f"Ollama embedding API not available at {self.base_url}. "
            "Make sure you have an embedding model installed (e.g., 'ollama pull nomic-embed-text')"
        )

    def _parse_embed_response(self, result: dict[str, Any]) -> EmbeddingResult:
        """Parse response from new /api/embed endpoint."""
        # New API returns {"embeddings": [[...]]} for single input
        embeddings_list = result.get("embeddings", [[]])
        embedding = embeddings_list[0] if embeddings_list else []
        embedding = self._normalize_embedding(embedding)

        if len(embedding) > 0 and len(embedding) != self._dimensions:
            self._dimensions = len(embedding)

        return {
            "embedding": embedding,
            "model": self.model,
            "dimensions": len(embedding),
            "tokens_used": result.get("prompt_eval_count"),
        }

    def _parse_embeddings_response(self, result: dict[str, Any]) -> EmbeddingResult:
        """Parse response from legacy /api/embeddings endpoint."""
        # Legacy API returns {"embedding": [...]} directly
        embedding = result.get("embedding", [])
        embedding = self._normalize_embedding(embedding)

        if len(embedding) > 0 and len(embedding) != self._dimensions:
            self._dimensions = len(embedding)

        return {
            "embedding": embedding,
            "model": self.model,
            "dimensions": len(embedding),
            "tokens_used": result.get("prompt_eval_count"),
        }

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts (batched)."""
        if not texts:
            return []

        # New API supports batch, legacy doesn't - fall back to sequential
        if self._use_legacy_api:
            return [self.embed_text(text) for text in texts]

        # Try new API with batch
        payload = {"model": self.model, "input": texts}
        try:
            response = self._try_embed_request("/api/embed", payload, timeout=120)
            if response is not None:
                self._use_legacy_api = False
                result = response.json()
                embeddings_list = result.get("embeddings", [])
                results: list[EmbeddingResult] = []
                for emb in embeddings_list:
                    normalized = self._normalize_embedding(emb)
                    results.append(
                        {
                            "embedding": normalized,
                            "model": self.model,
                            "dimensions": len(normalized),
                            "tokens_used": None,
                        }
                    )
                return results
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}.") from e

        # Fall back to legacy sequential
        self._use_legacy_api = True
        return [self.embed_text(text) for text in texts]

    def health_check(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            # First check if Ollama is reachable
            response = self._session.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            response.raise_for_status()

            # Check if the embedding model is available
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            model_base = self.model.split(":")[0]

            if model_base not in model_names:
                # Model not pulled yet, but Ollama is running
                return True  # Will fail gracefully on first embed

            # Try a test embedding
            return super().health_check()

        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List available embedding models from Ollama."""
        try:
            response = self._session.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            response.raise_for_status()

            models = response.json().get("models", [])
            # Filter to known embedding models
            embedding_models = []
            for model in models:
                name = model.get("name", "").split(":")[0]
                if name in OLLAMA_EMBEDDING_MODELS:
                    embedding_models.append(model.get("name", ""))

            return embedding_models

        except (requests.RequestException, json.JSONDecodeError, KeyError, TypeError):
            return []
