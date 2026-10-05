import pathlib
import typing

import requests
from langchain_core.embeddings import Embeddings


class EmbeddingProvider(typing.Protocol):
    """Anything with this method can be passed to `create_router(embedding_provider=...)`.

    The returned vectors must come from the encoder CLM was trained on (Qwen3-8B,
    last-token pooling): the projection heads are not compatible with other models.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class LangChainEmbeddingProvider:
    """Adapts any LangChain `Embeddings` object (e.g. a custom or hosted one)."""

    def __init__(self, embeddings: Embeddings) -> None:
        self._embeddings = embeddings

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)


class LlamaEmbeddingProvider:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8090",
        model: str = "Qwen3-8B-Q4_K_M.gguf",
        model_path: str | None = None,
    ) -> None:
        # Optional sanity check that the GGUF served by llama is actually on disk
        # (only checks existence, the file is never read).
        if model_path is not None and not pathlib.Path(model_path).is_file():
            raise FileNotFoundError(f"Embedding model not found at '{model_path}'")
        self.base_url = base_url.rstrip("/")
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            f"{self.base_url}/v1/embeddings",
            json={
                "model": self.model,
                "input": texts,
            },
            timeout=120,
        )
        response.raise_for_status()

        data = response.json()
        return [
            item["embedding"]
            for item in sorted(data["data"], key=lambda x: x["index"])
        ]
