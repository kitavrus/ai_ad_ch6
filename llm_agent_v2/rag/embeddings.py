"""Embedding generation using OpenAI-compatible embeddings API."""

import os
from typing import List


def _print_embed_bar(current: int, total: int, n_texts: int) -> None:
    width = 30
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    pct = int(100 * current / total)
    done = min(current * 50, n_texts)  # batch_size=50
    print(f"\rEmbedding: [{bar}] {pct:3d}% ({done}/{n_texts} chunks)", end="", flush=True)

import numpy as np
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(find_dotenv(usecwd=True))

_DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
_DEFAULT_EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))


class EmbeddingGenerator:
    """Generates embeddings via OpenAI API in batches."""

    EMBEDDING_DIM = _DEFAULT_EMBEDDING_DIM

    def __init__(self, model: str = _DEFAULT_EMBEDDING_MODEL, batch_size: int = 50):
        self.model = model
        self.batch_size = batch_size
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            kwargs: dict = {}
            base_url = os.getenv("OPENAI_BASE_URL")
            if base_url:
                kwargs["base_url"] = base_url
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=30.0, **kwargs)
        return self._client

    def generate(self, texts: List[str], progress: bool = False) -> np.ndarray:
        """Generate embeddings for a list of texts. Returns shape (N, 1536)."""
        if not texts:
            return np.zeros((0, self.EMBEDDING_DIM), dtype=np.float32)

        all_embeddings: List[List[float]] = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        try:
            for batch_num, i in enumerate(range(0, len(texts), self.batch_size), 1):
                batch = texts[i: i + self.batch_size]
                for attempt in range(3):
                    try:
                        response = self.client.embeddings.create(model=self.model, input=batch)
                        break
                    except Exception as e:
                        if attempt == 2:
                            raise
                        if progress:
                            print(f"\n  Ошибка батча {batch_num}, повтор {attempt + 1}/2: {e}", flush=True)
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                if progress:
                    _print_embed_bar(batch_num, total_batches, len(texts))
        except KeyboardInterrupt:
            if progress:
                print()
            raise
        if progress:
            print()

        return np.array(all_embeddings, dtype=np.float32)
