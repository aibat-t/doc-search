"""Этап 3 пайплайна: текст → вектор. Тот же эмбеддер, что в faq-search, плюс счётчик токенов."""

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, query_prefix: str, passage_prefix: str, batch_size: int = 32) -> None:
        self.name = model_name
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name)

    def count_tokens(self, text: str) -> int:
        """Настоящее число токенов по токенизатору модели — для нарезки чанков."""
        return len(self._model.tokenizer(text, add_special_tokens=False)["input_ids"])

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self._query_prefix)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self._passage_prefix)

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        embeddings = self._model.encode(
            [prefix + text for text in texts],
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32, copy=False)
