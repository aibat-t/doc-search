"""Второй этап поиска: cross-encoder пересортировывает кандидатов.

Эмбеддер (bi-encoder) кодирует вопрос и чанк ОТДЕЛЬНО, поэтому векторы чанков считаются заранее,
а поиск — это быстрое сравнение векторов. Cross-encoder читает пару «вопрос + чанк» ЦЕЛИКОМ:
внимание работает между словами вопроса и словами чанка. Это точнее, но заранее ничего не посчитать —
каждую пару нужно прогнать через модель на лету. Поэтому он работает только по top-N кандидатам.
"""

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str, max_length: int = 512, batch_size: int = 16) -> None:
        self.name = model_name
        self._batch_size = batch_size
        self._model = CrossEncoder(model_name, max_length=max_length)

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Чем больше число, тем лучше чанк отвечает на вопрос. Шкала своя у каждой модели."""
        if not passages:
            return []
        scores = self._model.predict(
            [(query, passage) for passage in passages],
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]
