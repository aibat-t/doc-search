"""Слияние нескольких ранжированных списков в один: Reciprocal Rank Fusion.

Скоры разных поисков несравнимы: косинусная близость лежит в 0..1, а ts_rank_cd — в своей шкале,
которая зависит от длины текста и числа совпадений. Складывать их — всё равно что складывать
метры с килограммами. RRF смотрит не на скоры, а на места в списках:

    score(чанк) = Σ по спискам 1 / (k + место)

Кто высоко в обоих списках — побеждает; кто нашёлся только одним поиском — тоже не теряется.
"""

from collections import defaultdict
from collections.abc import Hashable, Sequence


def rrf[T: Hashable](*ranked_lists: Sequence[T], k: int = 60) -> list[tuple[T, float]]:
    """Каждый список — id, лучшие первыми. Возвращает (id, score), лучшие первыми."""
    scores: defaultdict[T, float] = defaultdict(float)
    for ranked in ranked_lists:
        for place, item in enumerate(ranked, start=1):
            scores[item] += 1 / (k + place)
    # sorted стабилен: при равном счёте раньше идёт тот, кто раньше встретился в первом списке.
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
