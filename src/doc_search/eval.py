"""Урок 2.2.4: сравнение режимов поиска на наборе вопросов с известным правильным разделом.

    uv run doc-eval                               # все 4 режима, eval/questions.jsonl
    uv run doc-eval --modes vector hybrid         # без reranker'а — быстрее
    uv run doc-eval --verbose                     # плюс: какие вопросы провалил каждый режим

Нужны проиндексированные документы из samples/ (card-rules.md, tariffs.html, zakon.pdf).
Правильный ответ задаём не id чанка (он меняется при переиндексации), а парой «файл + раздел»:
чанк релевантен, если он из нужного файла и его раздел содержит нужную строку.
"""

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_QUESTIONS = Path("eval/questions.jsonl")
DEFAULT_REPORT = Path("eval/last_run.md")
ALL_MODES = ["vector", "keyword", "hybrid", "hybrid_rerank"]


@dataclass(frozen=True)
class Question:
    id: str
    kind: str          # semantic — перефразировка; exact — точные термины, номера, коды
    question: str
    source: str
    section: str       # подстрока раздела, например «Статья 36-1.» (с точкой: иначе подойдёт и 36-10)


@dataclass(frozen=True)
class Outcome:
    question: Question
    rank: int | None   # место первого релевантного чанка в выдаче (1..k) или None
    latency_ms: float


def load_questions(path: Path) -> list[Question]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Question(**json.loads(line)) for line in lines if line.strip()]


def is_relevant(hit: dict[str, Any], q: Question) -> bool:
    return hit["source"] == q.source and q.section in (hit.get("section") or "")


def first_relevant_rank(hits: list[dict[str, Any]], q: Question) -> int | None:
    return next((place for place, hit in enumerate(hits, start=1) if is_relevant(hit, q)), None)


def summarize(outcomes: list[Outcome]) -> dict[str, float]:
    """recall@k — доля вопросов, где нужный чанк попал в top-k.
    MRR — среднее 1/место: 1-е место даёт 1, 2-е — 0.5, не нашли — 0. Учитывает, НАСКОЛЬКО высоко."""
    latencies = np.array([o.latency_ms for o in outcomes])
    return {
        "recall": float(np.mean([o.rank is not None for o in outcomes])),
        "mrr": float(np.mean([1 / o.rank if o.rank else 0.0 for o in outcomes])),
        "mean_ms": float(latencies.mean()),
        "p95_ms": float(np.percentile(latencies, 95)),
    }


def render_report(results: dict[str, list[Outcome]], k: int) -> str:
    kinds = sorted({o.question.kind for outcomes in results.values() for o in outcomes})
    head = f"| Режим | recall@{k} | MRR | " + " | ".join(f"recall@{k} ({kind})" for kind in kinds)
    head += " | среднее, мс | p95, мс |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for mode, outcomes in results.items():
        s = summarize(outcomes)
        by_kind = [summarize([o for o in outcomes if o.question.kind == kind])["recall"] for kind in kinds]
        lines.append(
            f"| {mode} | {s['recall']:.0%} | {s['mrr']:.2f} | "
            + " | ".join(f"{r:.0%}" for r in by_kind)
            + f" | {s['mean_ms']:.0f} | {s['p95_ms']:.0f} |"
        )
    return "\n".join(lines)


def render_misses(results: dict[str, list[Outcome]]) -> str:
    """Таблица по вопросам: место правильного чанка в каждом режиме. «—» — не нашёлся в top-k."""
    modes = list(results)
    lines = ["| Вопрос | " + " | ".join(modes) + " |", "|" + "---|" * (len(modes) + 1)]
    by_question: dict[str, dict[str, int | None]] = defaultdict(dict)
    texts: dict[str, str] = {}
    for mode, outcomes in results.items():
        for o in outcomes:
            by_question[o.question.id][mode] = o.rank
            texts[o.question.id] = o.question.question
    for qid, ranks in by_question.items():
        cells = [str(ranks[m]) if ranks.get(m) else "—" for m in modes]
        lines.append(f"| {texts[qid]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение режимов поиска: recall@k, MRR, latency")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--modes", nargs="+", choices=ALL_MODES, default=ALL_MODES)
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--verbose", action="store_true", help="показать место ответа по каждому вопросу")
    args = parser.parse_args()

    # Тяжёлые импорты (torch, модели) — только здесь, чтобы тесты метрик работали без них.
    from doc_search.config import Settings
    from doc_search.db import init_schema, make_pool
    from doc_search.embedder import Embedder
    from doc_search.reranker import Reranker
    from doc_search.search import Searcher, SearchMode

    settings = Settings()
    questions = load_questions(args.questions)
    init_schema(settings)       # добавит колонку tsv и GIN-индекс в уже существующую базу
    pool = make_pool(settings)

    with pool.connection() as conn:
        ready = {r["source"] for r in conn.execute("SELECT source FROM documents WHERE status = 'ready'")}
    missing = sorted({q.source for q in questions} - ready)
    if missing:
        raise SystemExit(f"Не проиндексированы: {', '.join(missing)}. Загрузите их из samples/ и запустите воркер.")

    print("Загружаю модели...")
    embedder = Embedder(settings.embedding_model, settings.query_prefix, settings.passage_prefix)
    reranker = None
    if "hybrid_rerank" in args.modes:
        if not settings.rerank_model:
            raise SystemExit("hybrid_rerank требует DOC_RERANK_MODEL")
        reranker = Reranker(settings.rerank_model, settings.rerank_max_length)
    searcher = Searcher(pool, embedder, settings, reranker)

    results: dict[str, list[Outcome]] = {}
    for mode_name in args.modes:
        mode = SearchMode(mode_name)
        searcher.search(questions[0].question, mode, args.k)       # прогрев: модели и кэш БД
        outcomes = []
        for q in questions:
            start = time.perf_counter()
            hits = searcher.search(q.question, mode, args.k)
            latency = (time.perf_counter() - start) * 1000
            outcomes.append(Outcome(q, first_relevant_rank(hits, q), latency))
        results[mode_name] = outcomes
        s = summarize(outcomes)
        print(f"  {mode_name:<14} recall@{args.k} {s['recall']:.0%}  MRR {s['mrr']:.2f}  p95 {s['p95_ms']:.0f} мс")
    pool.close()

    report = render_report(results, args.k)
    details = render_misses(results)
    print("\n" + report)
    if args.verbose:
        print("\n" + details)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        f"# doc-eval: {len(questions)} вопросов, k={args.k}\n\n{report}\n\n## По вопросам\n\n{details}\n",
        encoding="utf-8",
    )
    print(f"\nОтчёт сохранён в {args.report}")
