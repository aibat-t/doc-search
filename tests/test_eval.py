from doc_search.eval import Outcome, Question, first_relevant_rank, summarize

Q = Question(id="q", kind="exact", question="статья 36-1", source="zakon.pdf", section="Статья 36-1.")


def hit(source: str, section: str) -> dict[str, str]:
    return {"source": source, "section": section}


def test_relevance_by_source_and_section() -> None:
    hits = [
        hit("zakon.pdf", "Глава 8 > Статья 36-2. Регистрация"),
        hit("other.pdf", "Статья 36-1. Перенос"),            # не тот файл
        hit("zakon.pdf", "Глава 8 > Статья 36-1. Перенос"),
    ]
    assert first_relevant_rank(hits, Q) == 3


def test_dot_in_section_prevents_prefix_match() -> None:
    assert first_relevant_rank([hit("zakon.pdf", "Статья 36-10. Что-то")], Q) is None


def test_recall_and_mrr() -> None:
    outcomes = [Outcome(Q, 1, 10.0), Outcome(Q, 2, 20.0), Outcome(Q, None, 30.0), Outcome(Q, 4, 40.0)]
    s = summarize(outcomes)
    assert s["recall"] == 0.75
    assert s["mrr"] == (1 + 0.5 + 0 + 0.25) / 4
