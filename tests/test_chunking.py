import pytest

from doc_search.chunking import approx_tokens, chunk_blocks
from doc_search.parsing import Block


def _sentences(n: int, words: int = 10) -> str:
    return " ".join(f"Предложение номер {i} " + "слово " * (words - 3) + "конец." for i in range(n))


def test_chunks_fit_into_max_tokens() -> None:
    chunks = chunk_blocks([Block(_sentences(40))], "Док", max_tokens=60, overlap_tokens=15)
    assert len(chunks) > 1
    assert all(c.tokens <= 60 for c in chunks)


def test_neighbour_chunks_overlap() -> None:
    chunks = chunk_blocks([Block(_sentences(40))], "Док", max_tokens=60, overlap_tokens=20)
    for prev, nxt in zip(chunks, chunks[1:]):
        last_sentence = prev.text.split(". ")[-1]
        assert last_sentence in nxt.text   # хвост предыдущего = начало следующего


def test_no_overlap_when_zero() -> None:
    chunks = chunk_blocks([Block(_sentences(20))], "Док", max_tokens=60, overlap_tokens=0)
    total = sum(approx_tokens(s) for s in [c.text for c in chunks])
    assert total == pytest.approx(approx_tokens(_sentences(20)), rel=0.1)


def test_sections_are_not_mixed_and_header_has_context() -> None:
    blocks = [
        Block("Карту можно заблокировать в приложении.", "Статья 1"),
        Block("Срок составляет 10 дней.", "Статья 2"),
    ]
    chunks = chunk_blocks(blocks, "Правила карт", max_tokens=100)
    assert [c.section for c in chunks] == ["Статья 1", "Статья 2"]
    assert chunks[1].embed_text == "Правила карт > Статья 2\nСрок составляет 10 дней."


def test_very_long_sentence_is_split() -> None:
    long_sentence = "слово " * 300 + "конец."
    chunks = chunk_blocks([Block(long_sentence)], "", max_tokens=100, overlap_tokens=0)
    assert len(chunks) >= 5
    assert all(c.tokens <= 100 for c in chunks)


def test_page_of_first_sentence() -> None:
    blocks = [Block("Первая.", page=3), Block("Вторая.", page=4)]
    assert chunk_blocks(blocks, "", max_tokens=100)[0].page == 3


def test_overlap_must_be_smaller_than_chunk() -> None:
    with pytest.raises(ValueError):
        chunk_blocks([Block("текст")], "", max_tokens=50, overlap_tokens=50)
