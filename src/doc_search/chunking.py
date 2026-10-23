"""Этап 2 пайплайна: блоки → чанки нужного размера с перекрытием."""

import math
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import groupby

from doc_search.parsing import Block

TokenCounter = Callable[[str], int]

# Режем по концу предложения (точка, !, ?, …, ;) и после двоеточия перед нумерованным пунктом:
# «используются следующие понятия: 1) ...» → «...понятия:» | «1) ...».
_SENTENCE_END = re.compile(r"(?<=[.!?…;])\s+|(?<=:)\s+(?=\d+(?:-\d+)?\)\s)")

# Статья-словарь: «Основные понятия», «Термины», «Негізгі ұғымдар».
_GLOSSARY = re.compile(r"понятия|термин|ұғым", re.IGNORECASE)
# Начало пункта словаря: «17) », «22-2) ».
_ITEM_START = re.compile(r"^\d+(?:-\d+)?\)\s")


def approx_tokens(text: str) -> int:
    """Грубая оценка без токенизатора: в русском и казахском ≈ 1.5 токена на слово.
    В пайплайне используется настоящий токенизатор модели, это — для тестов."""
    return max(1, math.ceil(len(text.split()) * 1.5))


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str        # кусок документа — его показываем пользователю
    header: str      # «Название документа > Раздел»
    section: str
    page: int | None
    tokens: int

    @property
    def embed_text(self) -> str:
        """Что уходит в модель: заголовок + текст. Так «Срок — 10 дней» знает, срок ЧЕГО."""
        return f"{self.header}\n{self.text}" if self.header else self.text


@dataclass(frozen=True)
class _Sentence:
    text: str
    page: int | None
    tokens: int


def chunk_blocks(
    blocks: Iterable[Block],
    title: str,
    max_tokens: int = 350,
    overlap_tokens: int = 50,
    count_tokens: TokenCounter = approx_tokens,
) -> list[Chunk]:
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens должен быть меньше max_tokens")

    chunks: list[Chunk] = []
    # Разделы не смешиваем: чанк никогда не начинается в одной статье и не заканчивается в другой.
    for section, section_blocks in groupby(blocks, key=lambda b: b.section):
        header = " > ".join(part for part in (title, section) if part)
        sentences = list(_sentences(section_blocks, max_tokens, count_tokens))
        groups = _glossary_items(sentences) if _GLOSSARY.search(section) else [sentences]
        for group in groups:
            for window in _windows(group, max_tokens, overlap_tokens):
                chunks.append(
                    Chunk(
                        index=len(chunks),
                        text=" ".join(s.text for s in window),
                        header=header,
                        section=section,
                        page=window[0].page,
                        tokens=sum(s.tokens for s in window),
                    )
                )
    return chunks


def _glossary_items(sentences: list[_Sentence]) -> list[list[_Sentence]]:
    """Словарь «17) термин – ...; 18) термин – ...» → одно определение на чанк.

    Иначе в чанк попадает десяток несвязанных терминов, его вектор становится
    «средним по всему», и такой чанк всплывает в топе почти на любой запрос.
    """
    items: list[list[_Sentence]] = []
    for sentence in sentences:
        if not items or _ITEM_START.match(sentence.text):
            items.append([sentence])
        else:
            items[-1].append(sentence)      # продолжение определения: «...обезличивание. Агрегированные...»
    return items


def _sentences(blocks: Iterable[Block], max_tokens: int, count: TokenCounter) -> Iterator[_Sentence]:
    """Блоки → предложения. Предложение длиннее max_tokens режем по словам на равные части."""
    for block in blocks:
        for raw in _SENTENCE_END.split(block.text):
            text = raw.strip()
            if not text:
                continue
            tokens = count(text)
            if tokens <= max_tokens:
                yield _Sentence(text, block.page, tokens)
                continue
            words = text.split()
            parts = math.ceil(tokens / max_tokens)
            size = math.ceil(len(words) / parts)
            for i in range(0, len(words), size):
                piece = " ".join(words[i : i + size])
                yield _Sentence(piece, block.page, count(piece))


def _windows(sentences: list[_Sentence], max_tokens: int, overlap_tokens: int) -> Iterator[list[_Sentence]]:
    """Набираем предложения, пока влезают в max_tokens. Не влезло — отдаём чанк,
    а следующий начинаем с хвоста предыдущего (перекрытие)."""
    window: list[_Sentence] = []
    size = 0
    for sentence in sentences:
        if window and size + sentence.tokens > max_tokens:
            yield window
            window = _tail(window, overlap_tokens)
            size = sum(s.tokens for s in window)
            if size + sentence.tokens > max_tokens:   # хвост + новое предложение не влезают
                window, size = [], 0
        window.append(sentence)
        size += sentence.tokens
    if window:
        yield window


def _tail(window: list[_Sentence], overlap_tokens: int) -> list[_Sentence]:
    """Последние целые предложения чанка, суммарно не больше overlap_tokens."""
    tail: list[_Sentence] = []
    size = 0
    for sentence in reversed(window):
        if size + sentence.tokens > overlap_tokens:
            break
        tail.append(sentence)
        size += sentence.tokens
    tail.reverse()
    return tail
