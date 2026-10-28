from doc_search.ranking import rrf


def test_item_high_in_both_lists_wins() -> None:
    vector = ["a", "b", "c"]
    keyword = ["b", "d", "a"]
    fused = [item for item, _ in rrf(vector, keyword)]
    assert fused[0] == "b"                       # 2-е и 1-е место побеждает 1-е и 3-е
    assert set(fused) == {"a", "b", "c", "d"}    # найденное одним поиском не теряется


def test_scores_use_ranks_not_raw_scores() -> None:
    (item, score), *_ = rrf(["x"], ["x"], k=60)
    assert item == "x"
    assert score == 2 / 61


def test_single_list_keeps_order() -> None:
    assert [item for item, _ in rrf([3, 1, 2])] == [3, 1, 2]


def test_empty_lists() -> None:
    assert rrf([], []) == []
