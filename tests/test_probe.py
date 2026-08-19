import numpy as np

from src.probe import make_grouped_split


def test_grouped_split_is_reproducible_disjoint_and_has_both_classes() -> None:
    groups = np.asarray([f"g{i // 2}" for i in range(200)], dtype=object)
    labels = np.asarray(([0, 1] * 100), dtype=np.int8)

    first = make_grouped_split(
        labels, groups, ratios=(0.7, 0.15, 0.15), seed=42, candidate_assignments=5000
    )
    repeated = make_grouped_split(
        labels, groups, ratios=(0.7, 0.15, 0.15), seed=42, candidate_assignments=5000
    )

    assert np.array_equal(first.train, repeated.train)
    assert np.array_equal(first.validation, repeated.validation)
    assert np.array_equal(first.test, repeated.test)
    split_groups = [set(groups[index]) for index in (first.train, first.validation, first.test)]
    assert not (split_groups[0] & split_groups[1])
    assert not (split_groups[0] & split_groups[2])
    assert not (split_groups[1] & split_groups[2])
    for index in (first.train, first.validation, first.test):
        assert set(labels[index]) == {0, 1}
