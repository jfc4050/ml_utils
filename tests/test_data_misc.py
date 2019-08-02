import pytest

from ml_utils.data import misc


@pytest.mark.parametrize("sizes", [[0.5, 0.75], [-0.5, 1.5]])
def test_partition_items_raises_exceptions_for_invalid_partitions(sizes):
    with pytest.raises(ValueError):
        misc.partition_items([1, 2, 3, 4, 5], sizes)


@pytest.mark.parametrize("sizes", [[0.5, 0.5], [0.6, 0.4], [1.0]])
def test_partition_items_gives_correct_sizes(sizes):
    examples = list(range(100))
    groups = misc.partition_items(examples, sizes)
    for group, size in zip(groups, sizes):
        assert len(group) == int(size * len(examples))
