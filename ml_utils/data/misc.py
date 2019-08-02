"""misc data utils."""

import itertools

from typing import Sequence, Tuple


def partition_items(items: Sequence, sizes: Sequence[float]) -> Tuple[Sequence]:
    """break sequence of items up into fractional partitions."""
    if not sum(sizes) == 1:
        raise ValueError(f"expected sum(sizes) to be 1. recieved {sum(sizes)}.")
    if any(not 0 <= x <= 1 for x in sizes):
        raise ValueError(f"recieved out of bounds partition sizes: {sizes}.")

    div = [0, *[int(x * len(items)) for x in itertools.accumulate(sizes)]]
    split = [items[div[i] : div[i + 1]] for i in range(len(div) - 1)]

    return split
