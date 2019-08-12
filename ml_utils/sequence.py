"""utils for operating on sequences."""

from itertools import islice
from typing import Iterable


def sliding_window(seq: Iterable, window_size: int = 2) -> object:
    """returns a sliding window (of width n) over data from the iterable
    s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    copied from https://docs.python.org/release/2.3.5/lib/itertools-example.html.
    """
    it = iter(seq)
    result = tuple(islice(it, window_size))
    if len(result) == window_size:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
