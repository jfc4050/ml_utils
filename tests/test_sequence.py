import pytest

from ml_utils import sequence


def test_sliding_window_handles_windowsize_1():
    test_input = range(5)
    expected_output = [(x,) for x in test_input]

    output = sequence.sliding_window(test_input, 1)
    assert list(output) == expected_output

def test_sliding_window_handles_windowsize_2():
    test_input = range(5)
    expected_output = [(0, 1), (1, 2), (2, 3), (3, 4)]

    output = sequence.sliding_window(test_input, 2)
    assert list(output) == expected_output

def test_sliding_window_handles_oversized_windowsize():
    test_input = range(5)
    expected_output = []

    output = sequence.sliding_window(test_input, 10)
    assert list(output) == expected_output
