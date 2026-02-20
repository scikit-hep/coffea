import pytest

from coffea.compute.util import merge_ranges


def test_merge_ranges():
    assert merge_ranges([(0, 10), (20, 30)], [(10, 15)]) == [(0, 15), (20, 30)]
    assert merge_ranges([(0, 10), (20, 30)], [(15, 20)]) == [(0, 10), (15, 30)]
    assert merge_ranges([(0, 10), (20, 30)], [(30, 40)]) == [(0, 10), (20, 40)]
    assert merge_ranges([(0, 10), (20, 30)], [(35, 40)]) == [
        (0, 10),
        (20, 30),
        (35, 40),
    ]
    assert merge_ranges([(0, 10), (20, 30)], [(10, 20)]) == [(0, 30)]
    with pytest.raises(ValueError):
        merge_ranges([(0, 10), (20, 30)], [(5, 25)])
    with pytest.raises(ValueError):
        merge_ranges([(0, 10), (20, 30)], [(15, 25)])
