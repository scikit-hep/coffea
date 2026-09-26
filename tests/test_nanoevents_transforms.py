import awkward as ak

from coffea.nanoevents import transforms


def test_get_index_ranges_zero_valued_indices():
    # begin=0, end=1 yields the single index [0]; the all-zero sum must not be
    # mistaken for an empty range and replaced with [[]].
    ranges = transforms.get_index_ranges(ak.Array([[0]]), ak.Array([[1]]))
    assert ranges.tolist() == [[[0]]]

    empty = transforms.get_index_ranges(ak.Array([[]]), ak.Array([[]]))
    assert empty.tolist() == [[]]
