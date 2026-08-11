Range = tuple[int, int]


def merge_ranges(left: list[Range], right: list[Range]) -> list[Range]:
    """Merge adjacent ranges into a minimal set of non-overlapping ranges.

    For example, [(0, 10), (10, 20), (30, 40)] would be merged into [(0, 20), (30, 40)].
    If any ranges overlap (e.g. [(0, 10), (5, 15)]), a ValueError is raised since this likely
    indicates a bug in the caller.
    """
    merged: list[Range] = []
    for start, stop in sorted(left + right):
        if not merged:
            merged.append((start, stop))
            continue
        last_start, last_stop = merged[-1]
        if start < last_stop:
            raise ValueError(
                f"Overlapping ranges: {last_start}-{last_stop} and {start}-{stop}"
            )
        elif start == last_stop:
            merged[-1] = (last_start, stop)
        else:
            merged.append((start, stop))
    return merged
