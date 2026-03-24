import hashlib
import json
import math

__all__ = [
    "split_fileset",
    "hash_fileset",
]


def hash_fileset(chunk):
    """
    Return a stable SHA-256 hash for a fileset chunk.
    The hash considers dataset names, file paths in sorted order.

    Input
    chunk: fileset dict  {dataset: {"files": {path: treename, ...}, ...}, ...}

    Output
    hex string uniquely identifying this chunk's file content
    """
    canonical = {
        dataset: dict(sorted(files.get("files", {}).items()))
        for dataset, files in chunk.items()
    }
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def split_fileset(fileset, strategy=None, datasets=None, percentage=None):
    """
    Split the fileset into chunks to enable getting a partial result if one or several
    of the chunks failed to produce a result while being processed.
    One chunk is one partial fileset(unique combination of files), these are not usual coffea chunks.


    Input
    fileset:    {dataset: {"files": {path: treename, ...}}}
    strategy:   "by_dataset" — one dataset is one chunk; None — all datasets together
    percentage: integer that divides 100 evenly (20, 25, 50...).
                Each chunk gets this percentage of each dataset's files.
    datasets: list, callable or tuple of datasets' names

    Output
    List of fileset dicts
    If strategy only:
        chunks = _split_fileset(fileset, "by_dataset") - one chunk per dataset
    If percentage only:
        chunks = _split_fileset(fileset, percentage=50) - 2 chunks (50 of each dataset in 1st chunk and 2nd, mixed chunks
    If strategy and percentage:
        chunks = _split_fileset(fileset, "by_dataset", percentage=50) - N_datasets * 2 chunks, not mixed chunks
    If datasets + any/nothing:
        strategies are only applied to chosen datasets
    """
    if strategy is not None and strategy != "by_dataset":
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'by_dataset' or None.")
    if percentage is not None:
        if (
            not isinstance(percentage, int)
            or not (1 <= percentage <= 100)
            or 100 % percentage != 0
        ):
            raise ValueError(
                "'percentage' must be an int that divides 100 evenly (e.g. 10, 20, 25, 50)."
            )

    if datasets is None:
        pass
    elif callable(datasets):
        fileset = {k: v for k, v in fileset.items() if datasets(k)}
    else:
        fileset = {k: fileset[k] for k in datasets if k in fileset}

    if strategy == "by_dataset":
        groups = [{name: data} for name, data in fileset.items()]
    else:
        groups = [fileset]

    if percentage is None:
        return groups

    n_chunks = 100 // percentage
    result = []
    for group in groups:
        for bin_idx in range(n_chunks):
            chunk = {}
            for dataset, data in group.items():
                files = data.get("files", {})
                if not files:
                    continue
                file_items = list(files.items())
                n = len(file_items)
                chunk_size = max(1, math.ceil(n / n_chunks))
                start = bin_idx * chunk_size
                end = min(start + chunk_size, n)
                if start >= n:
                    continue
                chunk[dataset] = {**data, "files": dict(file_items[start:end])}
            if chunk:
                result.append(chunk)
    return result
