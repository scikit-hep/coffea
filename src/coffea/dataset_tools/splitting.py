import hashlib
import json
import math

__all__ = [
    "split_fileset",
    "hash_fileset",
]


def _canonicalize_files(files):
    """
    Return a sorted list of ``(path, treename-or-None)`` pairs.

    Accepts either a mapping ``{path: treename}`` or a list of paths. The
    list form has no per-file treename, so the second element is ``None``.
    """
    if isinstance(files, dict):
        return sorted(files.items())
    if isinstance(files, (list, tuple)):
        return [(p, None) for p in sorted(files)]
    raise TypeError(
        f"Expected 'files' to be a dict or list, got {type(files).__name__}"
    )


def hash_fileset(chunk):
    """
    Return a stable SHA-256 hash for a fileset chunk.

    The hash considers dataset names, file paths and a fixed set of
    output-affecting dataset-level fields — ``treename``, ``preload`` and
    ``metadata`` — in a canonical sorted form, so chunks that differ in any
    of those fields produce different hashes. Any other dataset-level keys
    (e.g. preprocessing bookkeeping such as ``compressed_form``) are ignored
    by the hash on purpose, so they may evolve without invalidating caches.

    Parameters
    ----------
        chunk : dict
            A self-contained fileset chunk such as
            ``{dataset: {"files": {path: treename, ...}, "treename": ...,
            "preload": [...], "metadata": {...}}, ...}``. List-format
            ``files`` values are accepted only when accompanied by a
            dataset-level ``"treename"`` field (use :func:`split_fileset`
            with ``treename=...`` to produce such chunks from a bare list
            fileset).

    Returns
    -------
        out : str
            Hex string uniquely identifying this chunk's contents.
    """
    canonical = {}
    for dataset, data in sorted(chunk.items()):
        if not isinstance(data, dict):
            raise TypeError(
                f"Unsupported dataset value type for '{dataset}': "
                f"{type(data).__name__}. Pass a fileset chunk produced by "
                "split_fileset, or supply a dataset-level 'treename' for "
                "list-format filesets."
            )
        files = data.get("files", {})
        if isinstance(files, (list, tuple)) and "treename" not in data:
            raise ValueError(
                f"Dataset '{dataset}' uses list-format files without a "
                "dataset-level 'treename' field; the chunk is not "
                "self-contained and cannot be hashed reliably."
            )
        entry = {"files": _canonicalize_files(files)}
        for key in ("treename", "preload", "metadata"):
            if key in data:
                value = data[key]
                if key == "preload" and isinstance(
                    value, (list, tuple, set, frozenset)
                ):
                    value = sorted(value)
                entry[key] = value
        canonical[dataset] = entry
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def _get_files_entries(data):
    """
    Return ``(kind, entries)`` for a dataset's files in sorted order.

    ``kind`` is ``"list"`` or ``"dict"`` and indicates the original container
    shape so chunks rebuild with the same schema. Entries are either ``path``
    strings (list form) or ``(path, treename)`` tuples (dict form).
    """
    files = data.get("files", {})
    if isinstance(files, dict):
        return "dict", sorted(files.items())
    if isinstance(files, (list, tuple)):
        return "list", sorted(files)
    raise TypeError(
        f"Expected dataset['files'] to be a dict or list, "
        f"got {type(files).__name__}"
    )


def _rebuild_dataset(data, files_kind, files_slice):
    """
    Rebuild a dataset entry with the sliced files in the original schema.
    """
    new = {**data}
    if files_kind == "dict":
        new["files"] = dict(files_slice)
    else:
        new["files"] = list(files_slice)
    return new


def _normalize_fileset(fileset, treename):
    """
    Promote bare-list datasets to ``{"files": [...], "treename": treename}``.

    A bare list ``{dataset: [path, ...]}`` is a valid Runner input only when
    a ``treename`` is passed alongside it; on its own it is not a
    self-contained chunk. To keep chunks usable with :func:`hash_fileset`
    and cache-keyed workflows, list-format datasets are normalized to dict
    form here using the supplied ``treename``.
    """
    normalized = {}
    for dataset, data in fileset.items():
        if isinstance(data, (list, tuple)):
            if treename is None:
                raise ValueError(
                    f"Dataset '{dataset}' uses list-format files; "
                    "'treename' must be supplied to make the chunk "
                    "self-contained."
                )
            normalized[dataset] = {"files": list(data), "treename": treename}
        elif isinstance(data, dict):
            files = data.get("files", {})
            if (
                isinstance(files, (list, tuple))
                and "treename" not in data
                and treename is None
            ):
                raise ValueError(
                    f"Dataset '{dataset}' uses list-format files without a "
                    "dataset-level 'treename' field; 'treename' must be "
                    "supplied to make the chunk self-contained."
                )
            if (
                isinstance(files, (list, tuple))
                and "treename" not in data
                and treename is not None
            ):
                normalized[dataset] = {**data, "treename": treename}
            else:
                normalized[dataset] = data
        else:
            raise TypeError(
                f"Unsupported dataset value type for '{dataset}': "
                f"{type(data).__name__}"
            )
    return normalized


def split_fileset(
    fileset, strategy=None, datasets=None, percentage=None, treename=None
):
    """
    Split a fileset into partial filesets so that a partial result can still be
    obtained if one or more of them fail during processing.

    Each returned element is a partial fileset (a unique combination of files),
    not one of the usual coffea row-range chunks.

    Both fileset schemas accepted by ``coffea.processor.Runner`` are supported.
    For list-format datasets (``{dataset: [path, ...]}`` or
    ``{dataset: {"files": [...], }}`` without an inner ``"treename"`` field),
    the ``treename`` keyword must be supplied; it is folded into each
    resulting chunk so the chunks are self-contained and usable as cache keys
    via :func:`hash_fileset`. File paths are sorted before being sliced into
    bins, so the chunk composition is deterministic regardless of input dict
    insertion order.

    Parameters
    ----------
        fileset : dict
            A fileset of the form ``{dataset: [file, ...]}`` or
            ``{dataset: {"files": {path: treename, ...} | [path, ...], ...}}``.
        strategy : str or None, default None
            ``"by_dataset"`` puts each dataset in its own chunk; ``None`` keeps
            all datasets together.
        datasets : list, tuple, callable or None, default None
            Restrict splitting to a subset of datasets. If callable, it is
            applied to each dataset name and must return a truthy value to
            include it.
        percentage : int or None, default None
            An integer that divides 100 evenly (e.g. 10, 20, 25, 50). Each
            chunk receives this percentage of each dataset's files.
        treename : str or None, default None
            Tree name to attach to list-format datasets so the resulting
            chunks are self-contained. Required when any dataset uses
            list-format files without its own ``"treename"`` field.

    Returns
    -------
        out : list of dict
            The partial filesets. The behaviour depends on the arguments:

            - ``strategy="by_dataset"`` alone: one chunk per dataset.
            - ``percentage=p`` alone: ``100/p`` chunks, each containing ``p``
              percent of every dataset's files (mixed chunks).
            - ``strategy="by_dataset"`` with ``percentage=p``:
              ``N_datasets * (100/p)`` chunks, not mixed.
            - ``datasets`` combined with any of the above restricts splitting
              to the selected datasets.
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

    fileset = _normalize_fileset(fileset, treename)

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
                files_kind, file_items = _get_files_entries(data)
                n = len(file_items)
                if n == 0:
                    continue
                chunk_size = max(1, math.ceil(n / n_chunks))
                start = bin_idx * chunk_size
                end = min(start + chunk_size, n)
                if start >= n:
                    continue
                chunk[dataset] = _rebuild_dataset(
                    data, files_kind, file_items[start:end]
                )
            if chunk:
                result.append(chunk)
    return result
