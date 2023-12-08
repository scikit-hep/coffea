import copy

import awkward
import numpy


def max_chunks(fileset, maxchunks=None):
    return slice_chunks(fileset, slice(maxchunks))


def slice_chunks(fileset, theslice=slice(None)):
    if not isinstance(theslice, slice):
        theslice = slice(theslice)

    out = copy.deepcopy(fileset)
    for name, entry in fileset.items():
        for fname, finfo in entry["files"].items():
            out[name]["files"][fname]["steps"] = finfo["steps"][theslice]

    return out


def get_failed_steps_for_dataset(dataset, report):
    failed_dataset = {}
    failures = report[~awkward.is_none(report.exception)]

    if not awkward.all(report.args[:, 4] == "True"):
        raise RuntimeError(
            "step specification is not completely in starts/stops form, failed-step extraction is not available for steps_per_file."
        )

    for fdesc in dataset.values():
        if "steps" not in fdesc:
            raise RuntimeError(
                "steps specification not found in dataset, please specify steps in input dataset."
            )

    fnames = set(dataset.keys())
    rnames = set(numpy.unique(report.args[:, 0][:, 1:-1:]))
    if not rnames.issubset(fnames):
        raise RuntimeError(
            f"Files: {rnames - fnames} are not in input dataset, please ensure report corresponds to input dataset!"
        )

    for failure in failures:
        args_as_types = tuple(eval(arg) for arg in failure.args)

        fname, object_path, start, stop, is_step = args_as_types

        if fname in failed_dataset:
            failed_dataset[fname]["steps"].append([start, stop])
        else:
            failed_dataset[fname] = copy.deepcopy(dataset[fname])
            failed_dataset[fname]["steps"] = [[start, stop]]

    return failed_dataset


def get_failed_steps_for_fileset(fileset, report_dict):
    failed_fileset = {}
    for name, dataset in fileset.items():
        failed_dataset = get_failed_steps_for_dataset(dataset, report_dict[name])
        if len(failed_dataset) > 0:
            failed_fileset[name] = failed_dataset
    return failed_fileset