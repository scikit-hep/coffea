import json
import warnings

import awkward
import numpy
import uproot

from coffea.nanoevents.mapping.base import BaseSourceMapping, UUIDOpener
from coffea.nanoevents.util import quote, tuple_to_key


class TrivialUprootOpener(UUIDOpener):
    def __init__(self, uuid_pfnmap, uproot_options={}):
        super().__init__(uuid_pfnmap)
        self._uproot_options = uproot_options

    def open_uuid(self, uuid):
        pfn = self._uuid_pfnmap[uuid]
        rootdir = uproot.open({pfn: None}, **self._uproot_options)
        if str(rootdir.file.uuid) != uuid:
            raise RuntimeError(
                f"UUID of file {pfn} does not match expected value ({uuid})"
            )
        return rootdir


class CannotBeNanoEvents(Exception):
    pass


def _lazify_form(form, prefix, docstr=None):
    if not isinstance(form, dict) or "class" not in form:
        raise RuntimeError("form should have been normalized by now")

    parameters = _lazify_parameters(form.get("parameters", {}), docstr=docstr)
    if form["class"].startswith("ListOffset"):
        # awkward will add !offsets
        form["form_key"] = quote(prefix)
        form["content"] = _lazify_form(
            form["content"], prefix + ",!content", docstr=docstr
        )
    elif form["class"] == "NumpyArray":
        form["form_key"] = quote(prefix)
        if parameters:
            form["parameters"] = parameters
    elif form["class"] == "RegularArray":
        form["content"] = _lazify_form(
            form["content"], prefix + ",!content", docstr=docstr
        )
        if parameters:
            form["parameters"] = parameters
    elif form["class"] == "IndexedOptionArray":
        if (
            form["content"]["class"] != "NumpyArray"
            or form["content"]["primitive"] != "bool"
        ):
            raise ValueError(
                "Only boolean NumpyArrays can be created dynamically if "
                "missing in file!"
            )
        assert prefix.endswith("!load")
        form["form_key"] = quote(prefix + "allowmissing,!index")
        form["content"] = _lazify_form(
            form["content"], prefix + "allowmissing,!content", docstr=docstr
        )
        if parameters:
            form["parameters"] = parameters
    elif form["class"] == "RecordArray":
        newfields, newcontents = [], []
        for field, value in zip(form["fields"], form["contents"]):
            if "," in field or "!" in field:
                # Could also skip here
                raise CannotBeNanoEvents(
                    f"A subform contains a field with invalid characters: {field}"
                )
            elif field.startswith("@"):
                # workaround uproot5 bug
                continue

            newfields.append(field)
            newcontents.append(_lazify_form(value, prefix + f",{field},!item"))
        form["fields"] = newfields
        form["contents"] = newcontents
        if parameters:
            form["parameters"] = parameters
    else:
        raise CannotBeNanoEvents("Unknown form")
    return form


def _lazify_parameters(form_parameters, docstr=None):
    parameters = {}
    if "__array__" in form_parameters:
        parameters["__array__"] = form_parameters["__array__"]
    if docstr is not None:
        parameters["__doc__"] = docstr
    return parameters


class UprootSourceMapping(BaseSourceMapping):
    _debug = False
    _fix_awkward_form_of_iter = False

    def __init__(
        self,
        fileopener,
        start,
        stop,
        cache=None,
        access_log=None,
        use_ak_forth=False,
        decompression_executor=None,
        interpretation_executor=None,
    ):
        super().__init__(fileopener, start, stop, cache, access_log, use_ak_forth)
        self.decompression_executor = (
            decompression_executor or uproot.source.futures.TrivialExecutor()
        )
        self.interpretation_executor = (
            interpretation_executor or uproot.source.futures.TrivialExecutor()
        )

    @classmethod
    def _extract_base_form(cls, tree, iteritems_options={}):
        branch_forms = {}
        for key, branch in tree.iteritems(**iteritems_options):
            if key in branch_forms:
                warnings.warn(
                    f"Found duplicate branch {key} in {tree}, taking first instance"
                )
                continue
            if "," in key or "!" in key:
                warnings.warn(
                    f"Skipping {key} because it contains characters that NanoEvents cannot accept [,!]"
                )
                continue
            if len(branch):
                # The branch is split and its sub-branches will be enumerated by tree.iteritems
                continue
            if isinstance(
                branch.interpretation,
                uproot.interpretation.identify.UnknownInterpretation,
            ):
                warnings.warn(f"Skipping {key} as it is not interpretable by Uproot")
                continue
            try:
                form = branch.interpretation.awkward_form(None)
            except uproot.interpretation.objects.CannotBeAwkward:
                warnings.warn(
                    f"Skipping {key} as it is it cannot be represented as an Awkward array"
                )
                continue
            # until awkward-forth is available, this fixer is necessary
            if cls._fix_awkward_form_of_iter:
                form = uproot._util.recursively_fix_awkward_form_of_iter(
                    awkward, branch.interpretation, form
                )
            form = json.loads(
                form.to_json()
            )  # normalizes form (expand NumpyArray classes)
            try:
                form = _lazify_form(form, f"{key},!load", docstr=branch.title)
            except CannotBeNanoEvents as ex:
                warnings.warn(
                    f"Skipping {key} as it is not interpretable by NanoEvents\nDetails: {ex}"
                )
                continue
            branch_forms[key] = form
        return {
            "class": "RecordArray",
            "contents": [item for item in branch_forms.values()],
            "fields": [key for key in branch_forms.keys()],
            "parameters": {"__doc__": tree.title},
            "form_key": None,
        }

    def key_root(self):
        return "UprootSourceMapping:"

    def preload_column_source(self, uuid, path_in_source, source):
        """To save a double-open when using NanoEventsFactory.from_file"""
        key = self.key_root() + tuple_to_key((uuid, path_in_source))
        self._cache[key] = source

    def get_column_handle(self, columnsource, name, allow_missing):
        if allow_missing:
            return columnsource[name] if name in columnsource else None
        return columnsource[name]

    def extract_column(
        self, columnhandle, start, stop, allow_missing, use_ak_forth=True
    ):
        # make sure uproot is single-core since our calling context might not be
        if allow_missing and columnhandle is None:

            return awkward.contents.IndexedOptionArray(
                awkward.index.Index64(numpy.full(stop - start, -1, dtype=numpy.int64)),
                awkward.contents.NumpyArray(numpy.array([], dtype=bool)),
            )
        elif not allow_missing and columnhandle is None:
            raise RuntimeError(
                "Received columnhandle of None when missing column in file is not allowed!"
            )

        interp = columnhandle.interpretation
        interp._forth = use_ak_forth

        the_array = columnhandle.array(
            interp,
            entry_start=start,
            entry_stop=stop,
            decompression_executor=self.decompression_executor,
            interpretation_executor=self.interpretation_executor,
        )

        if allow_missing:
            the_array = awkward.contents.IndexedOptionArray(
                awkward.index.Index64(numpy.arange(stop - start, dtype=numpy.int64)),
                awkward.contents.NumpyArray(the_array),
            )

        return the_array

    def __len__(self):
        return self._stop - self._start

    def __iter__(self):
        raise NotImplementedError
