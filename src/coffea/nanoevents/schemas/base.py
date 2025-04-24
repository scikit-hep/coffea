from coffea.nanoevents import transforms
from coffea.nanoevents.util import concat, quote

import numpy
import awkward

def listarray_form(content, offsets):
    if offsets["class"] != "NumpyArray":
        raise ValueError
    if offsets["primitive"] == "int32":
        arrayclass = "ListOffsetArray"
        offsetstype = "i32"
    elif offsets["primitive"] == "int64":
        arrayclass = "ListOffsetArray"
        offsetstype = "i64"
    else:
        raise ValueError("Unrecognized offsets data type")
    return {
        "class": arrayclass,
        "offsets": offsetstype,
        "content": content,
        "form_key": concat(offsets["form_key"], "!skip"),
    }


def zip_forms(forms, name, record_name=None, offsets=None, bypass=False):
    if not isinstance(forms, dict):
        raise ValueError("Expected a dictionary")
    if all(form["class"].startswith("ListOffsetArray") for form in forms.values()):
        first = next(iter(forms.values()))
        if not all(form["class"] == first["class"] for form in forms.values()):
            print(
                tuple((name, form["class"]) for name, form in forms.items()),
                first["class"],
            )
            raise ValueError
        if not all(form["offsets"] == first["offsets"] for form in forms.values()):
            print(
                tuple((name, form["offsets"]) for name, form in forms.items()),
                first["offsets"],
            )
            raise ValueError
        record = {
            "class": "RecordArray",
            "fields": [k for k in forms.keys()],
            "contents": [form["content"] for form in forms.values()],
            "form_key": quote("!invalid," + name),
        }
        if record_name is not None:
            record["parameters"] = {"__record__": record_name}
        if offsets is None:
            return {
                "class": first["class"],
                "offsets": first["offsets"],
                "content": record,
                "form_key": first["form_key"],
            }
        else:
            return listarray_form(record, offsets)
    elif all(form["class"] == "NumpyArray" for form in forms.values()):
        record = {
            "class": "RecordArray",
            "fields": [key for key in forms.keys()],
            "contents": [value for value in forms.values()],
            "form_key": quote("!invalid," + name),
        }
        if record_name is not None:
            record["parameters"] = {"__record__": record_name}
        return record
    # elif all(form["class"] in [ "RecordArray", "NumpyArray", "ListOffsetArray"] for form in forms.values()):
    elif all("class" in form for form in forms.values()) and not bypass:
        record = {
            "class": "RecordArray",
            "fields": [key for key in forms.keys()],
            "contents": [value for value in forms.values()],
            "form_key": quote("!invalid," + name),
        }
        if record_name is not None:
            record["parameters"] = {"__record__": record_name}
        return record
    else:
        raise NotImplementedError("Cannot zip forms")


def nest_jagged_forms(parent, child, counts_name, name):
    """Place child listarray inside parent listarray as a double-jagged array"""
    if not parent["class"].startswith("ListOffsetArray"):
        raise ValueError
    if parent["content"]["class"] != "RecordArray":
        raise ValueError
    if not child["class"].startswith("ListOffsetArray"):
        raise ValueError
    counts_idx = parent["content"]["fields"].index(counts_name)
    counts = parent["content"]["contents"][counts_idx]
    offsets = transforms.counts2offsets_form(counts)
    inner = listarray_form(child["content"], offsets)
    parent["content"]["fields"].append(name)
    parent["content"]["contents"].append(inner)

#move to transforms?
def local2globalindex(index, counts):
    """
    Convert a local index to a global index

    This is the same as local2global(index, counts2offsets(counts))
    where local2global and counts2offsets are as in coffea.nanoevents.transforms

    TO DO: dask_awkward.map_partitions implementation
    """
    if awkward.backend(index) == "typetracer":
        return index
    offsets = numpy.empty(len(counts) + 1, dtype=numpy.int64)
    offsets[0] = 0
    numpy.cumsum(counts, out=offsets[1:])
    index = index.mask[index >= 0] + offsets[:-1]
    index = index.mask[index < offsets[1:]]  # guard against out of bounds
    # workaround ValueError: can not (unsafe) zip ListOffsetArrays with non-NumpyArray contents
    # index.type is N * var * int32?
    index = awkward.fill_none(index, -1)
    return index

#move to transforms?
def nestedindex_form(indices):
    '''
    Concatenate a list of indices along a new axis
    Outputs a jagged array with same outer shape as index arrays
    Add examples to documentation?

    '''
    if not all(isinstance(index.layout, awkward.contents.listoffsetarray.ListOffsetArray) for index in indices):
        raise RuntimeError
    # return awkward.concatenate([idx[:, None] for idx in indexers], axis=1)

    # store offsets to later reapply them to the arrays
    offsets_stored = indices[0].layout.offsets
    # also store parameters
    parameters = {}
    for i, idx in enumerate(indices):
        if '__doc__' in parameters:
            parameters['__doc__'] += ' and '
            parameters['__doc__'] += awkward.parameters(idx)['__doc__']
        else:
            parameters['__doc__'] = 'nested from '
            parameters['__doc__'] += awkward.parameters(idx)['__doc__']
        # flatten the index
        indices[i] = awkward.Array(idx.layout.content)

    n = len(indices)
    out = numpy.empty(n * len(indices[0]), dtype="int64")
    for i, idx in enumerate(indices):
        #  index arrays should all be same shape flat arrays
        out[i::n] = idx
    offsets = numpy.arange(0, len(out) + 1, n, dtype=numpy.int64)
    out = awkward.Array(
        awkward.contents.ListOffsetArray(
            awkward.index.Index64(offsets),
            awkward.contents.NumpyArray(out),
        )
    )
    #reapply the offsets
    out = awkward.Array(
        awkward.contents.ListOffsetArray(
            offsets_stored,
            out.layout,
            parameters = parameters,
        )
    )
    return out

#move to transforms?
def counts2nestedindex_form(local_counts, target_offsets):
    """Turn jagged local counts into doubly-jagged global index into a target
    Outputs a jagged array with same axis-0 shape as counts axis-1
    """
    if not isinstance(local_counts.layout, awkward.contents.listoffsetarray.ListOffsetArray):
        raise RuntimeError
    if not isinstance(target_offsets.layout, awkward.contents.numpyarray.NumpyArray):
        raise RuntimeError

    # count offsets the same way as with counts2offsets in coffea.nanoevents.transforms
    offsets = numpy.empty(len(target_offsets) + 1, dtype=numpy.int64)
    offsets[0] = 0
    numpy.cumsum(target_offsets, out=offsets[1:])

    # store offsets to later reapply them to the arrays
    offsets_stored = local_counts.layout.offsets

    out = awkward.unflatten(
        numpy.arange(offsets[-1], dtype=numpy.int64),
        awkward.flatten(local_counts),
    )
    #reapply the offsets
    out = awkward.Array(
        awkward.contents.ListOffsetArray(
            offsets_stored,
            out.layout,
        )
    )
    return out

#move to transforms?
def counts2offsets(counts):
    #Cumulative sum of counts
    offsets = numpy.empty(len(counts) + 1, dtype=numpy.int64)
    offsets[0] = 0
    numpy.cumsum(counts, out=offsets[1:])
    return offsets

#move to transforms?
def check_equal_lengths(
        contents: list[awkward.contents.Content],
) -> int | awkward._nplikes.shape.UnknownLength:
    length = contents[0].length
    for layout in contents:
        if layout.length != length:
            raise ValueError("all arrays must have the same length")
    return length

def zip_depth2(content, offsets, with_name, behavior, parameters=None):
    # if with_name is not None:
    #     if parameters is None:
    #         parameters = {}
    #     else:
    #         parameters = dict(parameters)
    #     parameters["__record__"] = with_name

    fields = list(content.keys())
    contents = [
        # take contents 2 layers deep
        v.layout.content
        for v in content.values()
    ]
    length = check_equal_lengths(contents)
    out = awkward.contents.ListOffsetArray(
        offsets=offsets,
        content=awkward.contents.RecordArray(
            contents, fields, length=length
        ),
    )
    out = awkward.Array(out, behavior=behavior, with_name=with_name)
    return out

def zip_depth1(content, with_name, behavior, parameters=None):
    # if with_name is not None:
    #     if parameters is None:
    #         parameters = {}
    #     else:
    #         parameters = dict(parameters)
    #     parameters["__record__"] = with_name

    fields = list(content.keys())
    contents = [
        # take contents 1 layer deep
        v.layout
        for v in content.values()
    ]
    length = check_equal_lengths(contents)
    out = awkward.contents.RecordArray(
        contents, fields, length
    )
    out = awkward.Array(out, behavior=behavior, with_name=with_name)
    return out

class BaseSchema:
    """Base schema builder

    The basic schema is essentially unchanged from the original ROOT file.
    A top-level `base.NanoEvents` object is returned, where each original branch
    form is accessible as a direct descendant.
    """

    __dask_capable__ = True

    def __init__(self, base_form, *args, **kwargs):
        params = dict(base_form.get("parameters", {}))
        params["__record__"] = "NanoEvents"
        if "metadata" in params and params["metadata"] is None:
            params.pop("metadata")
        params.setdefault("metadata", {})
        self._form = {
            "class": "RecordArray",
            "fields": base_form["fields"],
            "contents": base_form["contents"],
            "parameters": params,
            "form_key": None,
        }

    @property
    def form(self):
        """Awkward form of this schema (dict)"""
        return self._form

    @classmethod
    def behavior(cls):
        """Behaviors necessary to implement this schema (dict)"""
        from coffea.nanoevents.methods import base

        return base.behavior
