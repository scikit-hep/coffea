from coffea.nanoevents import transforms
from coffea.nanoevents.util import concat, quote


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
