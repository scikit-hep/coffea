"""Helpers for combining awkward forms across files and datasets.

A dataset's saved form is the union of its files' forms (NanoAOD files differ in
``HLT_*``/``GenModel`` fields). Imports only awkward so both ``preprocess`` and
``filespec`` can use it.
"""

from __future__ import annotations

import awkward

__all__ = [
    "union_form_jsonstr",
    "sort_form_fields",
    "prune_form_fields",
    "encode_field_bitset",
    "decode_field_bitset",
]


def union_form_jsonstr(forms: list, sort_fields: bool = False) -> str | None:
    """Union form (as JSON) of a list of flat-tuple-like awkward forms; consumes the list.

    Returns None for an empty list. Fields keep merge order unless ``sort_fields``.
    """
    union_array = None
    while len(forms):
        new_array = awkward.Array(forms.pop().length_zero_array())
        if union_array is None:
            union_array = new_array
        else:
            union_array = awkward.to_packed(
                awkward.merge_union_of_records(
                    awkward.concatenate([union_array, new_array]), axis=0
                )
            )
            union_array.layout.parameters.update(new_array.layout.parameters)
    if union_array is None:
        return None

    union_form = union_array.layout.form
    for icontent, content in enumerate(union_form.contents):
        if isinstance(content, awkward.forms.IndexedOptionForm):
            if (
                not isinstance(content.content, awkward.forms.NumpyForm)
                or content.content.primitive != "bool"
            ):
                raise ValueError(
                    "IndexedOptionArrays can only contain NumpyArrays of "
                    "bools in mergers of flat-tuple-like schemas!"
                )
            parameters = (
                content.content.parameters.copy()
                if content.content.parameters is not None
                else {}
            )
            # re-create IndexOptionForm with parameters of lower level array
            union_form.contents[icontent] = awkward.forms.IndexedOptionForm(
                content.index,
                content.content,
                parameters=parameters,
                form_key=content.form_key,
            )
    if sort_fields:
        union_form = sort_form_fields(union_form)
    return union_form.to_json()


def _sort_record_nodes(node) -> None:
    if isinstance(node, dict):
        if (
            node.get("class") == "RecordArray"
            and isinstance(node.get("fields"), list)
            and isinstance(node.get("contents"), list)
        ):
            pairs = sorted(
                zip(node["fields"], node["contents"]), key=lambda pair: pair[0]
            )
            node["fields"] = [field for field, _ in pairs]
            node["contents"] = [content for _, content in pairs]
        for value in node.values():
            _sort_record_nodes(value)
    elif isinstance(node, list):
        for value in node:
            _sort_record_nodes(value)


def sort_form_fields(form: awkward.forms.Form) -> awkward.forms.Form:
    """Copy of ``form`` with record fields recursively sorted by name (tuples untouched)."""
    form_dict = form.to_dict(verbose=True)
    _sort_record_nodes(form_dict)
    return awkward.forms.from_dict(form_dict)


def prune_form_fields(
    form: awkward.forms.Form, keep_fields: set[str]
) -> awkward.forms.Form:
    """Copy of ``form`` keeping only the top-level record fields in ``keep_fields``.

    Union forms merge file forms at the top level, so only that level is pruned.
    """
    form_dict = form.to_dict(verbose=True)
    if not (
        isinstance(form_dict.get("fields"), list)
        and isinstance(form_dict.get("contents"), list)
    ):
        return form
    pairs = [
        (field, content)
        for field, content in zip(form_dict["fields"], form_dict["contents"])
        if field in keep_fields
    ]
    form_dict["fields"] = [field for field, _ in pairs]
    form_dict["contents"] = [content for _, content in pairs]
    return awkward.forms.from_dict(form_dict)


def encode_field_bitset(present_fields, union_fields: list[str]) -> str:
    """Hex bitset over ``union_fields``: bit ``i`` set means ``union_fields[i]`` is present.

    Fields outside ``union_fields`` are ignored.
    """
    present = set(present_fields)
    bits = 0
    for index, field in enumerate(union_fields):
        if field in present:
            bits |= 1 << index
    return format(bits, "x")


def decode_field_bitset(bitset: str, union_fields: list[str]) -> set[str]:
    """Decode a hex bitset string (see :func:`encode_field_bitset`) into a set of field names."""
    bits = int(bitset, 16)
    return {field for index, field in enumerate(union_fields) if (bits >> index) & 1}
