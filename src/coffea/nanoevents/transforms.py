import copy

import awkward
import numba
import numpy

from coffea.nanoevents.util import concat


def to_layout(array):
    if isinstance(array, awkward.contents.Content):
        return array
    return array.layout


def ensure_array(arraylike):
    if isinstance(arraylike, (awkward.contents.Content, awkward.Array)):
        return awkward.to_numpy(arraylike)
    elif isinstance(arraylike, awkward.index.Index):
        return arraylike.data
    else:
        return numpy.asarray(arraylike)


def data(stack):
    """Extract content from array
    (currently a noop, can probably take place of !content)

    Signature: data,!data
    """
    pass


def offsets(stack):
    """Extract offsets from ListOffsetArray

    Signature: array,!offsets
    """
    stack.append(to_layout(stack.pop()).offsets)


def mask(stack):
    """Extract mask from a masked array

    Signature: array,!mask
    """
    stack.append(to_layout(stack.pop()).mask)


def index(stack):
    """Extract index from array

    Signature: array,!index
    """
    stack.append(to_layout(stack.pop()).index)


def starts(stack):
    """Extract offsets from ListArray

    Signature: array,!offsets
    """
    stack.append(to_layout(stack.pop()).starts)


def stops(stack):
    """Extract offsets from ListArray

    Signature: array,!stops
    """
    stack.append(to_layout(stack.pop()).stops)


def tags(stack):
    """Extract tags from UnionArray

    Signature: array,!tags
    """
    stack.append(to_layout(stack.pop()).tags)


def content(stack):
    """Extract content from array

    Signature: array,!content
    """
    stack.append(to_layout(stack.pop()).content)


def counts2offsets_form(counts_form):
    form = {
        "class": "NumpyArray",
        "itemsize": 8,
        "format": "i",
        "primitive": "int64",
        "parameters": counts_form.get("parameters", None),
        "form_key": concat(counts_form["form_key"], "!counts2offsets"),
    }
    return form


def counts2offsets(stack):
    """Cumulative sum of counts

    Signature: counts,!counts2offsets
    Outputs an array with length one larger than input
    """
    counts = ensure_array(stack.pop())
    offsets = numpy.empty(len(counts) + 1, dtype=numpy.int64)
    offsets[0] = 0
    numpy.cumsum(counts, out=offsets[1:])
    stack.append(offsets)


def local2global_form(index, target_offsets):
    if not index["class"].startswith("ListOffset"):
        raise RuntimeError
    if not target_offsets["class"] == "NumpyArray":
        raise RuntimeError
    form = copy.deepcopy(index)
    form["content"]["form_key"] = concat(
        index["form_key"], target_offsets["form_key"], "!local2global"
    )
    form["content"]["itemsize"] = 8
    form["content"]["primitive"] = "int64"
    return form


def local2global(stack):
    """Turn jagged local index into global index

    Signature: index,target_offsets,!local2global
    Outputs a content array with same shape as index content
    """
    target_offsets = ensure_array(stack.pop())
    index = stack.pop()
    index = index.mask[index >= 0] + target_offsets[:-1]
    index = index.mask[index < target_offsets[1:]]
    out = ensure_array(awkward.flatten(awkward.fill_none(index, -1), axis=None))
    if out.dtype != numpy.int64:
        raise RuntimeError
    stack.append(out)


@numba.njit
def _grow_local_index_to_target_shape_kernel(index, all_index, builder):
    for i in range(len(all_index)):
        builder.begin_list()
        event_all_index = all_index[i]
        event_index = index[i]
        for all_index_value in event_all_index:
            if all_index_value in event_index:
                builder.integer(all_index_value)
            else:
                builder.integer(-1)
        builder.end_list()

    return builder


def grow_local_index_to_target_shape_form(index, target):
    if not index["class"].startswith("ListOffset"):
        raise RuntimeError
    if not target["class"].startswith("ListOffset"):
        raise RuntimeError
    form = copy.deepcopy(index)
    form["content"]["form_key"] = concat(
        index["content"]["form_key"],
        target["content"][
            "form_key"
        ],  # , "!grow_local_index_to_target_shape", "!content"
    )
    form["form_key"] = concat(
        index["form_key"], target["form_key"], "!grow_local_index_to_target_shape"
    )
    form["content"]["itemsize"] = 8
    form["content"]["primitive"] = "int64"
    return form


def grow_local_index_to_target_shape(stack):
    """Grow the local index to the size of target size by replacing unreferenced indices as -1

    Signature: index,target,!grow_local_index_to_target_shape
    Outputs a content array with same shape as target content
    """
    target = stack.pop()
    index = stack.pop()
    all_index = awkward.local_index(target, axis=1)

    useable_index = awkward.Array(
        _grow_local_index_to_target_shape_kernel(
            awkward.Array(index), awkward.Array(all_index), awkward.ArrayBuilder()
        ).snapshot()
    )

    stack.append(useable_index)


def begin_and_end_to_counts_form(begin, end):
    if not begin["class"].startswith("ListOffset"):
        raise RuntimeError
    if not end["class"].startswith("ListOffset"):
        raise RuntimeError

    form = copy.deepcopy(begin)

    key = concat(begin["form_key"], end["form_key"], "!begin_and_end_to_counts")
    form["content"]["form_key"] = concat(key, "!content")
    return form


def begin_and_end_to_counts(stack):
    "Calculate the number of entries from begin to end (end - begin)"
    end = stack.pop()
    begin = stack.pop()

    stack.append(end - begin)


@numba.njit
def _index_range_kernel(begin_end, target, builder):
    for ev in range(len(begin_end)):
        builder.begin_list()
        for j in range(len(begin_end[ev])):
            builder.begin_list()
            for k in range(begin_end[ev][j][0], begin_end[ev][j][1]):
                # builder.integer(k)
                builder.integer(target[ev][k])
            builder.end_list()
        builder.end_list()
    return builder


def index_range_form(begin, end, target):
    if not begin["class"].startswith("ListOffset"):
        raise RuntimeError
    if not end["class"].startswith("ListOffset"):
        raise RuntimeError
    if not target["class"].startswith("ListOffset"):
        raise RuntimeError
    form = {
        "class": "ListOffsetArray",
        "offsets": "i64",
        "content": {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "itemsize": 8,
                "format": "i",
                "primitive": "int64",
                "form_key": concat(
                    begin["form_key"],
                    end["form_key"],
                    target["form_key"],
                    "!index_range",
                    "!content",
                ),
            },
            "form_key": concat(
                begin["form_key"], end["form_key"], target["form_key"], "!index_range"
            ),
        },
        "form_key": concat(begin["form_key"], end["form_key"], target["form_key"]),
    }
    return form


def index_range(stack):
    """
    Takes in begin and end arrays and a target array.
    This is the process:
        Get ranges (double nesting) of begin to end
        Corresponding to those index ranges, pick up elements from the target array(which are also indices)
        Convert the resulting doubly nested indices into doubly nested global indices (actually flattened across axis=1
        , because unflattening would be done in zip_forms)
    """
    target = stack.pop()
    end = stack.pop()
    begin = stack.pop()
    begin_end = awkward.concatenate(
        (begin[:, :, numpy.newaxis], end[:, :, numpy.newaxis]), axis=2
    )

    out = awkward.Array(
        _index_range_kernel(begin_end, target, awkward.ArrayBuilder()).snapshot()
    )

    # Convert to global index
    counts2 = awkward.flatten(awkward.num(out, axis=2), axis=1)

    out = awkward.flatten(out, axis=2)
    stack2 = [out, begin.layout.offsets.data]
    local2global(stack2)
    out = stack2.pop()

    out = awkward.unflatten(out, counts2, axis=0)

    stack.append(out)


def counts2nestedindex_form(local_counts, target_offsets):
    if not local_counts["class"].startswith("ListOffset"):
        raise RuntimeError
    if not target_offsets["class"] == "NumpyArray":
        raise RuntimeError
    form = {
        "class": "ListOffsetArray",
        "offsets": "i64",
        "content": copy.deepcopy(local_counts),
    }
    form["content"]["content"]["itemsize"] = 8
    form["content"]["content"]["primitive"] = "int64"
    form["content"]["content"]["parameters"] = {}
    key = concat(
        local_counts["form_key"], target_offsets["form_key"], "!counts2nestedindex"
    )
    form["form_key"] = local_counts["form_key"]
    form["content"]["form_key"] = key
    form["content"]["content"]["form_key"] = concat(key, "!content")
    return form


def counts2nestedindex(stack):
    """Turn jagged local counts into doubly-jagged global index into a target

    Signature: local_counts,target_offsets,!counts2nestedindex
    Outputs a jagged array with same axis-0 shape as counts axis-1
    """
    target_offsets = stack.pop()
    local_counts = stack.pop()
    out = awkward.unflatten(
        numpy.arange(target_offsets[-1], dtype=numpy.int64),
        awkward.flatten(local_counts),
    )
    stack.append(out)


@numba.njit
def _distinctParent_kernel(allpart_parent, allpart_pdg):
    out = numpy.empty(len(allpart_pdg), dtype=numpy.int64)
    for i in range(len(allpart_pdg)):
        parent = allpart_parent[i]
        if parent < 0:
            out[i] = -1
            continue
        thispdg = allpart_pdg[i]
        while parent >= 0 and allpart_pdg[parent] == thispdg:
            if parent >= len(allpart_pdg):
                raise RuntimeError("parent index beyond length of array!")
            parent = allpart_parent[parent]
        out[i] = parent
    return out


def distinctParent_form(parents, pdg):
    if not parents["class"].startswith("ListOffset"):
        raise RuntimeError
    if not pdg["class"].startswith("ListOffset"):
        raise RuntimeError
    form = {
        "class": "ListOffsetArray",
        "offsets": "i64",
        "content": {
            "class": "NumpyArray",
            "itemsize": 8,
            "format": "i",
            "primitive": "int64",
        },
        "form_key": parents["form_key"],
    }
    form["content"]["form_key"] = concat(
        parents["content"]["form_key"],
        pdg["content"]["form_key"],
        "!distinctParent",
    )
    return form


def distinctParent(stack):
    """Compute first parent with distinct PDG id

    Signature: globalparents,globalpdgs,!distinctParent
    Expects global indexes, flat arrays, which should be same length
    """
    pdg = stack.pop()
    parents = stack.pop()
    stack.append(_distinctParent_kernel(awkward.Array(parents), awkward.Array(pdg)))


@numba.njit
def _children_kernel(offsets_in, parentidx):
    offsets1_out = numpy.empty(len(parentidx) + 1, dtype=numpy.int64)
    content1_out = numpy.empty(len(parentidx), dtype=numpy.int64)
    offsets1_out[0] = 0

    offset0 = 1
    offset1 = 0
    for record_index in range(len(offsets_in) - 1):
        start_src, stop_src = offsets_in[record_index], offsets_in[record_index + 1]

        for index in range(start_src, stop_src):
            for possible_child in range(index, stop_src):
                if parentidx[possible_child] == index:
                    if offset1 >= len(content1_out):
                        raise RuntimeError("offset1 went out of bounds!")
                    content1_out[offset1] = possible_child
                    offset1 = offset1 + 1
            if offset0 >= len(offsets1_out):
                raise RuntimeError("offset0 went out of bounds!")
            offsets1_out[offset0] = offset1
            offset0 = offset0 + 1

    return offsets1_out, content1_out[:offset1]


def children_form(offsets, globalparents):
    if not globalparents["class"].startswith("ListOffset"):
        raise RuntimeError
    form = {
        "class": "ListOffsetArray",
        "offsets": "i64",
        "content": {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "itemsize": 8,
                "format": "i",
                "primitive": "int64",
            },
        },
    }
    form["form_key"] = offsets["form_key"]
    key = concat(offsets["form_key"], globalparents["content"]["form_key"], "!children")
    form["content"]["form_key"] = key
    form["content"]["content"]["form_key"] = concat(key, "!content")
    return form


def children(stack):
    """Compute children

    Signature: offsets,globalparents,!children
    Output will be a jagged array with same outer shape as globalparents content
    """
    parents = stack.pop()
    offsets = stack.pop()
    coffsets, ccontent = _children_kernel(offsets, parents)
    out = awkward.Array(
        awkward.contents.ListOffsetArray(
            awkward.index.Index64(coffsets),
            awkward.contents.NumpyArray(ccontent),
        )
    )
    stack.append(out)


@numba.njit
def _distinctChildrenDeep_kernel(offsets_in, global_parents, global_pdgs):
    offsets_out = numpy.empty(len(global_parents) + 1, dtype=numpy.int64)
    content_out = numpy.empty(len(global_parents), dtype=numpy.int64)
    offsets_out[0] = 0

    offset0 = 1
    offset1 = 0
    for record_index in range(len(offsets_in) - 1):
        start_src, stop_src = offsets_in[record_index], offsets_in[record_index + 1]

        for index in range(start_src, stop_src):
            this_pdg = global_pdgs[index]

            # only perform the deep lookup when this particle is not already part of a decay chain
            # otherwise, the same child indices would be repeated for every parent in the chain
            # which would require content_out to have a length that isa-priori unknown
            if (
                global_parents[index] >= 0
                and this_pdg != global_pdgs[global_parents[index]]
            ):
                # keep an index of parents with same pdg id
                parents = numpy.empty(stop_src - index, dtype=numpy.int64)
                parents[0] = index
                offset2 = 1

                # keep an additional index with parents that have at least one child
                parents_with_children = numpy.empty(stop_src - index, dtype=numpy.int64)
                offset3 = 0

                for possible_child in range(index, stop_src):
                    possible_parent = global_parents[possible_child]
                    possibe_child_pdg = global_pdgs[possible_child]

                    # compare with seen parents
                    for parent_index in range(offset2):
                        # check if we found a new child
                        if parents[parent_index] == possible_parent:
                            # first, remember that the parent has at least one child
                            if offset3 >= len(parents_with_children):
                                raise RuntimeError("offset3 went out of bounds!")
                            parents_with_children[offset3] = possible_parent
                            offset3 = offset3 + 1

                            # then, depending on the pdg id, add to parents or content
                            if possibe_child_pdg == this_pdg:
                                # has the same pdg id, add to parents
                                if offset2 >= len(parents):
                                    raise RuntimeError("offset2 went out of bounds!")
                                parents[offset2] = possible_child
                                offset2 = offset2 + 1
                            else:
                                # has a different pdg id, add to content
                                if offset1 >= len(content_out):
                                    raise RuntimeError("offset1 went out of bounds!")
                                content_out[offset1] = possible_child
                                offset1 = offset1 + 1
                            break

                # add parents with same pdg id that have no children
                for parent_index in range(1, offset2):
                    possible_child = parents[parent_index]
                    if possible_child not in parents_with_children[:offset3]:
                        if offset1 >= len(content_out):
                            raise RuntimeError("offset1 went out of bounds! pt2")
                        content_out[offset1] = possible_child

                        offset1 = offset1 + 1

            # finish this item by adding an offset
            if offset0 >= len(offsets_out):
                raise RuntimeError("offset0 went out of bounds!")
            offsets_out[offset0] = offset1
            offset0 = offset0 + 1

    return offsets_out, content_out[:offset1]


def distinctChildrenDeep_form(offsets, global_parents, global_pdgs):
    if not global_parents["class"].startswith("ListOffset"):
        raise RuntimeError
    if not global_pdgs["class"].startswith("ListOffset"):
        raise RuntimeError
    form = {
        "class": "ListOffsetArray",
        "offsets": "i64",
        "content": {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "itemsize": 8,
                "format": "i",
                "primitive": "int64",
            },
        },
    }
    form["form_key"] = offsets["form_key"]
    key = concat(
        offsets["form_key"],
        global_parents["content"]["form_key"],
        global_pdgs["content"]["form_key"],
        "!distinctChildrenDeep",
    )
    form["content"]["form_key"] = key
    form["content"]["content"]["form_key"] = concat(key, "!content")
    return form


def distinctChildrenDeep(stack):
    """Compute all distinct children, skipping children with same pdg id in between.

    Signature: offsets,global_parents,global_pdgs,!distinctChildrenDeep
    Expects global indexes, flat arrays, which should be same length
    """
    global_pdgs = stack.pop()
    global_parents = stack.pop()
    offsets = stack.pop()
    coffsets, ccontent = _distinctChildrenDeep_kernel(
        offsets,
        global_parents,
        awkward.Array(global_pdgs),
    )
    out = awkward.Array(
        awkward.contents.ListOffsetArray(
            awkward.index.Index64(coffsets),
            awkward.contents.NumpyArray(ccontent),
        )
    )
    stack.append(out)


def nestedindex_form(indices):
    if not all(index["class"].startswith("ListOffset") for index in indices):
        raise RuntimeError
    form = {
        "class": "ListOffsetArray",
        "offsets": indices[0]["offsets"],
        "content": copy.deepcopy(indices[0]),
    }
    # steal offsets from first input
    key = []
    for index in indices:
        key.append(index["content"]["form_key"])
    key.append("!nestedindex")
    key = concat(*key)
    form["form_key"] = indices[0]["form_key"]
    form["content"]["form_key"] = key
    form["content"]["content"]["form_key"] = concat(key, "!content")
    return form


def nestedindex(stack):
    """Concatenate a list of indices along a new axis

    Signature: index1,index2,...,!nestedindex
    Index arrays should all be same shape flat arrays
    Outputs a jagged array with same outer shape as index arrays
    """
    indexers = stack[:]
    stack.clear()
    # return awkward.concatenate([idx[:, None] for idx in indexers], axis=1)
    n = len(indexers)
    out = numpy.empty(n * len(indexers[0]), dtype="int64")
    for i, idx in enumerate(indexers):
        out[i::n] = idx
    offsets = numpy.arange(0, len(out) + 1, n, dtype=numpy.int64)
    out = awkward.Array(
        awkward.contents.ListOffsetArray(
            awkward.index.Index64(offsets),
            awkward.contents.NumpyArray(out),
        )
    )
    stack.append(out)


def item(stack):
    field = stack.pop()
    array = stack.pop()
    stack.append(array[field])


def eventindex(stack):
    out = stack.pop()
    out, _ = awkward.broadcast_arrays(numpy.arange(len(out), dtype=numpy.int64), out)
    stack.append(out)
