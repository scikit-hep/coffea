"""The ``graphed`` arm of :class:`coffea.nanoevents.NanoEventsFactory`.

``mode="graphed"`` records an analysis into a ``graphed`` graph instead of materialising it, through
the :class:`GraphedNanoArray` subclass installed here. Cross-references come out as ordinary graph edges with both collections as operands, so a worker
needs nothing but the recorded graph to replay them.
"""

from inspect import getattr_static

import awkward
import graphed
import uproot
from graphed.array import BoundMethod
from graphed.awkward import AwkwardBackend
from graphed.parquet import make_partitions
from graphed.provenance import register_internal

from coffea.util import _DaskProperty, _meta

# a recorded op's provenance is the analyst's line, not the coffea frame that recorded it
register_internal("coffea")

#: names whose deferred meaning is a systematics axis, which graphed spells its own way
_REFUSED = frozenset({"_ensure_systematics", "add_systematic"})

_VARY_POINTER = "graphed mode does not carry coffea systematics; build the variations with graphed.vary(...)"


class _GraphedBoundMethod(BoundMethod):
    """A behavior method call on a graphed array, recorded as a ``method`` op."""

    __slots__ = ("_arm",)

    def __init__(self, receiver, name, arm=None):
        super().__init__(receiver, name)
        self._arm = arm

    def __call__(self, *args, **kwargs):
        if any(callable(arg) for arg in (*args, *kwargs.values())):
            if self._arm is not None:
                # the method's dask arm builds its result from deferred ops, so a callable runs at record time
                return self._arm(*args, **kwargs)
            raise NotImplementedError(
                f"{self._name}() was passed a callable, and graphed mode has no way to record "
                f"one; if it varies event data, spell the variation with graphed.vary(...)"
            )
        return super().__call__(*args, **kwargs)


class _BranchDoc:
    """``__doc__``: the array's branch doc, as ``ak.Array`` sets it, else the class docstring."""

    def __init__(self, doc):
        self.doc = doc

    def __get__(self, obj, cls=None):
        if obj is None:
            return self.doc
        doc = _meta(obj).layout.purelist_parameter("__doc__")
        return doc if isinstance(doc, str) else self.doc


class GraphedNanoArray(graphed.awkward.AwkwardArray):
    """The deferred NanoEvents surface, mirroring ``dask_awkward.Array.__getattr__``."""

    __slots__ = ()
    __doc__ = _BranchDoc(__doc__)

    def __getattr__(self, name):
        # order matters: a record field shadows a behavior of the same name
        if name.startswith("__"):
            raise AttributeError(name)
        session = self.session
        form = session.form(self)
        if name in form.tt.fields:
            return session.record_op("field", [self], {"field": name})
        if name in _REFUSED:
            raise NotImplementedError(_VARY_POINTER)
        tracer = session.backend._with_behavior(form.tt)
        static = getattr_static(tracer, name, None)
        if getattr(static, "_no_dispatch", False) or isinstance(static, _DaskProperty):
            return static._dask_get(tracer, type(tracer), self)
        kind = session.backend.attribute_kind(form, name)
        if name.startswith("_") and (
            kind != "method" or static is getattr_static(awkward.Array, name, None)
        ):
            # ak.Array internals (_layout, ...) and display hooks (_repr_mimebundle_) are no data
            raise AttributeError(name)
        if kind == "method":
            arm = getattr(static, "_dask_get", None)
            return _GraphedBoundMethod(
                self, name, arm and arm(tracer, type(tracer), self)
            )
        if kind == "introspection":
            return session.backend.introspect(form, name)
        return session.record_op("field", [self], {"field": name})


class GraphedNanoBackend(AwkwardBackend):
    """An ``AwkwardBackend`` whose sessions hand back :class:`GraphedNanoArray`."""

    def array_type(self):
        return GraphedNanoArray


def check_from_root(
    schemaclass, steps_per_file, uproot_options, entry_start=None, entry_stop=None
):
    """Refuse the ``from_root`` arguments and schemas graphed mode cannot honour, and return the
    uproot options to forward."""
    if entry_start is not None or entry_stop is not None:
        raise NotImplementedError(
            "graphed mode does not take entry_start/entry_stop; bound the entries in the files "
            'spec, {path: {"object_path": tree, "steps": [[start, stop]]}}'
        )
    if steps_per_file is not uproot._util.unset:
        raise NotImplementedError(
            "graphed mode does not partition files here; pass steps_per_file where the plan is "
            "built, graphed.aggregate_plan(..., steps_per_file=...)"
        )
    if uproot_options.get("allow_read_errors_with_report"):
        raise NotImplementedError(
            "graphed mode has no report tuple; failed partitions are dead-lettered and retried "
            "by graphed.checkpoint.run_resumable"
        )
    check_schema(schemaclass)
    # uproot.graphed refuses the report key by presence, even switched off
    return {
        key: value
        for key, value in uproot_options.items()
        if key != "allow_read_errors_with_report"
    }


def check_schema(schemaclass):
    """Refuse a schema that does not declare graphed mode in its own class body."""
    if not isinstance(schemaclass, type) or not vars(schemaclass).get(
        "__graphed_capable__"
    ):
        raise NotImplementedError(
            f"graphed mode needs a schema declaring __graphed_capable__ = True in its own class "
            f"body, which {schemaclass!r} does not"
        )


class GraphedParquetSource:
    """A ``graphed.write.PartitionedSource`` over parquet files, each partition read into virtual
    NanoEvents by :meth:`NanoEventsFactory.from_parquet`, so only the columns evaluation touches are read.
    """

    def __init__(self, paths, options):
        self._paths = paths
        self._options = options

    def factory(self, path, entry_start=None, entry_stop=None):
        from coffea.nanoevents import NanoEventsFactory

        return NanoEventsFactory.from_parquet(
            path,
            mode="virtual",
            entry_start=entry_start,
            entry_stop=entry_stop,
            **self._options,
        )

    def __call__(self):
        parts = [self.factory(path).events() for path in self._paths]
        return parts[0] if len(parts) == 1 else awkward.concatenate(parts)

    def partitions(self, steps_per_file=1):
        return make_partitions(
            [str(path) for path in self._paths],
            steps_per_file=steps_per_file,
            open_files=False,
        )

    def read_partition(self, partition, columns, resources):
        partition = partition.resolve(len(self.factory(partition.uri)))
        return self.factory(
            partition.uri, partition.entry_start, partition.entry_stop
        ).events()


def parquet_opener(paths, options):
    """The ``mapping`` a graphed :class:`NanoEventsFactory` calls to record a parquet source."""

    def open_source(form_mapping, backend):
        source = GraphedParquetSource(paths, options)
        form = source.factory(paths[0], 0, 0)._schema.form  # reads metadata only
        typetracer = awkward.typetracer.typetracer_from_form(
            form, highlevel=True, behavior=form_mapping.behavior
        )
        return graphed.Session(backend).source(
            "Events", form=graphed.awkward.AwkwardForm(typetracer), data=source
        )

    return open_source
