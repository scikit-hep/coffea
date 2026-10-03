"""The admitting side of each guard graphed mode adds: what it lets through, not what it refuses.

The refusals live in ``test_nanoevents_graphed.py``; a guard asserted only on its refusing side
cannot be told apart from a wider or a narrower one.
"""

import os
import pickle
import subprocess
import sys

import awkward as ak
import numpy as np
import pytest
import uproot

if sys.version_info < (3, 11):
    pytest.skip("graphed requires Python 3.11 or newer", allow_module_level=True)

graphed = pytest.importorskip("graphed")
pytest.importorskip("graphed.awkward")

from graphed.awkward import from_awkward  # noqa: E402

from coffea.nanoevents import (  # noqa: E402
    BaseSchema,
    NanoAODSchema,
    NanoEventsFactory,
)
from coffea.nanoevents._graphed import GraphedNanoBackend  # noqa: E402
from coffea.nanoevents.factory import _map_schema_uproot  # noqa: E402
from coffea.util import (  # noqa: E402
    _is_interpretable,
    dask_property,
    maybe_map_partitions,
)


class _GraphedBaseSchema(BaseSchema):
    """Declares the flag, so files that no shipping graphed-capable schema covers can be
    recorded here."""

    __graphed_capable__ = True


def _sample(tests_directory, name):
    return os.path.join(tests_directory, "samples", name)


def _record(path, treepath="Events", schemaclass=NanoAODSchema, **kwargs):
    return NanoEventsFactory.from_root(
        {path: treepath}, schemaclass=schemaclass, mode="graphed", **kwargs
    ).events()


def _fields(events):
    return events.session.form(events).tt.fields


@pytest.fixture(scope="module")
def nanoaod(tests_directory):
    return _record(_sample(tests_directory, "nano_dy.root"))


# ---- coffea.util.maybe_map_partitions ---------------------------------------------------------
def test_a_dask_argument_still_maps_where_graphed_is_imported():
    dask_awkward = pytest.importorskip("dask_awkward")
    assert (
        "graphed" in sys.modules
    ), "the guard is vacuous unless graphed is loaded here"
    deferred = dask_awkward.from_awkward(ak.Array([[1.0, 2.0], [3.0]]), 1)
    assert isinstance(
        maybe_map_partitions(lambda a: a * 2, deferred), dask_awkward.Array
    )


#: run with graphed made unimportable, so an arm that imports it eagerly cannot survive
_GRAPHED_ABSENT = """
import sys


class _Blocked:
    def find_spec(self, name, path=None, target=None):
        if name == "graphed" or name.startswith("graphed."):
            raise ModuleNotFoundError("No module named 'graphed'", name=name)
        return None


sys.meta_path.insert(0, _Blocked())
{body}
"""

_MAP_PARTITIONS_WITHOUT_GRAPHED = _GRAPHED_ABSENT.format(body="""
import numpy as np

from coffea.util import maybe_map_partitions

doubled = maybe_map_partitions(lambda a: a * 2, np.array([1.0, 2.0]))
assert list(doubled) == [2.0, 4.0], doubled
assert "graphed" not in sys.modules
print("ok")
""")


def _run(script, *args):
    """``script`` in a fresh interpreter, inheriting this one's import path."""
    return subprocess.run(
        [sys.executable, "-c", script, *args], capture_output=True, text=True
    )


def test_maybe_map_partitions_works_with_graphed_absent():
    done = _run(_MAP_PARTITIONS_WITHOUT_GRAPHED)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


# ---- GraphedNanoArray.__getattr__ -------------------------------------------------------------
def test_protocol_probing_does_not_reach_the_dispatcher(nanoaod):
    """``np.asarray`` asks for ``__array__``; answering it with a recorded op would record an
    operation the caller never wrote."""
    pt = nanoaod.Muon.pt
    assert np.asarray(pt).item() is pt


def test_the_recorded_array_carries_no_instance_dict(nanoaod):
    """graphed.Array is slotted so that interning can rely on its identity."""
    assert not hasattr(nanoaod.Muon.pt, "__dict__")


class _Colliding(ak.Array):
    """A behavior whose descriptor name collides with a record field."""

    @dask_property
    def x(self):
        return "the eager descriptor"

    @x.dask
    def x(self, deferred):
        return "the deferred descriptor"


def test_a_record_field_shadows_a_colliding_descriptor():
    """Field first, as ``dask_awkward.Array.__getattr__`` resolves it — and unlike eager
    awkward, which answers with the behavior."""
    dask_awkward = pytest.importorskip("dask_awkward")
    behavior = {("*", "colliding"): _Colliding}
    array = ak.with_parameter(
        ak.Array([{"x": 1.0}, {"x": 2.0}]), "__record__", "colliding", behavior=behavior
    )
    assert array.x == "the eager descriptor"

    session = graphed.Session(GraphedNanoBackend(behavior=behavior))
    recorded = from_awkward(session, "source", array).x
    oracle = dask_awkward.from_awkward(array, 1).x
    assert ak.to_list(session.materialize(recorded)) == ak.to_list(oracle.compute())


def test_a_positional_callable_is_refused_too(nanoaod):
    with pytest.raises(NotImplementedError, match=r"callable"):
        nanoaod.Electron.nearest(nanoaod.Jet, 1, lambda a, b: a.delta_r(b))


# ---- what the from_root arm hands to uproot ---------------------------------------------------
def test_known_base_form_is_forwarded_and_opens_no_file(tests_directory, monkeypatch):
    path = _sample(tests_directory, "nano_dy.root")
    base_form = uproot.dask(
        {path: "Events"},
        open_files=False,
        full_paths=True,
        ak_add_doc={"__doc__": "title", "typename": "typename"},
        filter_branch=_is_interpretable,
    )._meta.layout.form

    opened = []
    unpatched = uproot.reading.ReadOnlyFile.__init__

    def counted(self, file_path, **kwargs):
        opened.append(file_path)
        return unpatched(self, file_path, **kwargs)

    monkeypatch.setattr(uproot.reading.ReadOnlyFile, "__init__", counted)
    given = _record(path, known_base_form=base_form)
    assert opened == []
    # the counter is live, and the form uproot was handed is the one it would have read
    derived = _record(path)
    assert len(opened) == 1
    assert _fields(given) == _fields(derived)


def test_full_paths_is_forwarded(tests_directory):
    """delphes names a branch ``Area`` under several trees; only ``full_paths`` keeps them
    apart."""
    events = _record(
        _sample(tests_directory, "delphes.root"),
        treepath="Delphes",
        schemaclass=_GraphedBaseSchema,
    )
    assert "GenJet/GenJet.Area" in _fields(events)


def test_filter_branch_is_forwarded(tests_directory):
    """``AnalysisElectrons`` groups sub-branches uproot cannot interpret; its Aux sibling, which
    the same filter keeps, is the control."""
    fields = _fields(
        _record(
            _sample(tests_directory, "PHYSLITE_example.root"),
            treepath="CollectionTree",
            schemaclass=_GraphedBaseSchema,
        )
    )
    assert "AnalysisElectrons" not in fields
    assert "AnalysisElectronsAux." in fields


def test_an_open_directory_is_resolved_through_treepath(tests_directory):
    path = _sample(tests_directory, "nano_dy.root")
    events = NanoEventsFactory.from_root(
        uproot.open(path), treepath="Events", schemaclass=NanoAODSchema, mode="graphed"
    ).events()
    assert _fields(events) == _fields(_record(path))


@pytest.mark.parametrize("mode", ["graphed", "dask"])
def test_an_open_directory_needs_a_treepath(tests_directory, mode):
    if mode == "dask":
        pytest.importorskip("dask_awkward")
    directory = uproot.open(_sample(tests_directory, "nano_dy.root"))
    with pytest.raises(ValueError, match="treepath"):
        NanoEventsFactory.from_root(directory, schemaclass=NanoAODSchema, mode=mode)


def test_the_dask_arm_still_resolves_an_open_directory(tests_directory):
    """The hoisted resolver serves the dask arm too. Only the form is asserted: computing one
    of these arrays already fails on coffea's own release, ahead of anything graphed mode adds.
    """
    pytest.importorskip("dask_awkward")
    path = _sample(tests_directory, "nano_dy.root")
    events = NanoEventsFactory.from_root(
        uproot.open(path), treepath="Events", schemaclass=NanoAODSchema, mode="dask"
    ).events()
    eager = NanoEventsFactory.from_root(
        {path: "Events"}, schemaclass=NanoAODSchema, mode="eager"
    ).events()
    assert events._meta.layout.form == eager.layout.form


_MISSING_GRAPHED = _GRAPHED_ABSENT.format(body="""
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

try:
    NanoEventsFactory.from_root(
        {sys.argv[1]: "Events"}, schemaclass=NanoAODSchema, mode="graphed"
    )
except ModuleNotFoundError as err:
    print(err)
else:
    raise AssertionError("graphed mode did not notice the missing install")
""")


def test_missing_graphed_names_the_install(tests_directory):
    done = _run(_MISSING_GRAPHED, _sample(tests_directory, "nano_dy.root"))
    assert done.returncode == 0, done.stderr
    assert "pip install coffea[graphed]" in done.stdout


# ---- the mapping the plan pickles -------------------------------------------------------------
def test_the_uproot_mapping_pickles_its_base_form_extras():
    """The podio shape: file-level extras the form cannot be rebuilt without."""
    extras = {"podio_collection_types": {"ReconstructedParticles": "edm4hep::Cluster"}}
    mapping = _map_schema_uproot(
        schemaclass=NanoAODSchema,
        behavior=dict(NanoAODSchema.behavior()),
        metadata={"dataset": "dy"},
        version="latest",
        base_form_extras=extras,
    )
    restored = pickle.loads(pickle.dumps(mapping))
    assert restored.schemaclass is NanoAODSchema
    assert restored.metadata == {"dataset": "dy"}
    assert restored.version == "latest"
    assert restored.base_form_extras == extras
    assert restored.behavior.keys() == mapping.behavior.keys()
