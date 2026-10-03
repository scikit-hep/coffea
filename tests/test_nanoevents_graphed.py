import os
import pickle
import subprocess
import sys

import awkward as ak
import pytest

if sys.version_info < (3, 11):
    pytest.skip("graphed requires Python 3.11 or newer", allow_module_level=True)

graphed = pytest.importorskip("graphed")
pytest.importorskip("graphed.awkward")

from graphed.awkward import gak  # noqa: E402
from graphed.core.execution import SequentialRunner  # noqa: E402

from coffea.nanoevents import (  # noqa: E402
    NanoAODSchema,
    NanoEventsFactory,
    PFNanoAODSchema,
)
from coffea.nanoevents.factory import _allowed_modes, _map_schema_uproot  # noqa: E402
from coffea.nanoevents.schemas.nanoaod import ScoutingNanoAODSchema  # noqa: E402


def _equal(left, right):
    """Bit-for-bit equality of two ``ak.to_list`` results, with NaN equal to NaN."""
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(map(_equal, left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _equal(left[key], right[key]) for key in left
        )
    if isinstance(left, float) and isinstance(right, float):
        return left == right or (left != left and right != right)
    return type(left) is type(right) and left == right


def assert_same(actual, expected):
    """The parity oracle: same awkward type, and the same values bit-for-bit."""
    assert str(ak.type(actual)) == str(ak.type(expected))
    assert _equal(ak.to_list(actual), ak.to_list(expected))


# ---- the analyses, spelled once and evaluated against both arms ------------------------------
# Each takes the events array and the array library to use on it (``ak`` eagerly, ``gak``
# deferred), and returns one row-wise output, so partitioning it concatenates.


def a_field(events, xp):
    return events.Muon.pt


def a_plain_property(events, xp):
    return events.Jet.isTight


def a_vector_behavior(events, xp):
    return events.Muon.pvec.rho


def a_vector_sum(events, xp):
    return events.Muon.sum().mass


def a_delta_r(events, xp):
    return events.Muon.delta_r(events.Muon.matched_jet)


def a_metric_table(events, xp):
    return events.Jet.metric_table(events.Muon)


def a_metric_table_combinations(events, xp):
    _metric, (left, _right) = events.Jet.metric_table(
        events.Muon, return_combinations=True
    )
    return left.pt


def a_nearest(events, xp):
    return events.Electron.nearest(events.Jet).pt


def a_matched_jet(events, xp):
    return events.Muon.matched_jet.pt


def a_matched_muons(events, xp):
    return events.Jet.matched_muons.pt


def a_gen_parent(events, xp):
    return events.GenPart.parent.pdgId


def a_gen_children(events, xp):
    return events.GenPart.children.pdgId


def a_gen_distinct_parent(events, xp):
    return events.GenPart.distinctParent.pdgId


def a_tuple_key(events, xp):
    return events.Jet[:, :2].pt


def a_count_only(events, xp):
    return xp.num(events.Jet, axis=1)


NANOAOD_ANALYSES = [
    a_field,
    a_plain_property,
    a_vector_behavior,
    a_vector_sum,
    a_delta_r,
    a_metric_table,
    a_metric_table_combinations,
    a_nearest,
    a_matched_jet,
    a_matched_muons,
    a_gen_parent,
    a_gen_children,
    a_gen_distinct_parent,
    a_tuple_key,
    a_count_only,
]


def a_constituents(events, xp):
    return events.Jet.constituents.pf.pt


PFNANO_ANALYSES = [
    a_field,
    a_constituents,
    a_matched_muons,
    a_delta_r,
    a_nearest,
    a_tuple_key,
    a_count_only,
]


# ---- a plan's aggregate: module level, so the plan pickles and a worker can import it ---------
def reduce_to_lists(values):
    return tuple(ak.to_list(value) for value in values)


def combine_lists(left, right):
    if left is None:
        return right
    if right is None:
        return left
    return tuple(a + b for a, b in zip(left, right))


def no_lists():
    return None


def _factory(path, schemaclass, mode, **kwargs):
    return NanoEventsFactory.from_root(
        {path: "Events"}, schemaclass=schemaclass, mode=mode, **kwargs
    )


def _both_arms(tests_directory, sample, schemaclass):
    path = os.path.join(tests_directory, "samples", sample)
    return (
        _factory(path, schemaclass, "graphed").events(),
        _factory(path, schemaclass, "eager").events(),
    )


@pytest.fixture(scope="module")
def nanoaod(tests_directory):
    return _both_arms(tests_directory, "nano_dy.root", NanoAODSchema)


@pytest.fixture(scope="module")
def pfnano(tests_directory):
    return _both_arms(tests_directory, "pfnano.root", PFNanoAODSchema)


def test_recorded_ops_point_at_the_analysts_line(tests_directory):
    path = os.path.join(tests_directory, "samples", "nano_dy.root")
    events = _factory(path, NanoAODSchema, "graphed").events()
    muons = events.Muon
    recorded = {
        "source": events,
        "field": muons,
        "method": muons.delta_r(muons),
    }
    here = {how: __file__ for how in recorded}
    assert {
        how: events.session.provenance(array).filename
        for how, array in recorded.items()
    } == here


# ---- the factory arm -------------------------------------------------------------------------
def test_graphed_is_an_allowed_mode_and_imported_lazily():
    clean = subprocess.run(
        [
            sys.executable,
            "-E",
            "-c",
            "import sys, coffea.nanoevents;"
            "print([m for m in sys.modules if m.split('.')[0] == 'graphed'])",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert clean.stdout.strip() == "[]"
    assert "graphed" in _allowed_modes


def test_events_is_a_deferred_coffea_graphed_array(nanoaod):
    events, _eager = nanoaod
    assert isinstance(events, graphed.Array)
    assert isinstance(events.session, graphed.Session)
    # coffea installs its own Array subclass through the backend, which is what routes behaviors
    assert type(events) is events.session.backend.array_type()
    assert type(events) is not graphed.Array


def test_recording_reads_no_event_data(tests_directory, monkeypatch):
    reads = []
    unpatched = _map_schema_uproot.load_buffers

    def counted(self, *args, **kwargs):
        reads.append(args[0])
        return unpatched(self, *args, **kwargs)

    monkeypatch.setattr(_map_schema_uproot, "load_buffers", counted)
    path = os.path.join(tests_directory, "samples", "nano_dy.root")
    events = _factory(path, NanoAODSchema, "graphed").events()
    recorded = events.Muon.matched_jet.pt
    assert reads == []
    # the counter is live: the same recorded output does read once it is materialised
    events.session.materialize(recorded)
    assert len(reads) == 1


def test_recorded_form_equals_the_dask_meta_form(tests_directory):
    pytest.importorskip("dask_awkward")
    path = os.path.join(tests_directory, "samples", "nano_dy.root")
    events = _factory(path, NanoAODSchema, "graphed").events()
    meta = _factory(path, NanoAODSchema, "dask").events()._meta
    assert events.session.form(events).tt.layout.form == meta.layout.form


# ---- parity, materialised -------------------------------------------------------------------
@pytest.mark.parametrize("analysis", NANOAOD_ANALYSES, ids=lambda a: a.__name__)
def test_nanoaod_matches_eager(nanoaod, analysis):
    events, eager = nanoaod
    assert_same(events.session.materialize(analysis(events, gak)), analysis(eager, ak))


@pytest.mark.parametrize("analysis", PFNANO_ANALYSES, ids=lambda a: a.__name__)
def test_pfnano_matches_eager(pfnano, analysis):
    events, eager = pfnano
    assert_same(events.session.materialize(analysis(events, gak)), analysis(eager, ak))


# ---- parity, through a plan -------------------------------------------------------------------
@pytest.mark.parametrize("steps_per_file", [1, 2])
@pytest.mark.parametrize(
    "sample, schemaclass, analyses",
    [
        ("nano_dy.root", NanoAODSchema, NANOAOD_ANALYSES),
        ("pfnano.root", PFNanoAODSchema, PFNANO_ANALYSES),
    ],
    ids=["nanoaod", "pfnano"],
)
def test_plan_matches_eager(
    tests_directory, sample, schemaclass, analyses, steps_per_file
):
    events, eager = _both_arms(tests_directory, sample, schemaclass)
    plan = graphed.aggregate_plan(
        *(analysis(events, gak) for analysis in analyses),
        reduce=reduce_to_lists,
        combine=combine_lists,
        empty=no_lists,
        steps_per_file=steps_per_file,
    )
    assert len(plan.tasks) == steps_per_file
    result = SequentialRunner().run(plan)
    expected = reduce_to_lists([analysis(eager, ak) for analysis in analyses])
    for analysis, got, want in zip(analyses, result.value, expected):
        assert _equal(got, want), analysis.__name__


def test_plan_read_list_names_the_counter_branch(nanoaod):
    events, eager = nanoaod
    outputs = (events.Muon.pt, gak.num(events.Jet, axis=1))
    plan = graphed.aggregate_plan(
        *outputs,
        reduce=reduce_to_lists,
        combine=combine_lists,
        empty=no_lists,
    )
    # the count-only output reads no Jet leaf, only the counter that gives it its length
    assert tuple(sorted(plan.process.columns)) == ("Muon_pt", "nJet", "nMuon")
    got = SequentialRunner().run(plan).value
    assert _equal(got, reduce_to_lists([eager.Muon.pt, ak.num(eager.Jet, axis=1)]))


def test_pickled_plan_runs_in_a_fresh_interpreter(nanoaod, tmp_path):
    events, eager = nanoaod
    plan = graphed.aggregate_plan(
        events.Muon.matched_jet.pt,
        reduce=reduce_to_lists,
        combine=combine_lists,
        empty=no_lists,
        steps_per_file=2,
    )
    blob = tmp_path / "plan.pkl"
    blob.write_bytes(pickle.dumps(plan))
    env = dict(os.environ)
    # this module holds the plan's reduce/combine/empty, so the worker must be able to import it
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [os.path.dirname(__file__), env.get("PYTHONPATH")])
    )
    worker = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pickle, sys;"
            "from graphed.core.execution import SequentialRunner;"
            "print(SequentialRunner().run("
            "pickle.load(open(sys.argv[1], 'rb'))).value[0])",
            str(blob),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert worker.returncode == 0, worker.stderr
    assert _equal(
        eval(worker.stdout), ak.to_list(eager.Muon.matched_jet.pt)  # noqa: S307
    )


# ---- refusals ---------------------------------------------------------------------------------
def test_refuses_arguments_that_belong_to_the_plan(tests_directory):
    path = os.path.join(tests_directory, "samples", "nano_dy.root")
    with pytest.raises(NotImplementedError, match=r"aggregate_plan\(.*steps_per_file"):
        _factory(path, NanoAODSchema, "graphed", steps_per_file=2)
    with pytest.raises(
        NotImplementedError, match=r"graphed\.checkpoint\.run_resumable"
    ):
        _factory(
            path,
            NanoAODSchema,
            "graphed",
            uproot_options={"allow_read_errors_with_report": True},
        )


def test_admits_the_report_option_switched_off(tests_directory):
    path = os.path.join(tests_directory, "samples", "nano_dy.root")
    factory = _factory(
        path,
        NanoAODSchema,
        "graphed",
        uproot_options={"allow_read_errors_with_report": False},
    )
    assert isinstance(factory.events(), graphed.Array)


def test_refuses_user_callables_and_coffea_systematics(nanoaod):
    events, _eager = nanoaod
    with pytest.raises(NotImplementedError, match=r"callable"):
        events.Jet.metric_table(events.Muon, metric=lambda a, b: a.delta_r(b))
    with pytest.raises(NotImplementedError, match=r"graphed\.vary"):
        events.Muon.add_systematic(
            "pt_scale", "UpDownSystematic", "pt", lambda x: x * 1.01
        )
    with pytest.raises(NotImplementedError, match=r"graphed\.vary"):
        events.Muon._ensure_systematics()


def _bare_graphed_array():
    """A deferred graphed array owing nothing to coffea, for the two analysis_tools guards."""
    from graphed.awkward import AwkwardBackend, from_awkward

    session = graphed.Session(AwkwardBackend())
    return from_awkward(session, "cut", ak.Array([True, False, True]))


def test_refuses_map_partitions_on_a_graphed_array():
    from coffea.util import maybe_map_partitions

    with pytest.raises(NotImplementedError):
        maybe_map_partitions(lambda x: x, _bare_graphed_array())


def test_refuses_parquet_input(tests_directory):
    path = os.path.join(tests_directory, "samples", "nano_dy.parquet")
    with pytest.raises(NotImplementedError):
        NanoEventsFactory.from_parquet(path, mode="graphed")


class _UserNanoAODSchema(NanoAODSchema):
    """A user subclass: it inherits the flag but does not declare it, so it is not admitted."""


def _auto_schema(
    base_form,
):  # pragma: no cover - never reached, the mode refuses it first
    return NanoAODSchema(base_form)


@pytest.mark.parametrize(
    "schemaclass",
    [ScoutingNanoAODSchema, _UserNanoAODSchema, "auto", _auto_schema],
    ids=["scouting", "user_subclass", "auto_string", "function"],
)
def test_refuses_schemas_that_do_not_declare_themselves(tests_directory, schemaclass):
    path = os.path.join(tests_directory, "samples", "nano_dy.root")
    with pytest.raises(NotImplementedError, match="__graphed_capable__"):
        _factory(path, schemaclass, "graphed")


@pytest.mark.parametrize(
    "schemaclass", [NanoAODSchema, PFNanoAODSchema], ids=["nanoaod", "pfnano"]
)
def test_admitted_schemas_declare_the_flag_in_their_own_body(schemaclass):
    assert vars(schemaclass).get("__graphed_capable__") is True


# ---- controls: nothing that worked before changes ---------------------------------------------
@pytest.mark.parametrize("mode", ["eager", "virtual", "dask"])
def test_other_modes_are_unchanged(tests_directory, mode):
    if mode == "dask":
        pytest.importorskip("dask_awkward")
    path = os.path.join(tests_directory, "samples", "nano_dy.root")
    events = _factory(path, NanoAODSchema, mode).events()
    eager = _factory(path, NanoAODSchema, "eager").events()
    meta = events._meta if mode == "dask" else events
    assert meta.layout.form == eager.layout.form
    pt = events.Muon.pt
    assert_same(pt.compute() if mode == "dask" else pt, eager.Muon.pt)


def test_dask_cross_references_are_unchanged(tests_directory):
    pytest.importorskip("dask_awkward")
    path = os.path.join(tests_directory, "samples", "nano_dy.root")
    events = _factory(path, NanoAODSchema, "dask").events()
    eager = _factory(path, NanoAODSchema, "eager").events()
    assert_same(events.Muon.matched_jet.pt.compute(), eager.Muon.matched_jet.pt)


def test_packed_selection_still_refuses_a_deferred_graphed_mask():
    """Admitting one here would turn ``add`` into a silent no-op, so it must keep refusing."""
    from coffea.analysis_tools import PackedSelection

    with pytest.raises(ValueError):
        PackedSelection().add("cut", _bare_graphed_array())
