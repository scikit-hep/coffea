"""Which graphed Session a NanoEvents read records into."""

import os

import pytest
import uproot

graphed = pytest.importorskip("graphed")
pytest.importorskip("graphed.awkward")

from graphed.awkward import AwkwardBackend  # noqa: E402
from test_dataset_tools_graphed import _write_flag_files  # noqa: E402

from coffea.dataset_tools import apply_to_fileset, preprocess  # noqa: E402
from coffea.dataset_tools.filespec import DataGroupSpec  # noqa: E402
from coffea.nanoevents import (  # noqa: E402
    BaseSchema,
    NanoAODSchema,
    NanoEventsFactory,
)
from coffea.nanoevents._graphed import GraphedNanoBackend  # noqa: E402

pytestmark = pytest.mark.skipif(
    not hasattr(graphed, "current_session"), reason="graphed has no current Session"
)


def _root(tests_directory, sample="nano_dy.root", schemaclass=NanoAODSchema, **kw):
    path = os.path.join(tests_directory, "samples", sample)
    return NanoEventsFactory.from_root(
        {path: "Events"}, schemaclass=schemaclass, mode="graphed", **kw
    ).events()


def _parquet(tests_directory, copies=1):
    path = os.path.join(tests_directory, "samples", "nano_dy.parquet")
    return NanoEventsFactory.from_parquet([path] * copies, mode="graphed").events()


@pytest.fixture
def received(monkeypatch):
    """The ``(session=, backend=)`` keywords of each ``uproot.graphed`` call."""
    calls = []
    unpatched = uproot.graphed

    def recorded(*args, **kwargs):
        calls.append((kwargs.get("session"), kwargs.get("backend")))
        return unpatched(*args, **kwargs)

    recorded.__wrapped__ = unpatched  # keeps the signature coffea inspects
    monkeypatch.setattr(uproot, "graphed", recorded)
    return calls


def _names(session):
    return sorted(session.source_name(nid) for nid in session.source_ids())


def test_a_read_records_into_the_default(tests_directory, received):
    events = _root(tests_directory)
    assert events.session is graphed.current_session()
    again = _root(tests_directory, uproot_options={"name": "b"})
    assert again.session is events.session
    assert received == [(events.session, None)] * 2
    assert _names(events.session) == ["Events", "b"]


def test_a_read_records_into_the_open_scope(tests_directory, received):
    scope = graphed.Session(GraphedNanoBackend(behavior=dict(NanoAODSchema.behavior())))
    with scope:
        events = _root(tests_directory)
    assert events.session is scope
    assert received == [(scope, None)]
    assert graphed.set_session(None) is None


def test_without_current_session_a_read_gets_its_own(tests_directory, monkeypatch):
    with monkeypatch.context() as m:
        m.delattr(graphed, "current_session")
        first, second = _root(tests_directory), _root(tests_directory)
    assert first.session is not second.session
    assert graphed.set_session(None) is None


@pytest.mark.parametrize(
    "backend, held",
    [
        (AwkwardBackend(behavior=dict(NanoAODSchema.behavior())), "AwkwardBackend"),
        (GraphedNanoBackend(behavior={}), "GraphedNanoBackend"),
    ],
    ids=["class", "behavior"],
)
def test_an_unusable_default_is_refused_at_the_read(tests_directory, backend, held):
    graphed.new_session(backend)
    with pytest.raises(
        TypeError,
        match=f"the Session records on {held}, and NanoEvents reads record on "
        "GraphedNanoBackend with this schema's behavior",
    ):
        _root(tests_directory)
    assert graphed.current_session().source_ids() == []


def test_another_schemas_default_is_refused(tests_directory):
    _root(tests_directory)
    with pytest.raises(TypeError, match="records on GraphedNanoBackend, and"):
        _root(tests_directory, schemaclass=BaseSchema, uproot_options={"name": "base"})


def test_an_explicit_session_is_refused_as_before_and_misses_the_scope(tests_directory):
    events = _root(tests_directory)
    before = _names(events.session)
    other = graphed.Session(AwkwardBackend())
    with events.session, pytest.raises(TypeError, match="not both"):
        _root(tests_directory, uproot_options={"session": other, "name": "x"})
    assert _names(events.session) == before
    assert other.source_ids() == []


def test_a_parquet_read_records_into_the_scope(tests_directory):
    events = _root(tests_directory, uproot_options={"name": "nano"})
    with events.session:
        parquet = _parquet(tests_directory)
    assert parquet.session is events.session
    with (
        graphed.Session(AwkwardBackend()),
        pytest.raises(TypeError, match="records on AwkwardBackend"),
    ):
        _parquet(tests_directory)


@pytest.mark.skipif(
    not hasattr(uproot._graphed._GraphedTTreeSource, "source_identity"),
    reason="uproot has no source identity",
)
def test_the_same_root_read_twice_reuses_its_node(tests_directory):
    first = _root(tests_directory)
    count = first.session.node_count()
    again = _root(tests_directory)
    assert again.session is first.session
    assert again.node_id == first.node_id
    assert first.session.node_count() == count


def test_the_same_parquet_read_twice_reuses_its_node(tests_directory):
    first = _parquet(tests_directory)
    count = first.session.node_count()
    again = _parquet(tests_directory)
    assert again.session is first.session
    assert again.node_id == first.node_id
    assert first.session.node_count() == count


def test_a_parquet_read_of_other_paths_is_refused(tests_directory):
    _parquet(tests_directory)
    with pytest.raises(ValueError, match="already bound to other data.*new_session"):
        _parquet(tests_directory, copies=2)


def test_each_dataset_records_into_its_own_session(tmp_path):
    has_root, missing_root, _, _ = _write_flag_files(tmp_path)
    seen = {}

    def analysis(events):
        seen[events.metadata["dataset"]] = graphed.current_session()
        return events["x"]

    out = apply_to_fileset(
        analysis,
        {
            "has": {"files": {has_root: "Events"}},
            "missing": {"files": {missing_root: "Events"}},
        },
        schemaclass=BaseSchema,
        mode="graphed",
    )
    has, missing = out["has"], out["missing"]
    assert has.session is not missing.session
    assert seen == {"has": has.session, "missing": missing.session}
    assert graphed.set_session(None) is None


def test_preprocessing_leaves_the_open_scope_unchanged(
    tests_directory, tmp_path, received
):
    """Graphed preprocessing builds its forms without an uproot.graphed read, so an open
    Session gains no sources or nodes."""
    has_root, *_ = _write_flag_files(tmp_path)
    events = _root(tests_directory)
    before, count = _names(events.session), events.session.node_count()
    received.clear()
    with events.session:
        available, _ = preprocess(
            DataGroupSpec({"ds": {"files": {has_root: "Events"}}}),
            save_form=True,
            backend="graphed",
        )
    assert available["ds"].compressed_form is not None
    assert received == []
    assert (_names(events.session), events.session.node_count()) == (before, count)


def test_an_uproot_without_session_keeps_the_backend(tests_directory, monkeypatch):
    unpatched = uproot.graphed

    def released(files, *, backend=None, **kwargs):
        kwargs.pop("session", None)
        session = graphed.Session(backend or AwkwardBackend())
        return unpatched(files, session=session, **kwargs)

    monkeypatch.setattr(uproot, "graphed", released)
    events = _root(tests_directory)
    assert isinstance(events.session.backend, GraphedNanoBackend)


def test_a_backend_in_uproot_options_still_records(tests_directory, received):
    backend = GraphedNanoBackend(behavior=dict(NanoAODSchema.behavior()))
    events = _root(tests_directory, uproot_options={"backend": backend})
    assert events.session is graphed.current_session()
    assert received[0][0] is None


def test_a_dataset_with_an_incapable_schema_is_refused_by_name(tmp_path):
    has_root, *_ = _write_flag_files(tmp_path)
    with pytest.raises(NotImplementedError, match="__graphed_capable__"):
        apply_to_fileset(
            lambda events: events,
            {"has": {"files": {has_root: "Events"}}},
            schemaclass=lambda *a, **k: None,
            mode="graphed",
        )
