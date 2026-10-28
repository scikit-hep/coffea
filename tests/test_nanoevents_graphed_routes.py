"""Every name a graphed-mode collection can be asked for takes exactly one known route."""

import os
import sys
import types
from inspect import getattr_static

import awkward as ak
import pytest

if sys.version_info < (3, 11):
    pytest.skip("graphed requires Python 3.11 or newer", allow_module_level=True)

graphed = pytest.importorskip("graphed")
pytest.importorskip("graphed.awkward")

from graphed.array import BoundMethod  # noqa: E402
from test_nanoevents_graphed import _both_arms, assert_same  # noqa: E402

from coffea.nanoevents import (  # noqa: E402
    NanoAODSchema,
    NanoEventsFactory,
    PFNanoAODSchema,
)
from coffea.util import _DaskMethod, _DaskProperty  # noqa: E402

#: the routes of the graphed dispatch, in the order they are tried
FIELD, NO_DISPATCH, REFUSED, DASK_ARM, FALLBACK = (
    "field",
    "no_dispatch",
    "refused",
    "dask_arm",
    "fallback",
)


def _behavior_names(schemaclass):
    """Every attribute coffea's own behavior classes add, found by walking the schema's
    behavior dict — descriptors (``no_dispatch`` ones included), plain properties and plain
    methods. Class constants and nested classes are not routed and are left out."""
    routed = (property, _DaskProperty, _DaskMethod, types.FunctionType)
    names = set()
    for value in schemaclass.behavior().values():
        if not isinstance(value, type):
            continue
        for klass in value.__mro__:
            if not klass.__module__.startswith("coffea.nanoevents.methods"):
                continue
            names.update(
                name
                for name, member in vars(klass).items()
                if not name.startswith("__") and isinstance(member, routed)
            )
    return names


def _tracer(array):
    session = array.session
    return session.backend._with_behavior(session.form(array).tt)


def _route(holder, eager_holder, name):
    """Which route ``holder.<name>`` takes, observed; ``None`` when nothing explains it."""
    tracer = _tracer(holder)
    if name in (tracer.fields or []):
        return FIELD
    static = getattr_static(tracer, name, None)
    failure = None
    try:
        got = getattr(holder, name)
    except NotImplementedError:
        return REFUSED
    except Exception as exc:
        got, failure = None, type(exc)
    if failure is not None:
        # an arm ran and its own body refused, exactly as the eager arm refuses
        with pytest.raises(failure):
            getattr(eager_holder, name)
    elif isinstance(got, BoundMethod):
        return FALLBACK
    elif not isinstance(got, graphed.Array):
        return NO_DISPATCH  # ran eagerly on the typetracer, recording nothing
    if isinstance(static, _DaskProperty) and getattr(static, "_dask_get", None):
        return DASK_ARM
    if failure is None:
        return FALLBACK
    return None


@pytest.fixture(scope="module")
def nanoaod(tests_directory):
    return _both_arms(tests_directory, "nano_dy.root", NanoAODSchema)


@pytest.fixture(scope="module")
def pfnano(tests_directory):
    return _both_arms(tests_directory, "pfnano.root", PFNanoAODSchema)


@pytest.fixture(
    params=[(NanoAODSchema, "nanoaod"), (PFNanoAODSchema, "pfnano")],
    ids=["nanoaod", "pfnano"],
)
def admitted(request, nanoaod, pfnano):
    schemaclass, fixture = request.param
    return (schemaclass, *request.getfixturevalue(fixture))


def test_every_name_takes_exactly_one_route(admitted):
    schemaclass, events, eager = admitted
    behaviors = _behavior_names(schemaclass)
    assert behaviors, "the behavior walk found nothing to route"

    holders = [("events", events, eager)]
    holders += [(f, events[f], eager[f]) for f in _tracer(events).fields]

    routes, unclassified = {}, []
    for collection, holder, eager_holder in holders:
        tracer = _tracer(holder)
        fields = set(tracer.fields or [])
        for name in sorted(behaviors | fields):
            if name not in fields and getattr_static(tracer, name, None) is None:
                continue  # this behavior is not attached to this collection
            route = _route(holder, eager_holder, name)
            if route is None:
                unclassified.append(f"{collection}.{name}")
            else:
                routes.setdefault(route, set()).add(name)

    assert unclassified == []
    assert set(routes) == {
        FIELD,
        NO_DISPATCH,
        REFUSED,
        DASK_ARM,
        FALLBACK,
    }
    assert routes[REFUSED] == {"_ensure_systematics", "add_systematic"}
    assert {"_events", "_content", "_collection_name"} <= routes[NO_DISPATCH]
    assert {"matched_jet", "children", "distinctParent"} <= routes[DASK_ARM]
    assert {"delta_r", "metric_table", "isTight", "_apply_global_index"} <= routes[
        FALLBACK
    ]


# ---- one representative per route, asserted bit-for-bit against the eager arm ------------------
def r_field(events):
    return events.Muon.pt


def r_no_dispatch(events):
    return events.Muon._events().Jet.pt


def r_fallback_global_index(events):
    return events.GenJet._apply_global_index(events.Jet.genJetIdxG).pt


def r_dask_arm(events):
    return events.Muon.matched_jet.pt


def r_fallback_method(events):
    return events.Muon.delta_r(events.Muon.matched_jet)


def r_fallback_property(events):
    return events.Jet.isTight


REPRESENTATIVES = {
    FIELD: (r_field, "Muon", "pt"),
    NO_DISPATCH: (r_no_dispatch, "Muon", "_events"),
    "fallback_global_index": (r_fallback_global_index, "GenJet", "_apply_global_index"),
    DASK_ARM: (r_dask_arm, "Muon", "matched_jet"),
    "fallback_method": (r_fallback_method, "Muon", "delta_r"),
    "fallback_property": (r_fallback_property, "Jet", "isTight"),
}


@pytest.mark.parametrize("route", sorted(REPRESENTATIVES))
def test_route_representative_matches_eager(nanoaod, route):
    events, eager = nanoaod
    analysis, collection, name = REPRESENTATIVES[route]
    taken = _route(events[collection], eager[collection], name)
    assert taken == (FALLBACK if route.startswith("fallback") else route)
    assert_same(events.session.materialize(analysis(events)), analysis(eager))


def test_refused_route_representative_raises(nanoaod):
    events, _eager = nanoaod
    with pytest.raises(NotImplementedError):
        events.Muon._ensure_systematics()
    assert _route(events.Muon, _eager.Muon, "_ensure_systematics") == REFUSED


def test_eager_arms_are_untouched(tests_directory):
    """The same names on the eager array still answer with real values."""
    path = os.path.join(tests_directory, "samples", "nano_dy.root")
    eager = NanoEventsFactory.from_root(
        {path: "Events"}, schemaclass=NanoAODSchema, mode="eager"
    ).events()
    assert eager.Muon._collection_name() == "Muon"
    assert ak.all(eager.Muon.matched_jet.pt == eager.Jet[eager.Muon.jetIdx].pt)


# ---- ak.Array's own attributes are not event data ---------------------------------------------
def test_array_metadata_answers_at_record_time_and_records_nothing(nanoaod):
    events, eager = nanoaod
    muons = events.Muon
    before = events.session.node_count()
    assert "pt" in muons.fields
    assert muons.fields == eager.Muon.fields
    assert muons.ndim == eager.Muon.ndim == 2
    with pytest.raises(AttributeError, match="materialize"):
        muons.nbytes
    with pytest.raises(AttributeError, match=r"gak\.mask"):
        muons.mask
    assert events.session.node_count() == before


def test_a_field_named_like_array_metadata_is_still_a_field():
    # no NanoAOD sample has such a field, so the record is built here
    from graphed.awkward import from_awkward

    from coffea.nanoevents._graphed import GraphedNanoBackend

    session = graphed.Session(GraphedNanoBackend())
    records = from_awkward(session, "events", ak.Array([{"ndim": 1}, {"ndim": 3}]))
    assert ak.to_list(session.materialize(records.ndim)) == [1, 3]


def test_array_display_hooks_are_not_dispatched(nanoaod, capsys):
    events, _eager = nanoaod
    for hook in ("_repr_mimebundle_", "_ipython_key_completions_", "_repr"):
        assert getattr(events.Muon, hook, None) is None
    formatters = pytest.importorskip("IPython.core.formatters")
    shown, _metadata = formatters.DisplayFormatter().format(events.Muon)
    assert "text/plain" in shown
    assert capsys.readouterr().err == ""
