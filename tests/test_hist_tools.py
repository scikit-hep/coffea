import awkward as ak
import numpy as np
import pytest
from dummy_distributions import dummy_jagged_eta_pt

pytest.importorskip("cupy")

from coffea.jitters import hist


def test_hist():
    counts, test_eta, test_pt = dummy_jagged_eta_pt()

    h_nothing = hist.Hist("empty inside")
    assert h_nothing.sparse_dim() == h_nothing.dense_dim() == 0
    assert h_nothing.values() == {}

    h_regular_bins = hist.Hist(
        "regular joe", hist.Bin("x", "x", 20, 0, 200), hist.Bin("y", "why", 20, -3, 3)
    )
    h_regular_bins.fill(x=test_pt, y=test_eta)
    nentries = np.sum(counts)
    assert h_regular_bins.sum("x", "y", overflow="all").values(sumw2=True)[()] == (
        nentries,
        nentries,
    )
    # bin x=2, y=10 (when overflow removed)
    count_some_bin = np.sum(
        (test_pt >= 20.0) & (test_pt < 30.0) & (test_eta >= 0.0) & (test_eta < 0.3)
    )
    assert (
        h_regular_bins.integrate("x", slice(20, 30)).values()[()][10] == count_some_bin
    )
    assert (
        h_regular_bins.integrate("y", slice(0, 0.3)).values()[()][2] == count_some_bin
    )

    h_reduced = h_regular_bins[10:, -0.6:]
    # bin x=1, y=2
    assert h_reduced.integrate("x", slice(20, 30)).values()[()][2] == count_some_bin
    assert h_reduced.integrate("y", slice(0, 0.3)).values()[()][1] == count_some_bin
    h_reduced.fill(x=23, y=0.1)
    assert h_reduced.integrate("x", slice(20, 30)).values()[()][2] == count_some_bin + 1
    assert h_reduced.integrate("y", slice(0, 0.3)).values()[()][1] == count_some_bin + 1

    animal = hist.Cat("animal", "type of animal")
    vocalization = hist.Cat("vocalization", "onomatopoiea is that how you spell it?")
    h_cat_bins = hist.Hist("I like cats", animal, vocalization)
    h_cat_bins.fill(animal="cat", vocalization="meow", weight=2.0)
    h_cat_bins.fill(
        animal="dog", vocalization="meow", weight=np.array([-1.0, -1.0, -5.0])
    )
    h_cat_bins.fill(animal="dog", vocalization="woof", weight=100.0)
    h_cat_bins.fill(animal="dog", vocalization="ruff")
    assert h_cat_bins.values()[("cat", "meow")] == 2.0
    assert h_cat_bins.values(sumw2=True)[("dog", "meow")] == (-7.0, 27.0)
    assert h_cat_bins.integrate("vocalization", ["woof", "ruff"]).values(sumw2=True)[
        ("dog",)
    ] == (101.0, 10001.0)

    height = hist.Bin("height", "height [m]", 10, 0, 5)
    h_mascots_1 = hist.Hist(
        "fermi mascot showdown",
        animal,
        vocalization,
        height,
        # weight is a reserved keyword
        hist.Bin(
            "mass", "weight (g=9.81m/s**2) [kg]", np.power(10.0, np.arange(5) - 1)
        ),
    )

    h_mascots_2 = hist.Hist(
        "fermi mascot showdown",
        axes=(
            animal,
            vocalization,
            height,
            # weight is a reserved keyword
            hist.Bin(
                "mass", "weight (g=9.81m/s**2) [kg]", np.power(10.0, np.arange(5) - 1)
            ),
        ),
    )

    h_mascots_3 = hist.Hist(
        axes=[
            animal,
            vocalization,
            height,
            # weight is a reserved keyword
            hist.Bin(
                "mass", "weight (g=9.81m/s**2) [kg]", np.power(10.0, np.arange(5) - 1)
            ),
        ],
        label="fermi mascot showdown",
    )

    with pytest.warns(UserWarning):
        h_mascots_4 = hist.Hist(
            "fermi mascot showdown",
            animal,
            vocalization,
            height,
            # weight is a reserved keyword
            hist.Bin(
                "mass", "weight (g=9.81m/s**2) [kg]", np.power(10.0, np.arange(5) - 1)
            ),
            axes=[
                animal,
                vocalization,
                height,
                # weight is a reserved keyword
                hist.Bin(
                    "mass",
                    "weight (g=9.81m/s**2) [kg]",
                    np.power(10.0, np.arange(5) - 1),
                ),
            ],
        )

    assert h_mascots_1._dense_shape == h_mascots_2._dense_shape
    assert h_mascots_2._dense_shape == h_mascots_3._dense_shape
    assert h_mascots_3._dense_shape == h_mascots_4._dense_shape

    assert h_mascots_1._axes == h_mascots_2._axes
    assert h_mascots_2._axes == h_mascots_3._axes
    assert h_mascots_3._axes == h_mascots_4._axes

    adult_bison_h = np.random.normal(loc=2.5, scale=0.2, size=40)
    adult_bison_w = np.random.normal(loc=700, scale=100, size=40)
    h_mascots_1.fill(
        animal="bison", vocalization="huff", height=adult_bison_h, mass=adult_bison_w
    )
    goose_h = np.random.normal(loc=0.4, scale=0.05, size=1000)
    goose_w = np.random.normal(loc=7, scale=1, size=1000)
    h_mascots_1.fill(animal="goose", vocalization="honk", height=goose_h, mass=goose_w)
    crane_h = np.random.normal(loc=1, scale=0.05, size=4)
    crane_w = np.random.normal(loc=10, scale=1, size=4)
    h_mascots_1.fill(animal="crane", vocalization="none", height=crane_h, mass=crane_w)

    with pytest.raises(ValueError):
        h_mascots_1.fill(
            beast="crane", yelling="none", tallness=crane_h, heavitivity=crane_w
        )

    h_mascots_2 = h_mascots_1.copy()
    h_mascots_2.clear()
    baby_bison_h = np.random.normal(loc=0.5, scale=0.1, size=20)
    baby_bison_w = np.random.normal(loc=200, scale=10, size=20)
    baby_bison_cutefactor = 2.5 * np.ones_like(baby_bison_w)
    h_mascots_2.fill(
        animal="bison",
        vocalization="baa",
        height=baby_bison_h,
        mass=baby_bison_w,
        weight=baby_bison_cutefactor,
    )
    h_mascots_2.fill(animal="fox", vocalization="none", height=1.0, mass=30.0)

    h_mascots = h_mascots_1 + h_mascots_2
    assert (
        h_mascots.integrate("vocalization", "h*")
        .sum("height", "mass", "animal")
        .values()[()]
        == 1040.0
    )

    species_class = hist.Cat("species_class", "where the subphylum is vertibrates")
    classes = {
        "birds": ["goose", "crane"],
        "mammals": ["bison", "fox"],
    }
    h_mascots.scale({("goose",): 0.5}, axis=("animal",))
    h_mascots.scale({("goose", "honk"): 2.0}, axis=("animal", "vocalization"))
    h_species = h_mascots.group("animal", species_class, classes)

    assert set(h_species.integrate("vocalization").values().keys()) == {
        ("birds",),
        ("mammals",),
    }
    nbirds_bin = np.sum(
        (goose_h >= 0.5) & (goose_h < 1) & (goose_w > 10) & (goose_w < 100)
    )
    nbirds_bin += np.sum(
        (crane_h >= 0.5) & (crane_h < 1) & (crane_w > 10) & (crane_w < 100)
    )
    assert h_species.integrate("vocalization").values()[("birds",)][1, 2] == nbirds_bin
    tally = h_species.sum("mass", "height", "vocalization").values()
    assert tally[("birds",)] == 1004.0
    assert tally[("mammals",)] == 91.0

    h_species.scale({"honk": 0.1, "huff": 0.9}, axis="vocalization")
    h_species.scale(5.0)
    tally = h_species.sum("mass", height, vocalization).values(sumw2=True)
    assert tally[("birds",)] == (520.0, 350.0)
    assert tally[("mammals",)] == (435.0, 25 * (40 * (0.9**2) + 20 * (2.5**2) + 1))

    assert h_species.axis("vocalization") is vocalization
    assert h_species.axis("height") is height
    assert h_species.integrate("vocalization", "h*").axis("height") is height

    tall_class = hist.Cat("tall_class", "species class (species above 1m)")
    mapping = {
        "birds": (["goose", "crane"], slice(1.0, None)),
        "mammals": (["bison", "fox"], slice(1.0, None)),
    }
    h_tall = h_mascots.group((animal, height), tall_class, mapping)
    tall_bird_count = np.sum(goose_h >= 1.0) + np.sum(crane_h >= 1)
    assert h_tall.sum("mass", "vocalization").values()[("birds",)] == tall_bird_count
    tall_mammal_count = np.sum(adult_bison_h >= 1.0) + np.sum(baby_bison_h >= 1) + 1
    assert (
        h_tall.sum("mass", "vocalization").values()[("mammals",)] == tall_mammal_count
    )

    h_less = h_mascots.remove(["fox", "bison"], axis="animal")
    assert h_less.sum("vocalization", "height", "mass", "animal").values()[()] == 1004.0


def test_hist_serdes():
    import pickle

    h_regular_bins = hist.Hist(
        "regular joe", hist.Bin("x", "x", 20, 0, 200), hist.Bin("y", "why", 20, -3, 3)
    )

    h_regular_bins.fill(
        x=np.array([1.0, 2.0, 3.0, 4.0, 5.0]), y=np.array([-2.0, 1.0, 0.0, 1.0, 2.0])
    )

    h_regular_bins.sum("x").identifiers("y")

    spkl = pickle.dumps(h_regular_bins)

    hnew = pickle.loads(spkl)

    hnew.sum("x").identifiers("y")

    assert h_regular_bins._dense_shape == hnew._dense_shape
    assert h_regular_bins._axes == hnew._axes


def test_hist_serdes_labels():
    import pickle

    ax = hist.Bin("asdf", "asdf", 3, 0, 3)
    ax.identifiers()[0].label = "type 1"
    h = hist.Hist("a", ax)
    h.identifiers("asdf")

    spkl = pickle.dumps(h)

    hnew = pickle.loads(spkl)

    for old, new in zip(h.identifiers("asdf"), hnew.identifiers("asdf")):
        assert old.label == new.label

    assert h._dense_shape == hnew._dense_shape
    assert h._axes == hnew._axes


def test_issue_247():
    from coffea.jitters import hist

    h = hist.Hist("stuff", hist.Bin("old", "old", 20, -1, 1))
    h.fill(old=h.axis("old").centers())
    h2 = h.rebin(h.axis("old"), hist.Bin("new", "new", 10, -1, 1))
    # check first if its even possible to have correct binning
    assert np.all(h2.axis("new").edges() == h.axis("old").edges()[::2])
    # make sure the lookup works properly
    assert np.all(h2.values()[()] == 2.0)
    h3 = h.rebin(h.axis("old"), 2)
    assert np.all(h3.values()[()] == 2.0)

    with pytest.raises(ValueError):
        # invalid division
        _ = h.rebin(h.axis("old"), hist.Bin("new", "new", 8, -1, 1))

    newaxis = hist.Bin("new", "new", h.axis("old").edges()[np.cumsum([0, 2, 3, 5])])
    h.rebin("old", newaxis)


def test_issue_333():
    axis = hist.Bin("channel", "Channel b1", 50, 0, 2000)
    temp = np.arange(0, 2000, 40, dtype=np.int16)
    assert np.all(axis.index(temp).get() == np.arange(50) + 1)


def test_issue_394():
    dummy = hist.Hist(
        "Dummy",
        hist.Cat("sample", "sample"),
        hist.Bin("dummy", "Number of events", 1, 0, 1),
    )
    dummy.fill(sample="test", dummy=1, weight=0.5)


def test_fill_none():
    dummy = hist.Hist("Dummy", hist.Bin("x", "asdf", 1, 0, 1))
    with pytest.raises(ValueError):
        # attempt to fill with none
        dummy.fill(x=ak.Array([0.1, None, 0.3]))

    # allow fill when masked type but no Nones remain
    dummy.fill(x=ak.Array([0.1, None, 0.3])[[True, False, True]])


def test_boost_conversion():
    import boost_histogram as bh

    dummy = hist.Hist(
        "Dummy",
        hist.Cat("sample", "sample"),
        hist.Bin("dummy", "Number of events", 1, 0, 1),
    )
    dummy.fill(sample="test", dummy=1, weight=0.5)
    dummy.fill(sample="test", dummy=0.1)
    dummy.fill(sample="test2", dummy=-0.1)
    dummy.fill(sample="test3", dummy=0.5, weight=0.1)
    dummy.fill(sample="test3", dummy=0.5, weight=0.9)

    h = dummy.to_boost()
    assert len(h.axes) == 2
    assert h[bh.loc("test"), bh.loc(1)].value == 0.5
    assert h[bh.loc("test"), bh.loc(100)].value == 0.5
    assert h[bh.loc("test"), bh.loc(1)].variance == 0.25
    assert h[0, 0].value == 1.0
    assert h[0, 0].variance == 1.0
    assert h[1, 0].value == 0.0
    assert h[bh.loc("test2"), 0].value == 0.0
    assert h[1, bh.underflow].value == 1.0
    assert h[bh.loc("test3"), bh.loc(0.5)].value == 1.0
    assert h[bh.loc("test3"), bh.loc(0.5)].variance == 0.1 * 0.1 + 0.9 * 0.9

    dummy = hist.Hist(
        "Dummy",
        hist.Cat("sample", "sample"),
        hist.Bin("dummy", "Number of events", 1, 0, 1),
    )
    dummy.fill(sample="test", dummy=0.1)
    dummy.fill(sample="test", dummy=0.2)
    dummy.fill(sample="test2", dummy=0.2)
    # No sumw2 -> simple bh storage
    h = dummy.to_boost()
    assert len(h.axes) == 2
    assert h[0, 0] == 2.0
    assert h[1, 0] == 1.0
