import os

import pytest
import requests

from coffea.util import numpy as np

pytest.importorskip("cupy")

url = (
    "https://github.com/scikit-hep/uproot3/blob/master/tests/samples/HZZ.root?raw=true"
)
r = requests.get(url)
with open(os.path.join(os.getcwd(), "HZZ.root"), "wb") as f:
    f.write(r.content)


def fill_lepton_kinematics():
    import awkward as ak
    import uproot

    from coffea.nanoevents.methods import candidate

    ak.behavior.update(candidate.behavior)

    # histogram creation and manipulation
    from coffea.jitters import hist

    fin = uproot.open("HZZ.root")
    tree = fin["events"]

    arrays = {
        k.replace("Electron_", "").strip("P").replace("E", "t").lower(): v
        for k, v in tree.arrays(filter_name="Electron_*", how=dict).items()
    }
    electrons = ak.zip(arrays, with_name="Candidate")

    arrays = {
        k.replace("Muon_", "").strip("P").replace("E", "t").lower(): v
        for k, v in tree.arrays(filter_name="Muon_*", how=dict).items()
    }
    muons = ak.zip(arrays, with_name="Candidate")

    # Two types of axes exist presently: bins and categories
    lepton_kinematics = hist.Hist(
        "Events",
        hist.Cat("flavor", "Lepton flavor"),
        hist.Bin("pt", "$p_{T}$", 19, 10, 100),
        hist.Bin("eta", r"$\eta$", [-2.5, -1.4, 0, 1.4, 2.5]),
    )

    # Pass keyword arguments to fill, all arrays must be flat numpy arrays
    # User is responsible for ensuring all arrays have same jagged structure!
    lepton_kinematics.fill(
        flavor="electron", pt=ak.flatten(electrons.pt), eta=ak.flatten(electrons.eta)
    )
    lepton_kinematics.fill(
        flavor="muon", pt=ak.flatten(muons.pt), eta=ak.flatten(muons.eta)
    )

    return lepton_kinematics


@pytest.mark.mpl_image_compare(style="default", remove_text=True)
def test_plot1d():
    # histogram creation and manipulation
    # matplotlib
    import matplotlib.pyplot as plt

    plt.switch_backend("agg")

    from coffea.jitters import hist

    lepton_kinematics = fill_lepton_kinematics()

    # looking at lepton pt for all eta
    lepton_pt = lepton_kinematics.integrate("eta", overflow="under")

    ax = hist.plot1d(
        lepton_pt,
        overlay="flavor",
        stack=True,
        fill_opts={"alpha": 0.5, "edgecolor": (0, 0, 0, 0.3)},
    )
    # all matplotlib primitives are returned, in case one wants to tweak them
    # e.g. maybe you really miss '90s graphics...

    # Clearly the yields are much different, are the shapes similar?
    lepton_pt.label = "Density"
    hist.plot1d(lepton_pt, overlay="flavor", density=True)

    return ax.figure


@pytest.mark.mpl_image_compare(style="default", remove_text=True)
def test_plot2d():
    # histogram creation and manipulation
    # matplotlib
    import matplotlib.pyplot as plt

    from coffea.jitters import hist

    plt.switch_backend("agg")

    lepton_kinematics = fill_lepton_kinematics()

    # looking at lepton pt for all eta
    muon_kinematics = lepton_kinematics.integrate("flavor", "muon")

    ax = hist.plot2d(muon_kinematics, "eta")

    return ax.figure


def test_plotratio():
    # histogram creation and manipulation
    # matplotlib
    import matplotlib.pyplot as plt

    from coffea.jitters import hist

    plt.switch_backend("agg")

    lepton_kinematics = fill_lepton_kinematics()

    # Add some pseudodata to a pt histogram so we can make a nice data/mc plot
    pthist = lepton_kinematics.sum("eta")
    bin_values = pthist.axis("pt").centers().get()
    poisson_means = pthist.sum("flavor").values()[()].get()
    values = np.repeat(bin_values, np.random.poisson(poisson_means))
    pthist.fill(flavor="pseudodata", pt=values)

    # Set nicer labels, by accessing the string bins' label property
    pthist.axis("flavor").index("electron").label = "e Flavor"
    pthist.axis("flavor").index("muon").label = r"$\mu$ Flavor"
    pthist.axis("flavor").index("pseudodata").label = r"Pseudodata from e/$\mu$"

    # using regular expressions on flavor name to select just the data
    # another method would be to fill a separate data histogram
    import re

    notdata = re.compile("(?!pseudodata)")

    # make a nice ratio plot
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True
    )
    fig.subplots_adjust(hspace=0.07)

    # Here is an example of setting up a color cycler to color the various fill patches
    # http://colorbrewer2.org/#type=qualitative&scheme=Paired&n=6
    from cycler import cycler

    colors = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c"]
    ax.set_prop_cycle(cycler(color=colors))

    fill_opts = {"edgecolor": (0, 0, 0, 0.3), "alpha": 0.8}
    error_opts = {
        "label": "Stat. Unc.",
        "hatch": "///",
        "facecolor": "none",
        "edgecolor": (0, 0, 0, 0.5),
        "linewidth": 0,
    }
    data_err_opts = {
        "linestyle": "none",
        "marker": ".",
        "markersize": 10.0,
        "color": "k",
        "elinewidth": 1,
    }

    hist.plot1d(
        pthist[notdata],
        overlay="flavor",
        ax=ax,
        clear=False,
        stack=True,
        line_opts=None,
        fill_opts=fill_opts,
        error_opts=error_opts,
    )
    hist.plot1d(
        pthist["pseudodata"],
        overlay="flavor",
        ax=ax,
        clear=False,
        error_opts=data_err_opts,
    )

    ax.autoscale(axis="x", tight=True)
    ax.set_ylim(0, None)
    ax.set_xlabel(None)
    ax.legend()

    hist.plotratio(
        pthist["pseudodata"].sum("flavor"),
        pthist[notdata].sum("flavor"),
        ax=rax,
        error_opts=data_err_opts,
        denom_fill_opts={},
        guide_opts={},
        unc="num",
    )
    rax.set_ylabel("Ratio")
    rax.set_ylim(0, 2)

    plt.text(
        0.0,
        1.0,
        "☕",
        fontsize=28,
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )
    plt.text(
        1.0,
        1.0,
        r"1 fb$^{-1}$ (?? TeV)",
        fontsize=16,
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )


@pytest.mark.mpl_image_compare(style="default", remove_text=True)
def test_plotgrid():
    # histogram creation and manipulation
    # matplotlib
    import matplotlib.pyplot as plt

    from coffea.jitters import hist

    plt.switch_backend("agg")

    lepton_kinematics = fill_lepton_kinematics()

    # Let's stack them, after defining some nice styling
    stack_fill_opts = {"alpha": 0.8, "edgecolor": (0, 0, 0, 0.5)}
    stack_error_opts = {
        "label": "Stat. Unc.",
        "hatch": "///",
        "facecolor": "none",
        "edgecolor": (0, 0, 0, 0.5),
        "linewidth": 0,
    }
    # maybe we want to compare different eta regions
    # plotgrid accepts row and column axes, and creates a grid of 1d plots as appropriate
    axs = hist.plotgrid(
        lepton_kinematics,
        row="eta",
        overlay="flavor",
        stack=True,
        fill_opts=stack_fill_opts,
        error_opts=stack_error_opts,
    )

    return axs.flatten()[0].figure


def test_clopper_pearson_interval():
    from coffea.jitters.hist.plot import clopper_pearson_interval

    # Reference values for CL=0.6800 calculated with ROOT's TEfficiency
    num = np.array([1.0, 5.0, 10.0, 10.0])
    denom = np.array([10.0, 10.0, 10.0, 437.0])
    ref_hi = np.array(
        [0.293313782248242, 0.6944224231766912, 1.0, 0.032438865381336446]
    )
    ref_lo = np.array(
        [
            0.01728422272382846,
            0.3055775768233088,
            0.8325532074018731,
            0.015839046981153772,
        ]
    )

    interval = clopper_pearson_interval(num, denom, coverage=0.68)

    threshold = 1e-6
    assert all((interval[1, :] / ref_hi) - 1 < threshold)
    assert all((interval[0, :] / ref_lo) - 1 < threshold)


def test_normal_interval():
    from coffea.jitters.hist.plot import normal_interval

    # Reference weighted efficiency and error from ROOTs TEfficiency

    denom = np.array(
        [
            89.01457591590004,
            2177.066076428943,
            6122.5256890981855,
            0.0,
            100.27757990710668,
        ]
    )
    num = np.array(
        [
            75.14287743709515,
            2177.066076428943,
            5193.454723043864,
            0.0,
            84.97723540536361,
        ]
    )
    denom_sumw2 = np.array(
        [94.37919737476827, 10000.0, 6463.46795877633, 0.0, 105.90898005417333]
    )
    num_sumw2 = np.array(
        [67.2202147680005, 10000.0, 4647.983931785646, 0.0, 76.01275761253757]
    )
    ref_hi = np.array(
        [0.0514643476600107, 0.0, 0.0061403263960343, np.nan, 0.0480731185500146]
    )
    ref_lo = np.array(
        [0.0514643476600107, 0.0, 0.0061403263960343, np.nan, 0.0480731185500146]
    )

    interval = normal_interval(num, denom, num_sumw2, denom_sumw2)
    threshold = 1e-6

    lo, hi = interval

    assert len(ref_hi) == len(hi)
    assert len(ref_lo) == len(lo)

    for i in range(len(ref_hi)):
        if np.isnan(ref_hi[i]):
            assert np.isnan(ref_hi[i])
        elif ref_hi[i] == 0.0:
            assert hi[i] == 0.0
        else:
            assert np.abs(hi[i] / ref_hi[i] - 1) < threshold

        if np.isnan(ref_lo[i]):
            assert np.isnan(ref_lo[i])
        elif ref_lo[i] == 0.0:
            assert lo[i] == 0.0
        else:
            assert np.abs(lo[i] / ref_lo[i] - 1) < threshold
