# -*- coding: utf-8 -*-
from __future__ import print_function, division

import numpy as np


def fill_lepton_kinematics():
    # histogram creation and manipulation
    from coffea import hist

    electrons_pt = np.array([45.0, 32.0, 51.0, 28.0, 18.0])
    electrons_eta = np.array([0.3, -1.1, 0.2, 2.1, -2.3])

    muons_pt = np.array([25.0, 78.0, 62.0, 53.0, 41.0, 18.0])
    muons_eta = np.array([-0.4, 1.3, -0.3, 0.5, -1.8, 2.4])

    # Two types of axes exist presently: bins and categories
    lepton_kinematics = hist.Hist(
        "Events",
        hist.Cat("flavor", "Lepton flavor"),
        hist.Bin("pt", "$p_{T}$", 19, 10, 100),
        hist.Bin("eta", r"$\eta$", [-2.5, -1.4, 0, 1.4, 2.5]),
    )

    # Pass keyword arguments to fill, all arrays must be flat numpy arrays
    # User is responsible for ensuring all arrays have same jagged structure!
    lepton_kinematics.fill(flavor="electron", pt=electrons_pt, eta=electrons_eta)
    lepton_kinematics.fill(flavor="muon", pt=muons_pt, eta=muons_eta)

    return lepton_kinematics


def test_plot1d():
    # histogram creation and manipulation
    # matplotlib
    import matplotlib.pyplot as plt

    plt.switch_backend("agg")

    from coffea import hist

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

    assert ax.figure is not None
    # Two stacked categories should produce at least two artists
    filled_patches = [patch for patch in ax.patches if patch.get_height() > 0]
    assert len(filled_patches) >= 2


def test_plot2d():
    # histogram creation and manipulation
    from coffea import hist

    # matplotlib
    import matplotlib.pyplot as plt

    plt.switch_backend("agg")

    lepton_kinematics = fill_lepton_kinematics()

    # looking at lepton pt for all eta
    muon_kinematics = lepton_kinematics.integrate("flavor", "muon")

    ax = hist.plot2d(muon_kinematics, "eta")

    assert ax.figure is not None
    assert len(ax.collections) >= 1


def test_plotratio():
    # histogram creation and manipulation
    from coffea import hist

    # matplotlib
    import matplotlib.pyplot as plt

    plt.switch_backend("agg")

    lepton_kinematics = fill_lepton_kinematics()

    # Add some pseudodata to a pt histogram so we can make a nice data/mc plot
    pthist = lepton_kinematics.sum("eta")
    bin_values = pthist.axis("pt").centers()
    poisson_means = pthist.sum("flavor").values()[()]
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


def test_plotgrid():
    # histogram creation and manipulation
    from coffea import hist

    # matplotlib
    import matplotlib.pyplot as plt

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

    assert axs.shape[0] == len(lepton_kinematics.axis("eta").centers())
    assert axs.shape[1] == 1
    for axis_row in axs:
        for axis in axis_row:
            assert axis.figure is not None


def test_clopper_pearson_interval():
    from coffea.hist.plot import clopper_pearson_interval

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
    from coffea.hist.plot import normal_interval

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
