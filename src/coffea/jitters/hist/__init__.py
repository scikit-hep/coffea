"""Histogramming tools

`coffea.hist` is a histogram filling, transformation, and plotting sub-package, utilizing
numpy arrays for storage and matplotlib plotting routines for visualization.

Features found in this package are similar to those found in
packages such as `histbook <https://github.com/scikit-hep/histbook>`__ (deprecated),
`boost-histogram <https://github.com/scikit-hep/boost-histogram>`__ (in development),
`physt <https://github.com/scikit-hep/boost-histogram>`__, and built-in numpy
`histogram <https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html>`__ utilities.

"""

from coffea.jitters.hist.hist_tools import Bin, Cat, Hist, Interval, StringBin
from coffea.jitters.hist.plot import (
    clopper_pearson_interval,
    normal_interval,
    plot1d,
    plot2d,
    plotgrid,
    plotratio,
    poisson_interval,
)

__all__ = [
    "Hist",
    "Bin",
    "Interval",
    "Cat",
    "StringBin",
    "poisson_interval",
    "clopper_pearson_interval",
    "normal_interval",
    "plot1d",
    "plotratio",
    "plot2d",
    "plotgrid",
]
