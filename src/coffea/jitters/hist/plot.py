import numbers
import warnings

import numpy
import scipy.stats

from .hist_tools import DenseAxis, Interval, SparseAxis, overflow_behavior

# Plotting is always terrible
# Let's try our best to follow matplotlib idioms
# https://matplotlib.org/tutorials/introductory/usage.html#coding-styles

_coverage1sd = scipy.stats.norm.cdf(1) - scipy.stats.norm.cdf(-1)


def poisson_interval(sumw, sumw2, coverage=_coverage1sd):
    """Frequentist coverage interval for Poisson-distributed observations

    Parameters
    ----------
        sumw : numpy.ndarray
            Sum of weights vector
        sumw2 : numpy.ndarray
            Sum weights squared vector
        coverage : float, optional
            Central coverage interval, defaults to 68%

    Calculates the so-called 'Garwood' interval,
    c.f. https://www.ine.pt/revstat/pdf/rs120203.pdf or
    http://ms.mcmaster.ca/peter/s743/poissonalpha.html
    For weighted data, this approximates the observed count by ``sumw**2/sumw2``, which
    effectively scales the unweighted poisson interval by the average weight.
    This may not be the optimal solution: see https://arxiv.org/pdf/1309.1287.pdf for a proper treatment.
    When a bin is zero, the scale of the nearest nonzero bin is substituted to scale the nominal upper bound.
    If all bins zero, a warning is generated and interval is set to ``sumw``.
    """
    scale = numpy.empty_like(sumw)
    scale[sumw != 0] = sumw2[sumw != 0] / sumw[sumw != 0]
    if numpy.sum(sumw == 0) > 0:
        missing = numpy.where(sumw == 0)
        available = numpy.nonzero(sumw)
        if len(available[0]) == 0:
            warnings.warn(
                "All sumw are zero!  Cannot compute meaningful error bars",
                RuntimeWarning,
            )
            return numpy.vstack([sumw, sumw])
        nearest = sum(
            [numpy.subtract.outer(d, d0) ** 2 for d, d0 in zip(available, missing)]
        ).argmin(axis=0)
        argnearest = tuple(dim[nearest] for dim in available)
        scale[missing] = scale[argnearest]
    counts = sumw / scale
    lo = scale * scipy.stats.chi2.ppf((1 - coverage) / 2, 2 * counts) / 2.0
    hi = scale * scipy.stats.chi2.ppf((1 + coverage) / 2, 2 * (counts + 1)) / 2.0
    interval = numpy.array([lo, hi])
    interval[interval == numpy.nan] = 0.0  # chi2.ppf produces nan for counts=0
    return interval


def clopper_pearson_interval(num, denom, coverage=_coverage1sd):
    """Compute Clopper-Pearson coverage interval for a binomial distribution

    Parameters
    ----------
        num : numpy.ndarray
            Numerator, or number of successes, vectorized
        denom : numpy.ndarray
            Denominator or number of trials, vectorized
        coverage : float, optional
            Central coverage interval, defaults to 68%

    c.f. http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    """
    if numpy.any(num > denom):
        raise ValueError(
            "Found numerator larger than denominator while calculating binomial uncertainty"
        )
    lo = scipy.stats.beta.ppf((1 - coverage) / 2, num, denom - num + 1)
    hi = scipy.stats.beta.ppf((1 + coverage) / 2, num + 1, denom - num)
    interval = numpy.array([lo, hi])
    interval[:, num == 0.0] = 0.0
    interval[1, num == denom] = 1.0
    return interval


def normal_interval(pw, tw, pw2, tw2, coverage=_coverage1sd):
    """Compute errors based on the expansion of pass/(pass + fail), possibly weighted

    Parameters
    ----------
    pw : numpy.ndarray
        Numerator, or number of (weighted) successes, vectorized
    tw : numpy.ndarray
        Denominator or number of (weighted) trials, vectorized
    pw2 : numpy.ndarray
        Numerator sum of weights squared, vectorized
    tw2 : numpy.ndarray
        Denominator sum of weights squared, vectorized
    coverage : float, optional
        Central coverage interval, defaults to 68%

    c.f. https://root.cern.ch/doc/master/TEfficiency_8cxx_source.html#l02515
    """

    eff = pw / tw

    variance = (pw2 * (1 - 2 * eff) + tw2 * eff**2) / (tw**2)
    sigma = numpy.sqrt(variance)

    prob = 0.5 * (1 - coverage)
    delta = numpy.zeros_like(sigma)
    delta[sigma != 0] = scipy.stats.norm.ppf(prob, scale=sigma[sigma != 0])

    lo = eff - numpy.minimum(eff + delta, numpy.ones_like(eff))
    hi = numpy.maximum(eff - delta, numpy.zeros_like(eff)) - eff

    return numpy.array([lo, hi])


def plot1d(
    hist,
    ax=None,
    clear=True,
    overlay=None,
    stack=False,
    overflow="none",
    line_opts=None,
    fill_opts=None,
    error_opts=None,
    legend_opts={},
    overlay_overflow="none",
    density=False,
    binwnorm=None,
    order=None,
):
    """Create a 1D plot from a 1D or 2D `Hist` object

    Parameters
    ----------
        hist : Hist
            Histogram with maximum of two dimensions
        ax : matplotlib.axes.Axes, optional
            Axes object (if None, one is created)
        clear : bool, optional
            Whether to clear Axes before drawing (if passed); if False, this function will skip drawing the legend
        overlay : str, optional
            In the case that ``hist`` is 2D, specify the axis of hist to overlay (remaining axis will be x axis)
        stack : bool, optional
            Whether to stack or overlay non-axis dimension (if it exists)
        order : list, optional
            How to order when stacking. Take a list of identifiers.
        overflow : str, optional
            If overflow behavior is not 'none', extra bins will be drawn on either end of the nominal
            axis range, to represent the contents of the overflow bins.  See `Hist.sum` documentation
            for a description of the options.
        line_opts : dict, optional
            A dictionary of options to pass to the matplotlib
            `ax.step <https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.step.html>`_ call
            internal to this function.  Leave blank for defaults.
        fill_opts : dict, optional
            A dictionary of options to pass to the matplotlib
            `ax.fill_between <https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.fill_between.html>`_ call
            internal to this function.  Leave blank for defaults.
        error_opts : dict, optional
            A dictionary of options to pass to the matplotlib
            `ax.errorbar <https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.errorbar.html>`_ call
            internal to this function.  Leave blank for defaults.  Some special options are interpreted by
            this function and not passed to matplotlib: 'emarker' (default: '') specifies the marker type
            to place at cap of the errorbar.
        legend_opts : dict, optional
            A dictionary of options  to pass to the matplotlib
            `ax.legend <https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.legend.html>`_ call
            internal to this function.  Leave blank for defaults.
        overlay_overflow : str, optional
            If overflow behavior is not 'none', extra bins in the overlay axis will be overlaid or stacked,
            to represent the contents of the overflow bins.  See `Hist.sum` documentation for a description of the options.
        density : bool, optional
            If true, convert sum weights to probability density (i.e. integrates to 1 over domain of axis)
            (Note: this option conflicts with ``binwnorm``)
        binwnorm : float, optional
            If true, convert sum weights to bin-width-normalized, with unit equal to supplied value (usually you want to specify 1.)


    Returns
    -------
        ax : matplotlib.axes.Axes
            A matplotlib `Axes <https://matplotlib.org/3.1.1/api/axes_api.html>`_ object
    """
    import matplotlib.pyplot as plt
    import mplhep as hep

    if ax is None:
        ax = plt.gca()
    else:
        if not isinstance(ax, plt.Axes):
            raise ValueError("ax must be a matplotlib Axes object")
        if clear:
            ax.clear()
    if hist.dim() > 2:
        raise ValueError(
            "plot1d() can only support up to two dimensions (one for axis, one to stack or overlay)"
        )
    if overlay is None and hist.sparse_dim() == 1 and hist.dense_dim() == 1:
        overlay = hist.sparse_axes()[0].name
    elif overlay is None and hist.dim() > 1:
        raise ValueError(
            "plot1d() can only support one dimension without an overlay axis chosen"
        )
    if density and binwnorm is not None:
        raise ValueError("Cannot use density and binwnorm at the same time!")
    if binwnorm is not None:
        if not isinstance(binwnorm, numbers.Number):
            raise ValueError(
                "Bin width normalization not a number, but a %r" % binwnorm.__class__
            )
    if line_opts is None and fill_opts is None and error_opts is None:
        if stack:
            fill_opts = {}
        else:
            line_opts = {}
            error_opts = {}

    axis = hist.axes()[0]
    if overlay is not None:
        overlay = hist.axis(overlay)
        if axis == overlay:
            axis = hist.axes()[1]
    if isinstance(axis, SparseAxis):
        raise NotImplementedError("Plot a sparse axis (e.g. bar chart)")
    elif isinstance(axis, DenseAxis):
        ax.set_xlabel(axis.label)
        ax.set_ylabel(hist.label)
        edges = axis.edges(overflow=overflow).get()
        if order is None:
            identifiers = (
                hist.identifiers(overlay, overflow=overlay_overflow)
                if overlay is not None
                else [None]
            )
        else:
            identifiers = order
        plot_info = {
            "identifier": identifiers,
            "label": list(map(str, identifiers)),
            "sumw": [],
            "sumw2": [],
        }
        for i, identifier in enumerate(identifiers):
            if identifier is None:
                sumw, sumw2 = hist.values(sumw2=True, overflow=overflow)[()]
            elif isinstance(overlay, SparseAxis):
                sumw, sumw2 = hist.integrate(overlay, identifier).values(
                    sumw2=True, overflow=overflow
                )[()]
            else:
                sumw, sumw2 = hist.values(sumw2=True, overflow="allnan")[()]
                the_slice = (
                    i if overflow_behavior(overlay_overflow).start is None else i + 1,
                    overflow_behavior(overflow),
                )
                if hist._idense(overlay) == 1:
                    the_slice = (the_slice[1], the_slice[0])
                sumw = sumw[the_slice]
                sumw2 = sumw2[the_slice]
            plot_info["sumw"].append(sumw.get())
            plot_info["sumw2"].append(sumw2.get())

        def w2err(sumw, sumw2):
            err = []
            for a, b in zip(sumw, sumw2):
                err.append(numpy.abs(poisson_interval(a, b) - a))
            return err

        kwargs = None
        if line_opts is not None and error_opts is None:
            _error = None
        else:
            _error = w2err(plot_info["sumw"], plot_info["sumw2"])
        if fill_opts is not None:
            histtype = "fill"
            kwargs = fill_opts
        elif error_opts is not None and line_opts is None:
            histtype = "errorbar"
            kwargs = error_opts
        else:
            histtype = "step"
            kwargs = line_opts
        if kwargs is None:
            kwargs = {}

        hep.histplot(
            plot_info["sumw"],
            edges,
            label=plot_info["label"],
            yerr=_error,
            histtype=histtype,
            ax=ax,
            density=density,
            binwnorm=binwnorm,
            stack=stack,
            **kwargs,
        )

        if stack and error_opts is not None:
            stack_sumw = numpy.sum(plot_info["sumw"], axis=0)
            stack_sumw2 = numpy.sum(plot_info["sumw2"], axis=0)
            err = poisson_interval(stack_sumw, stack_sumw2)
            if binwnorm is not None:
                err *= binwnorm / numpy.diff(edges)[None, :]
            opts = {
                "step": "post",
                "label": "Sum unc.",
                "hatch": "///",
                "facecolor": "none",
                "edgecolor": (0, 0, 0, 0.5),
                "linewidth": 0,
            }
            opts.update(error_opts)
            ax.fill_between(
                x=edges,
                y1=numpy.r_[err[0, :], err[0, -1]],
                y2=numpy.r_[err[1, :], err[1, -1]],
                **opts,
            )

        if legend_opts is not None:
            _label = overlay.label if overlay is not None else ""
            ax.legend(title=_label, **legend_opts)
        else:
            ax.legend(title=_label)
        ax.autoscale(axis="x", tight=True)
        ax.set_ylim(0, None)

    return ax


def plotratio(
    num,
    denom,
    ax=None,
    clear=True,
    overflow="none",
    xerr=False,
    error_opts=None,
    denom_fill_opts=None,
    guide_opts=None,
    unc="clopper-pearson",
    label=None,
):
    """Create a ratio plot, dividing two compatible histograms

    Parameters
    ----------
        num : Hist
            Numerator, a single-axis histogram
        denom : Hist
            Denominator, a single-axis histogram
        ax : matplotlib.axes.Axes, optional
            Axes object (if None, one is created)
        clear : bool, optional
            Whether to clear Axes before drawing (if passed); if False, this function will skip drawing the legend
        overflow : str, optional
            If overflow behavior is not 'none', extra bins will be drawn on either end of the nominal
            axis range, to represent the contents of the overflow bins.  See `Hist.sum` documentation
            for a description of the options.
        xerr: bool, optional
            If true, then error bars are drawn for x-axis to indicate the size of the bin.
        error_opts : dict, optional
            A dictionary of options to pass to the matplotlib
            `ax.errorbar <https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.errorbar.html>`_ call
            internal to this function.  Leave blank for defaults.  Some special options are interpreted by
            this function and not passed to matplotlib: 'emarker' (default: '') specifies the marker type
            to place at cap of the errorbar.
        denom_fill_opts : dict, optional
            A dictionary of options to pass to the matplotlib
            `ax.fill_between <https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.fill_between.html>`_ call
            internal to this function, filling the denominator uncertainty band.  Leave blank for defaults.
        guide_opts : dict, optional
            A dictionary of options to pass to the matplotlib
            `ax.axhline <https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.axhline.html>`_ call
            internal to this function, to plot a horizontal guide line at ratio of 1.  Leave blank for defaults.
        unc : str, optional
            Uncertainty calculation option: 'clopper-pearson' interval for efficiencies; 'poisson-ratio' interval
            for ratio of poisson distributions; 'num' poisson interval of numerator scaled by denominator value
            (common for data/mc, for better or worse).
        label : str, optional
            Associate a label to this entry (note: y axis label set by ``num.label``)

    Returns
    -------
        ax : matplotlib.axes.Axes
            A matplotlib `Axes <https://matplotlib.org/3.1.1/api/axes_api.html>`_ object
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        if not isinstance(ax, plt.Axes):
            raise ValueError("ax must be a matplotlib Axes object")
        if clear:
            ax.clear()
    if not num.compatible(denom):
        raise ValueError(
            "numerator and denominator histograms have incompatible axis definitions"
        )
    if num.dim() > 1:
        raise ValueError("plotratio() can only support one-dimensional histograms")
    if error_opts is None and denom_fill_opts is None and guide_opts is None:
        error_opts = {}
        denom_fill_opts = {}

    axis = num.axes()[0]
    if isinstance(axis, SparseAxis):
        raise NotImplementedError("Ratio for sparse axes (labeled axis with errorbars)")
    elif isinstance(axis, DenseAxis):
        ax.set_xlabel(axis.label)
        ax.set_ylabel(num.label)
        edges = axis.edges(overflow=overflow).get()
        centers = axis.centers(overflow=overflow).get()
        ranges = (edges[1:] - edges[:-1]) / 2 if xerr else None

        sumw_num, sumw2_num = num.values(sumw2=True, overflow=overflow)[()]
        sumw_denom, sumw2_denom = denom.values(sumw2=True, overflow=overflow)[()]
        sumw_num, sumw2_num = sumw_num.get(), sumw2_num.get()
        sumw_denom, sumw2_denom = sumw_denom.get(), sumw2_denom.get()

        rsumw = sumw_num / sumw_denom
        if unc == "clopper-pearson":
            rsumw_err = numpy.abs(
                clopper_pearson_interval(sumw_num, sumw_denom) - rsumw
            )
        elif unc == "poisson-ratio":
            # poisson ratio n/m is equivalent to binomial n/(n+m)
            rsumw_err = numpy.abs(
                clopper_pearson_interval(sumw_num, sumw_num + sumw_denom) - rsumw
            )
        elif unc == "num":
            rsumw_err = numpy.abs(
                poisson_interval(rsumw, sumw2_num / sumw_denom**2) - rsumw
            )
        elif unc == "normal":
            rsumw_err = numpy.abs(
                normal_interval(sumw_num, sumw_denom, sumw2_num, sumw2_denom)
            )
        else:
            raise ValueError("Unrecognized uncertainty option: %r" % unc)

        if error_opts is not None:
            opts = {"label": label, "linestyle": "none"}
            opts.update(error_opts)
            emarker = opts.pop("emarker", "")
            errbar = ax.errorbar(
                x=centers, y=rsumw, xerr=ranges, yerr=rsumw_err, **opts
            )
            plt.setp(errbar[1], "marker", emarker)
        if denom_fill_opts is not None:
            unity = numpy.ones_like(sumw_denom)
            denom_unc = poisson_interval(unity, sumw2_denom / sumw_denom**2)
            opts = {"step": "post", "facecolor": (0, 0, 0, 0.3), "linewidth": 0}
            opts.update(denom_fill_opts)
            ax.fill_between(
                edges,
                numpy.r_[denom_unc[0], denom_unc[0, -1]],
                numpy.r_[denom_unc[1], denom_unc[1, -1]],
                **opts,
            )
        if guide_opts is not None:
            opts = {"linestyle": "--", "color": (0, 0, 0, 0.5), "linewidth": 1}
            opts.update(guide_opts)
            ax.axhline(1.0, **opts)

    if clear:
        ax.autoscale(axis="x", tight=True)
        ax.set_ylim(0, None)

    return ax


def plot2d(
    hist,
    xaxis,
    ax=None,
    clear=True,
    xoverflow="none",
    yoverflow="none",
    patch_opts=None,
    text_opts=None,
    density=False,
    binwnorm=None,
):
    """Create a 2D plot from a 2D `Hist` object

    Parameters
    ----------
        hist : Hist
            Histogram with two dimensions
        xaxis : str or Axis
            Which of the two dimensions to use as an x axis
        ax : matplotlib.axes.Axes, optional
            Axes object (if None, one is created)
        clear : bool, optional
            Whether to clear Axes before drawing (if passed); if False, this function will skip drawing the legend
        xoverflow : str, optional
            If overflow behavior is not 'none', extra bins will be drawn on either end of the nominal x
            axis range, to represent the contents of the overflow bins.  See `Hist.sum` documentation
            for a description of the options.
        yoverflow : str, optional
            Similar to ``xoverflow``
        patch_opts : dict, optional
            Options passed to the matplotlib `pcolormesh <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.pcolormesh.html>`_
            call internal to this function, to plot a rectangular grid of patches colored according to the bin values.
            Leave empty for defaults.
        text_opts : dict, optional
            Options passed to the matplotlib `text <https://matplotlib.org/api/text_api.html#matplotlib.text.Text>`_
            call internal to this function, to place a text label at each bin center with the bin value.  Special
            options interpreted by this function and not passed to matplotlib: 'format': printf-style float format
            , default '%.2g'.
        density : bool, optional
            If true, convert sum weights to probability density (i.e. integrates to 1 over domain of axis)
            (Note: this option conflicts with ``binwnorm``)
        binwnorm : float, optional
            If true, convert sum weights to bin-width-normalized, with unit equal to supplied value (usually you want to specify 1.)

    Returns
    -------
        ax : matplotlib.axes.Axes
            A matplotlib `Axes <https://matplotlib.org/3.1.1/api/axes_api.html>`_ object
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        if not isinstance(ax, plt.Axes):
            raise ValueError("ax must be a matplotlib Axes object")
        if clear:
            ax.clear()
        fig = ax.figure
    if hist.dim() != 2:
        raise ValueError("plot2d() can only support exactly two dimensions")
    if density and binwnorm is not None:
        raise ValueError("Cannot use density and binwnorm at the same time!")
    if binwnorm is not None:
        if not isinstance(binwnorm, numbers.Number):
            raise ValueError(
                "Bin width normalization not a number, but a %r" % binwnorm.__class__
            )
    if patch_opts is None and text_opts is None:
        patch_opts = {}

    xaxis = hist.axis(xaxis)
    yaxis = hist.axes()[1]
    transpose = False
    if yaxis == xaxis:
        yaxis = hist.axes()[0]
        transpose = True
    if isinstance(xaxis, SparseAxis) or isinstance(yaxis, SparseAxis):
        raise NotImplementedError("Plot a sparse axis (e.g. bar chart or labeled bins)")
    else:
        xedges = xaxis.edges(overflow=xoverflow).get()
        yedges = yaxis.edges(overflow=yoverflow).get()
        sumw, sumw2 = hist.values(sumw2=True, overflow="allnan")[()]
        sumw, sumw2 = sumw.get(), sumw2.get()
        if transpose:
            sumw = sumw.T
            sumw2 = sumw2.T
        # no support for different overflow behavior per axis, do it ourselves
        sumw = sumw[overflow_behavior(xoverflow), overflow_behavior(yoverflow)]
        sumw2 = sumw2[overflow_behavior(xoverflow), overflow_behavior(yoverflow)]
        if (density or binwnorm is not None) and numpy.sum(sumw) > 0:
            overallnorm = numpy.sum(sumw) * binwnorm if binwnorm is not None else 1.0
            areas = numpy.multiply.outer(numpy.diff(xedges), numpy.diff(yedges))
            binnorms = overallnorm / (areas * numpy.sum(sumw))
            sumw = sumw * binnorms
            sumw2 = sumw2 * binnorms**2

        if patch_opts is not None:
            opts = {"cmap": "viridis"}
            opts.update(patch_opts)
            pc = ax.pcolormesh(xedges, yedges, sumw.T, **opts)
            ax.add_collection(pc)
            if clear:
                fig.colorbar(pc, ax=ax, label=hist.label)
        if text_opts is not None:
            for ix, xcenter in enumerate(xaxis.centers()):
                for iy, ycenter in enumerate(yaxis.centers()):
                    opts = {
                        "horizontalalignment": "center",
                        "verticalalignment": "center",
                    }
                    if patch_opts is not None:
                        opts["color"] = (
                            "black" if pc.norm(sumw[ix, iy]) > 0.5 else "lightgrey"
                        )
                    opts.update(text_opts)
                    txtformat = opts.pop("format", r"%.2g")
                    ax.text(xcenter, ycenter, txtformat % sumw[ix, iy], **opts)

    if clear:
        ax.set_xlabel(xaxis.label)
        ax.set_ylabel(yaxis.label)
        ax.set_xlim(xedges[0], xedges[-1])
        ax.set_ylim(yedges[0], yedges[-1])

    return ax


def plotgrid(
    h,
    figure=None,
    row=None,
    col=None,
    overlay=None,
    row_overflow="none",
    col_overflow="none",
    **plot_opts,
):
    """Create a grid of plots, enumerating identifiers on up to 3 axes

    Parameters
    ----------
        h : Hist
            A histogram with up to 3 axes
        figure : matplotlib.figure.Figure, optional
            If omitted, a new figure is created. Otherwise, the axes will be redrawn on this existing figure.
        row : str
            Name of row axis
        col : str
            Name of column axis
        overlay : str
            name of overlay axis
        row_overflow : str, optional
            If overflow behavior is not 'none', extra bins will be drawn on either end of the nominal x
            axis range, to represent the contents of the overflow bins.  See `Hist.sum` documentation
            for a description of the options.
        col_overflow : str, optional
            Similar to ``row_overflow``
        ``**plot_opts`` : kwargs
            The remaining axis of the histogram, after removing any of ``row,col,overlay`` specified,
            will be the plot axis, with ``plot_opts`` passed to the `plot1d` call.

    Returns
    -------
        axes : numpy.ndarray
            An array of matplotlib `Axes <https://matplotlib.org/3.1.1/api/axes_api.html>`_ objects
    """
    import matplotlib.pyplot as plt

    haxes = {ax.name for ax in h.axes()}
    nrow, ncol = 1, 1
    if row:
        row_identifiers = h.identifiers(row, overflow=row_overflow)
        nrow = len(row_identifiers)
        haxes.remove(row)
    if col:
        col_identifiers = h.identifiers(col, overflow=col_overflow)
        ncol = len(col_identifiers)
        haxes.remove(col)
    if overlay:
        haxes.remove(overlay)
    if len(haxes) > 1:
        raise ValueError(
            "More than one dimension left: {}".format(",".join(ax for ax in haxes))
        )
    elif len(haxes) == 0:
        raise ValueError("Not enough dimensions available in %r" % h)

    figsize = plt.rcParams["figure.figsize"]
    figsize = figsize[0] * max(ncol, 1), figsize[1] * max(nrow, 1)
    if figure is None:
        fig, axes = plt.subplots(
            nrow, ncol, figsize=figsize, squeeze=False, sharex=True, sharey=True
        )
    else:
        fig = figure
        shape = (0, 0)
        lastax = fig.get_children()[-1]
        if isinstance(lastax, plt.Axes):
            shape = lastax.rowNum + 1, lastax.colNum + 1
        if shape[0] == nrow and shape[1] == ncol:
            axes = numpy.array(fig.axes).reshape(shape)
        else:
            fig.clear()
            # fig.set_size_inches(figsize)
            axes = fig.subplots(nrow, ncol, squeeze=False, sharex=True, sharey=True)

    for icol in range(ncol):
        hcol = h
        coltitle = None
        if col:
            vcol = col_identifiers[icol]
            hcol = h.integrate(col, vcol)
            coltitle = str(vcol)
            if isinstance(vcol, Interval) and vcol.label is None:
                coltitle = f"{h.axis(col).label} ∈ {coltitle}"
        for irow in range(nrow):
            ax = axes[irow, icol]
            hplot = hcol
            rowtitle = None
            if row:
                vrow = row_identifiers[irow]
                hplot = hcol.integrate(row, vrow)
                rowtitle = str(vrow)
                if isinstance(vrow, Interval) and vrow.label is None:
                    rowtitle = f"{h.axis(row).label} ∈ {rowtitle}"

            plot1d(hplot, ax=ax, overlay=overlay, **plot_opts)
            if row is not None and col is not None:
                ax.set_title(f"{rowtitle}, {coltitle}")
            elif row is not None:
                ax.set_title(rowtitle)
            elif col is not None:
                ax.set_title(coltitle)

    for ax in axes.flatten():
        ax.autoscale(axis="y")
        ax.set_ylim(0, None)

    return axes


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


def bokeh_plot(histo, jup_url="http://127.0.0.1:8889"):
    if not isnotebook():
        raise NotImplementedError("Only usable in jupyter notebook")
    import bokeh.plotting.figure as bk_figure
    from bokeh import palettes

    # init bokeh
    from bokeh.application import Application
    from bokeh.application.handlers import FunctionHandler
    from bokeh.core.validation import silence
    from bokeh.core.validation.warnings import EMPTY_LAYOUT
    from bokeh.io import output_notebook  # enables plot interface in J notebook
    from bokeh.io import show
    from bokeh.layouts import column, row
    from bokeh.models import ColumnDataSource
    from bokeh.models.widgets import (
        CheckboxButtonGroup,
        Div,
        RadioButtonGroup,
        RangeSlider,
    )

    silence(EMPTY_LAYOUT, True)

    output_notebook()

    # Set up widgets
    cfg_labels = ["Ghost"]
    wi_config = CheckboxButtonGroup(labels=cfg_labels, active=[0])
    wi_dense_select = RadioButtonGroup(
        labels=[ax.name for ax in histo.dense_axes()], active=0
    )
    wi_sparse_select = RadioButtonGroup(
        labels=[ax.name for ax in histo.sparse_axes()], active=0
    )

    # Dense widgets
    sliders = {}
    for ax in histo.dense_axes():
        edge_vals = (histo.axis(ax.name).edges()[0], histo.axis(ax.name).edges()[-1])
        _smallest_bin = numpy.min(numpy.diff(histo.axis(ax.name).edges()))
        sliders[ax.name] = RangeSlider(
            title=ax.name,
            value=edge_vals,
            start=edge_vals[0],
            end=edge_vals[1],
            step=_smallest_bin,
            name=ax.name,
        )

    # Cat widgets
    togglers = {}
    for ax in histo.sparse_axes():
        togglers[ax.name] = CheckboxButtonGroup(
            labels=[i.name for i in ax.identifiers()], active=[0], name=ax.name
        )

    # Toggles for all widgets
    configers = {}
    for ax in histo.sparse_axes():
        configers[ax.name] = CheckboxButtonGroup(
            labels=["Display", "Ghost"], active=[0, 1], name=ax.name
        )
    for ax in histo.dense_axes():
        configers[ax.name] = CheckboxButtonGroup(
            labels=["Display"], active=[0], name=ax.name
        )

    # Figure
    fig = bk_figure(
        title="1D Projection",
        plot_width=500,
        plot_height=500,
        min_border=20,
        toolbar_location=None,
    )
    fig.yaxis.axis_label = "N"
    fig.xaxis.axis_label = "Quantity"

    # Iterate over possible overlays
    _max_idents = 0  # Max number of simultaneou histograms
    for ax in histo.sparse_axes():
        _max_idents = max(_max_idents, len([i.name for i in ax.identifiers()]))

    # Data source list
    sources = []
    sources_ghost = []
    for i in range(_max_idents):
        sources.append(ColumnDataSource(dict(left=[], top=[], right=[], bottom=[])))
        sources_ghost.append(
            ColumnDataSource(dict(left=[], top=[], right=[], bottom=[]))
        )

    # Hist list
    hists = []
    hists_ghost = []
    for i in range(_max_idents):
        if _max_idents < 10:
            _color = palettes.Category10[min(max(3, _max_idents), 10)][i]
        else:
            _color = palettes.magma(_max_idents)[i]
        hists.append(
            fig.quad(
                left="left",
                right="right",
                top="top",
                bottom="bottom",
                source=sources[i],
                alpha=0.9,
                color=_color,
            )
        )
        hists_ghost.append(
            fig.quad(
                left="left",
                right="right",
                top="top",
                bottom="bottom",
                source=sources_ghost[i],
                alpha=0.05,
                color=_color,
            )
        )

    def update_data(attrname, old, new):
        sparse_active = wi_sparse_select.active
        sparse_name = [ax.name for ax in histo.sparse_axes()][sparse_active]
        sparse_other = [ax.name for ax in histo.sparse_axes() if ax.name != sparse_name]

        dense_active = wi_dense_select.active
        dense_name = [ax.name for ax in histo.dense_axes()][dense_active]
        dense_other = [ax.name for ax in histo.dense_axes() if ax.name != dense_name]

        # Apply cuts in projections
        _h = histo.copy()
        for proj_ax in sparse_other:
            _idents = histo.axis(proj_ax).identifiers()
            _labels = [ident.name for ident in _idents]
            if 0 in configers[proj_ax].active:
                _h = _h.integrate(
                    proj_ax, [_labels[i] for i in togglers[proj_ax].active]
                )
            else:
                _h = _h.integrate(proj_ax)

        for proj_ax in dense_other:
            _h = _h.integrate(
                proj_ax, slice(sliders[proj_ax].value[0], sliders[proj_ax].value[1])
            )

        for cat_ix in range(_max_idents):
            # Update histo for each toggled overlay
            if cat_ix in togglers[sparse_name].active:
                cat_value = histo.axis(sparse_name).identifiers()[cat_ix]
                h1d = _h.integrate(sparse_name, cat_value)

                # Get shown histogram
                values = h1d.project(dense_name).values()
                if values != {}:
                    h = values[()]
                    bins = h1d.axis(dense_name).edges()

                    # Apply cuts on shown axis
                    bin_los = bins[:-1][bins[:-1] > sliders[dense_name].value[0]]
                    bin_his = bins[1:][bins[1:] < sliders[dense_name].value[1]]
                    new_bins = numpy.intersect1d(bin_los, bin_his)
                    bin_ixs = numpy.searchsorted(bins, new_bins)[:-1]
                    h = h[bin_ixs]

                    sources[cat_ix].data = dict(
                        left=new_bins[:-1],
                        right=new_bins[1:],
                        top=h,
                        bottom=numpy.zeros_like(h),
                    )
                else:
                    sources[cat_ix].data = dict(left=[], right=[], top=[], bottom=[])

                # Add ghosts
                if 0 in wi_config.active:
                    h1d = histo.integrate(sparse_name, cat_value)
                    for proj_ax in sparse_other:
                        _idents = histo.axis(proj_ax).identifiers()
                        _labels = [ident.name for ident in _idents]
                        if 1 not in configers[proj_ax].active:
                            h1d = h1d.integrate(
                                proj_ax, [_labels[i] for i in togglers[proj_ax].active]
                            )
                        else:
                            h1d = h1d.integrate(proj_ax)
                    values = h1d.project(dense_name).values()
                    if values != {}:
                        h = h1d.project(dense_name).values()[()]
                        bins = h1d.axis(dense_name).edges()
                        sources_ghost[cat_ix].data = dict(
                            left=bins[:-1],
                            right=bins[1:],
                            top=h,
                            bottom=numpy.zeros_like(h),
                        )
                    else:
                        sources_ghost[cat_ix].data = dict(
                            left=[], right=[], top=[], bottom=[]
                        )
            else:
                sources[cat_ix].data = dict(left=[], right=[], top=[], bottom=[])
                sources_ghost[cat_ix].data = dict(left=[], right=[], top=[], bottom=[])

        # Cosmetics
        fig.xaxis.axis_label = dense_name

    for name, slider in sliders.items():
        slider.on_change("value", update_data)
    for name, toggler in togglers.items():
        toggler.on_change("active", update_data)
    for name, configurer in configers.items():
        configurer.on_change("active", update_data)
    # Button
    for w in [wi_dense_select, wi_sparse_select, wi_config]:
        w.on_change("active", update_data)

    from bokeh.models.widgets import Panel, Tabs

    layout = row(
        fig,
        column(
            Div(
                text="<b>Overlay Axis:</b>",
                style={"font-size": "100%", "color": "black"},
            ),
            wi_sparse_select,
            Div(
                text="<b>Plot Axis:</b>", style={"font-size": "100%", "color": "black"}
            ),
            wi_dense_select,
            Div(
                text="<b>Categorical Cuts:</b>",
                style={"font-size": "100%", "color": "black"},
            ),
            *[toggler for name, toggler in togglers.items()],
            Div(
                text="<b>Dense Cuts:</b>", style={"font-size": "100%", "color": "black"}
            ),
            *[slider for name, slider in sliders.items()],
        ),
    )

    # Config prep
    incl_lists = [[], [], []]
    for i, key in enumerate(list(configers.keys())):
        incl_lists[i // max(5, len(list(configers.keys())) / 3)].append(
            Div(
                text=f"<b>{key}:</b>",
                style={"font-size": "70%", "color": "black"},
            )
        )
        incl_lists[i // max(5, len(list(configers.keys())) / 3)].append(configers[key])

    layout_cfgs = column(
        row(
            column(
                Div(
                    text="<b>Configs:</b>",
                    style={"font-size": "100%", "color": "black"},
                ),
                wi_config,
            )
        ),
        Div(
            text="<b>Axis togglers:</b>", style={"font-size": "100%", "color": "black"}
        ),
        row(
            column(incl_lists[0]),
            column(incl_lists[1]),
            column(incl_lists[2]),
        ),
    )

    # Update active buttons
    def update_layout(attrname, old, new):
        active_axes = [None]
        for name, wi in configers.items():
            if 0 in wi.active:
                active_axes.append(name)
        for child in layout.children[1].children:
            if child.name not in active_axes:
                child.visible = False
            else:
                child.visible = True

    for name, configurer in configers.items():
        configurer.on_change("active", update_layout)

    tab1 = Panel(child=layout, title="Projection")
    tab2 = Panel(child=layout_cfgs, title="Configs")
    tabs = Tabs(tabs=[tab1, tab2])

    def modify_doc(doc):
        doc.add_root(row(tabs, width=800))
        doc.title = "Sliders"

    handler = FunctionHandler(modify_doc)
    app = Application(handler)

    show(app, notebook_url=jup_url)
    update_data("", "", "")
