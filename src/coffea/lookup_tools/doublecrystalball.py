import numpy as np
from scipy.stats._continuous_distns import (
    _lazywhere,
    _norm_cdf,
    _norm_pdf_C,
    _norm_ppf,
    rv_continuous,
)


class doublecrystalball_gen(rv_continuous):
    r"""
    Double-sided Crystalball distribution
    %(before_notes)s
    Notes
    -----
    The probability density function for `doublecrystalball` is:
    .. math::
        f(x, \beta_{L}, \beta_{H}, m_{L}, m_{H})= \begin{cases}
                                                      N \exp(-x^2 / 2),  &\text{for } -\beta_{L} < x < \beta_{H}\\
                                                      N A_{L} (B_{L} - x)^{-m_{L}}  &\text{for } x \le -\beta_{L}\\
                                                      N A_{H} (B_{H} - x)^{-m_{H}}  &\text{for } x \ge \beta_{H}\\
                                                    \end{cases}
    where :math:`A_{i}=(m_{i} / |\beta_{i}|)^m_{i}  \exp(-\beta_{i}^2 / 2)`,
    :math:`B_{i}=m_{i}/|\beta_{i}| - |\beta_{i}|` and :math:`N` is a normalisation constant.
    `doublecrystalball` takes :math:`\beta_{i} > 0` and :math:`m_{i} > 1` as shape
    parameters.  :math:`\beta_{i}` defines the point where the pdf changes
    from a power-law to a Gaussian distribution.  :math:`m_{i}` is the power
    of the power-law tail. Here, :math:`i=L, H` refers to the low and high tails, respectively.
    References
    ----------
    .. [1] "Crystal Ball Function",
           https://en.wikipedia.org/wiki/Crystal_Ball_function
    %(after_notes)s
    .. versionadded:: 0.19.0
    %(example)s
    """

    def _pdf(self, x, betaL, betaH, mL, mH):
        """
        Return PDF of the double-sided crystalball function.
        """
        N = 1.0 / (
            mL / betaL / (mL - 1) * np.exp(-0.5 * betaL * betaL)
            + mH / betaH / (mH - 1) * np.exp(-0.5 * betaH * betaH)
            + _norm_pdf_C * (_norm_cdf(betaH) - _norm_cdf(-betaL))
        )

        def core(x, beta, m):
            return np.exp(-0.5 * x * x)

        def tail(x, beta, m):
            return (
                (m / beta) ** m
                * np.exp(-0.5 * beta * beta)
                * (m / beta - beta - x) ** (-m)
            )

        def lhs(x, betaL, betaH, mL, mH):
            return tail(x, betaL, mL)

        def rhs(x, betaL, betaH, mL, mH):
            return _lazywhere(x < betaH, (-x, betaH, mH), f=core, f2=tail)

        return N * _lazywhere(x > -betaL, (x, betaL, betaH, mL, mH), f=rhs, f2=lhs)

    def _logpdf(self, x, betaL, betaH, mL, mH):
        """
        Return the log of the PDF of the double-sided crystalball function.
        """
        N = 1.0 / (
            mL / betaL / (mL - 1) * np.exp(-0.5 * betaL * betaL)
            + mH / betaH / (mH - 1) * np.exp(-0.5 * betaH * betaH)
            + _norm_pdf_C * (_norm_cdf(betaH) - _norm_cdf(-betaL))
        )

        def core(x, beta, m):
            return -0.5 * x * x

        def tail(x, beta, m):
            return (
                m * np.log(m / beta)
                - 0.5 * beta * beta
                - m * np.log(m / beta - beta - x)
            )

        def lhs(x, betaL, betaH, mL, mH):
            return tail(x, betaL, mL)

        def rhs(x, betaL, betaH, mL, mH):
            return _lazywhere(x < betaH, (-x, betaH, mH), f=core, f2=tail)

        return np.log(N) + _lazywhere(
            x > -betaL, (x, betaL, betaH, mL, mH), f=rhs, f2=lhs
        )

    def _cdf(self, x, betaL, betaH, mL, mH):
        """
        Return CDF of the double-sided crystalball function
        """
        N = 1.0 / (
            mL / betaL / (mL - 1) * np.exp(-0.5 * betaL * betaL)
            + mH / betaH / (mH - 1) * np.exp(-0.5 * betaH * betaH)
            + _norm_pdf_C * (_norm_cdf(betaH) - _norm_cdf(-betaL))
        )

        def inttail(beta, m):
            return m / beta / (m - 1) * np.exp(-0.5 * beta * beta)

        def intcore(betaL, betaH):
            return _norm_pdf_C * (_norm_cdf(betaH) - _norm_cdf(-betaL))

        def tail(x, beta, m):
            return (
                (m / beta) ** m
                * np.exp(-0.5 * beta * beta)
                * (m / beta - beta - x) ** (1 - m)
                / (m - 1)
            )

        def hightail(x, betaL, betaH, mL, mH):
            return (
                inttail(betaL, mL)
                + intcore(betaL, betaH)
                + inttail(betaH, mH)
                - tail(-x, betaH, mH)
            )

        def core(x, betaL, betaH, mL, mH):
            return inttail(betaL, mL) + _norm_pdf_C * (_norm_cdf(x) - _norm_cdf(-betaL))

        def lhs(x, betaL, betaH, mL, mH):
            return tail(x, betaL, mL)

        def rhs(x, betaL, betaH, mL, mH):
            return _lazywhere(x < betaH, (x, betaL, betaH, mL, mH), f=core, f2=hightail)

        return N * _lazywhere(x > -betaL, (x, betaL, betaH, mL, mH), f=rhs, f2=lhs)

    def _ppf(self, p, betaL, betaH, mL, mH):
        """
        Return PPF of the double-sided crystalball function
        """

        def inttail(beta, m):
            return m / beta / (m - 1) * np.exp(-0.5 * beta * beta)

        def intcore(betaL, betaH):
            return _norm_pdf_C * (_norm_cdf(betaH) - _norm_cdf(-betaL))

        def hightail(p, betaL, betaH, mL, mH):
            CL = inttail(betaL, mL)
            CH = inttail(betaH, mH)
            C = CL + CH
            N = 1 / (C + intcore(betaL, betaH))
            eb2H = np.exp(-0.5 * betaH * betaH)
            return -(
                mH / betaH
                - betaH
                - ((mH - 1) * (mH / betaH) ** (-mH) / eb2H * (1 - p) / N)
                ** (1 / (1 - mH))
            )

        def lowtail(p, betaL, betaH, mL, mH):
            CL = inttail(betaL, mL)
            CH = inttail(betaH, mH)
            C = CL + CH
            N = 1 / (C + intcore(betaL, betaH))
            eb2L = np.exp(-0.5 * betaL * betaL)
            return (
                mL / betaL
                - betaL
                - ((mL - 1) * (mL / betaL) ** (-mL) / eb2L * p / N) ** (1 / (1 - mL))
            )

        def core(p, betaL, betaH, mL, mH):
            CL = inttail(betaL, mL)
            CH = inttail(betaH, mH)
            C = CL + CH
            N = 1 / (C + intcore(betaL, betaH))
            return _norm_ppf(_norm_cdf(-betaL) + (1 / _norm_pdf_C) * (p / N - CL))

        def ppf_greater(p, betaL, betaH, mL, mH):
            N = 1.0 / (inttail(betaL, mL) + intcore(betaL, betaH) + inttail(betaH, mH))
            pbetaH = 1 - (N * (mH / betaH) * np.exp(-0.5 * betaH * betaH) / (mH - 1))
            return _lazywhere(
                p > pbetaH, (p, betaL, betaH, mL, mH), f=hightail, f2=core
            )

        N = 1.0 / (inttail(betaL, mL) + intcore(betaL, betaH) + inttail(betaH, mH))
        pbetaL = N * (mL / betaL) * np.exp(-0.5 * betaL * betaL) / (mL - 1)
        return _lazywhere(
            p < pbetaL, (p, betaL, betaH, mL, mH), f=lowtail, f2=ppf_greater
        )

    def _munp(self, n, betaL, betaH, mL, mH):
        """
        Returns the n-th non-central moment of the double-sided crystalball function.
        """
        # this should be copied from crystalball and updated
        raise (NotImplementedError)

    def _argcheck(self, betaL, betaH, mL, mH):
        """
        Shape parameter bounds are m > 1 and beta > 0.
        """
        return (mL > 1) & (betaL > 0) & (mH > 1) & (betaH > 1)


doublecrystalball = doublecrystalball_gen(
    name="doublecrystalball", longname="A Double-sided Crystalball Function"
)
