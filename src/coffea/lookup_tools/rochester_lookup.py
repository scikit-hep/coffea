import awkward
import dask_awkward as dak
import numpy

from coffea.lookup_tools.dense_lookup import dense_lookup

# crystalball is single sided, local reimplementation of double-sided here until
# the PR can be merged
# from scipy.stats import crystalball
from coffea.lookup_tools.doublecrystalball import doublecrystalball


class rochester_lookup:
    def __init__(self, wrapped_values):
        self._build(wrapped_values)

    def _build(self, wrapped_values):
        self._nsets = wrapped_values["nsets"]
        self._members = wrapped_values["members"]

        # now build the lookup tables
        # for data scale, simple, just M A in bins of eta,phi
        edges = wrapped_values["edges"]["scales"]
        M = wrapped_values["values"]["M"]
        A = wrapped_values["values"]["A"]
        self._M = {
            s: {m: {t: dense_lookup(M[s][m][t], edges) for t in M[s][m]} for m in M[s]}
            for s in M
        }
        self._A = {
            s: {m: {t: dense_lookup(A[s][m][t], edges) for t in A[s][m]} for m in A[s]}
            for s in A
        }

        # for mc scale, more complicated
        # version 1 if gen pt available
        # only requires the kRes lookup
        edges = wrapped_values["edges"]["res"]
        kRes = wrapped_values["values"]["kRes"]
        self._kRes = {
            s: {
                m: {t: dense_lookup(kRes[s][m][t], edges) for t in kRes[s][m]}
                for m in kRes[s]
            }
            for s in kRes
        }

        # version 2 if gen pt not available
        edges = wrapped_values["edges"]["cb"]
        rsPars = wrapped_values["values"]["rsPars"]
        cbS = wrapped_values["values"]["cbS"]
        cbA = wrapped_values["values"]["cbA"]
        cbN = wrapped_values["values"]["cbN"]
        self._rsPars = {
            s: {
                m: {t: dense_lookup(rsPars[s][m][t], edges) for t in rsPars[s][m]}
                for m in rsPars[s]
            }
            for s in rsPars
        }
        self._cbS = {
            s: {m: dense_lookup(cbS[s][m], edges) for m in cbS[s]} for s in cbS
        }
        self._cbA = {
            s: {m: dense_lookup(cbA[s][m], edges) for m in cbA[s]} for s in cbA
        }
        self._cbN = {
            s: {m: dense_lookup(cbN[s][m], edges) for m in cbN[s]} for s in cbN
        }

        self._loaduncs = len(self._M.keys()) > 1

    def _error(self, func, *args):
        if not self._loaduncs:
            return None

        newargs = args + (0, 0)
        default = func(*newargs)
        result = awkward.zeros_like(default)
        for s in range(self._nsets):
            oneOver = 1.0 / self._members[s]
            for m in range(self._members[s]):
                newargs = args + (s, m)
                d = func(*newargs) - default
                result = result + d * d * oneOver
        return result**0.5

    def kScaleDT(self, charge, pt, eta, phi, s=0, m=0):
        """Momentum scale correction for data
        required:
            charge
            pt
            eta
            phi
        optional:
            s: Rochester correction set to use (default 0)
            m: Rochester correction member to use (default 0)
        """
        # type = 1 corresponds to data
        M = self._M[s][m][1](eta, phi)
        A = self._A[s][m][1](eta, phi)
        return 1.0 / (M + charge * A * pt)

    def kScaleDTerror(self, charge, pt, eta, phi):
        """Momentum scale correction uncertainty for data
        required:
            charge
            pt
            eta
            phi
        """
        return self._error(self.kScaleDT, charge, pt, eta, phi)

    def kScaleMC(self, charge, pt, eta, phi, s=0, m=0):
        """Momentum scale correction for mc (not recommended, use kSpreadMC instead)
        required:
            charge
            pt
            eta
            phi
        optional:
            s: Rochester correction set to use (default 0)
            m: Rochester correction member to use (default 0)
        """
        # type = 0 corresponds to mc
        M = self._M[s][m][0](eta, phi)
        A = self._A[s][m][0](eta, phi)
        return 1.0 / (M + charge * A * pt)

    def kScaleMCerror(self, charge, pt, eta, phi):
        """Momentum scale correction uncertainty for mc (not recommended, use kSpreadMC instead)
        required:
            charge
            pt
            eta
            phi
        """
        return self._error(self.kScaleMC, charge, pt, eta, phi)

    def kSpreadMC(self, charge, pt, eta, phi, genpt, s=0, m=0):
        """Momentum scale correction for mc (if genpt not available, use kSmearMC)
        required:
            charge
            pt
            eta
            phi
            genpt: the match gen particle pt
        optional:
            s: Rochester correction set to use (default 0)
            m: Rochester correction member to use (default 0)
        """
        k = self.kScaleMC(charge, pt, eta, phi, s, m)
        return k * self._kSpread(genpt, k * pt, eta, s, m)

    def kSpreadMCerror(self, charge, pt, eta, phi, genpt):
        """Momentum scale correction uncertainty for mc (if genpt not available, use kSmearMC)
        required:
            charge
            pt
            eta
            phi
            genpt: the match gen particle pt
        """
        return self._error(self.kSpreadMC, charge, pt, eta, phi, genpt)

    def _kSpread(self, genpt, kpt, eta, s=0, m=0):
        x = genpt / kpt
        abseta = abs(eta)
        kData = self._kRes[s][m][1](abseta)  # type 1 is data
        kMC = self._kRes[s][m][0](abseta)  # type 0 is MC
        return x / (1.0 + (x - 1.0) * kData / kMC)

    def kSmearMC(self, charge, pt, eta, phi, nl, u, s=0, m=0):
        """Momentum scale correction for mc (should prefer to use kSpreadMC)
        required:
            charge
            pt
            eta
            phi
            nl: number of tracker layers with measurements
            u: random float between 0 and 1
        optional:
            s: Rochester correction set to use (default 0)
            m: Rochester correction member to use (default 0)
        """
        k = self.kScaleMC(charge, pt, eta, phi, s, m)
        return k * self._kExtra(k * pt, eta, nl, u, s, m)

    def kSmearMCerror(self, charge, pt, eta, phi, nl, u):
        """Momentum scale correction uncertainty for mc (should prefer to use kSpreadMC)
        required:
            charge
            pt
            eta
            phi
            nl: number of tracker layers with measurements
            u: random float between 0 and 1
        """
        return self._error(self.kSmearMC, charge, pt, eta, phi, nl, u)

    def _sigma(self, pt, eta, nl, s=0, m=0):
        dpt = pt - 45
        abseta = abs(eta)
        return (
            self._rsPars[s][m][0](abseta, nl)
            + self._rsPars[s][m][1](abseta, nl) * dpt
            + self._rsPars[s][m][2](abseta, nl) * dpt * dpt
        )

    def _kExtra(self, kpt, eta, nl, u, s=0, m=0):
        # if it is a jagged array, save the offsets then flatten everything
        # needed for the ternary conditions later
        abseta = abs(eta)
        kData = self._kRes[s][m][1](abseta)  # type 1 is data
        kMC = self._kRes[s][m][0](abseta)  # type 0 is MC
        mask = kData > kMC
        x = awkward.zeros_like(kpt)
        sigma = self._sigma(kpt, eta, nl, s, m)
        # Rochester cbA = beta, cbN = m, as well as cbM (always 0?) = loc and cbS = scale to transform y = (x-loc)/scale in the pdf method
        cbA = self._cbA[s][m](abseta, nl)
        cbN = self._cbN[s][m](abseta, nl)
        cbS = self._cbS[s][m](abseta, nl)
        counts = awkward.num(u)
        u_flat = awkward.flatten(u)
        loc = awkward.zeros_like(u_flat)
        cbA_flat = awkward.flatten(cbA)
        cbN_flat = awkward.flatten(cbN)
        cbS_flat = awkward.flatten(cbS)

        args = (u_flat, cbA_flat, cbA_flat, cbN_flat, cbN_flat, loc, cbS_flat)

        if any(isinstance(arg, dak.Array) for arg in args):

            def apply(*args):
                args_lz = [
                    awkward.typetracer.length_zero_if_typetracer(arg) for arg in args
                ]
                out = awkward.Array(doublecrystalball.ppf(*args_lz))
                if awkward.backend(args[0]) == "typetracer":
                    out = awkward.Array(
                        out.layout.to_typetracer(forget_length=True),
                        behavior=out.behavior,
                    )
                return out

            invcdf = dak.map_partitions(apply, *args)
        else:
            invcdf = doublecrystalball.ppf(*args)

        invcdf = awkward.unflatten(invcdf, counts)

        x = awkward.where(
            mask,
            (numpy.sqrt(kData * kData - kMC * kMC) * sigma * invcdf),
            x,
        )
        result = awkward.where(x > -1, 1.0 / (1.0 + x), awkward.ones_like(kpt))
        if isinstance(kpt, numpy.ndarray):
            result = numpy.array(result)
        return result
