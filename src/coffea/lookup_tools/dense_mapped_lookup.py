import numbers
from threading import Lock

import numba
import numpy

from coffea.lookup_tools.lookup_base import lookup_base


class dense_mapped_lookup(lookup_base):
    _formulaLock = Lock()
    _formulaCache = {}

    def __init__(self, axes, mapping, formulas, feval_dim):
        super().__init__()
        self._axes = axes
        self._mapping = mapping
        self._formulas = formulas
        self._feval_dim = feval_dim

    @classmethod
    def _compile(cls, formula):
        with dense_mapped_lookup._formulaLock:
            try:
                return dense_mapped_lookup._formulaCache[formula]
            except KeyError:
                if "x" in formula:
                    feval = eval(
                        "lambda x: " + formula, {"log": numpy.log, "sqrt": numpy.sqrt}
                    )
                    out = numba.jit(nopython=True)(feval)
                else:
                    out = eval(formula)
                dense_mapped_lookup._formulaCache[formula] = out
                return out

    def _lookup(self, axis, values):
        if len(axis) == 2:
            return numpy.zeros(shape=values.shape, dtype=numpy.uint)
        return numpy.clip(
            numpy.searchsorted(axis, values, side="right") - 1, 0, len(axis) - 2
        )

    def _evaluate(self, *args, ignore_missing=False, **kwargs):
        if len(args) != len(self._axes):
            raise ValueError(
                "Incorrect number of arguments specified (expected %d got %d)"
                % (len(args), len(self._axes))
            )
        idx = (self._lookup(axis, arg) for axis, arg in zip(self._axes, args))
        mapidx = self._mapping[tuple(idx)]
        out = numpy.ones(mapidx.shape, dtype=numpy.common_type(*args))
        for ifunc in numpy.unique(mapidx):
            if ifunc < 0 and not ignore_missing:
                raise ValueError("No correction was available for some items")
            func = dense_mapped_lookup._compile(self._formulas[ifunc])
            where = mapidx == ifunc
            if isinstance(func, numbers.Number):
                out[where] = func
            else:
                if self._feval_dim is None:
                    raise RuntimeError("expected a dimension to pass to the formula")
                out[where] = func(
                    numpy.clip(
                        args[self._feval_dim][where],
                        self._axes[self._feval_dim][0],
                        self._axes[self._feval_dim][-1],
                    )
                )

        return out

    def __repr__(self):
        myrepr = object.__repr__(self) + "\n"
        myrepr += f"{self._dimension} dimensional histogram with axes:\n"
        temp = ""
        for idim, axis in enumerate(self._axes):
            temp += f"\t{idim + 1}: {axis}\n"
        myrepr += temp
        return myrepr
