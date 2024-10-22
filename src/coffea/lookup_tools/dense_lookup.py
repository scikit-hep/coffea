from copy import deepcopy

import numpy

from coffea.lookup_tools.lookup_base import lookup_base


class dense_lookup(lookup_base):
    def __init__(self, values, dims, feval_dim=None):
        super().__init__()
        self._dimension = 0
        whattype = type(dims)
        if whattype == numpy.ndarray:
            self._dimension = 1
        else:
            self._dimension = len(dims)
        if self._dimension == 0:
            raise Exception(f"Could not define dimension for {whattype}")
        self._axes = deepcopy(dims)
        self._feval_dim = None
        vals_are_strings = (
            "string" in values.dtype.name
            or "str" in values.dtype.name
            or "unicode" in values.dtype.name
            or "bytes" in values.dtype.name
        )  # ....
        if not isinstance(values, numpy.ndarray):
            raise TypeError("values is not a numpy array, but %r" % type(values))
        if vals_are_strings:
            raise Exception("dense_lookup cannot handle string values!")
        self._values = deepcopy(values)

    def _evaluate(self, *args, **kwargs):
        if len(args) != self._dimension:
            raise ValueError(f"Insufficient arguments for correction {self}")
        indices = []
        if self._dimension == 1:
            axes = (
                self._axes if isinstance(self._axes, numpy.ndarray) else self._axes[0]
            )
            indices.append(
                numpy.clip(
                    numpy.searchsorted(axes, args[0], side="right") - 1,
                    0,
                    self._values.shape[0] - 1,
                )
            )
        else:
            for dim in range(self._dimension):
                indices.append(
                    numpy.clip(
                        numpy.searchsorted(self._axes[dim], args[dim], side="right")
                        - 1,
                        0,
                        self._values.shape[dim] - 1,
                    )
                )
        return self._values[tuple(indices)]

    def __repr__(self):
        myrepr = object.__repr__(self)
        myrepr += f" {self._dimension} dimensional histogram with axes:\n"
        temp = ""
        if self._dimension == 1:
            temp = f"\t1: {self._axes}\n"
        else:
            temp = f"\t1: {self._axes[0]}\n"
        for idim in range(1, self._dimension):
            temp += f"\t{idim + 1}: {self._axes[idim]}\n"
        myrepr += temp
        return myrepr
