import numbers
from functools import partial

import awkward
import dask_awkward
import numpy


def getfunction(
    args,
    thelookup,
    __non_array_args__=tuple(),
    __arg_indices__=tuple(),
    **kwargs,
):
    if not isinstance(args, (list, tuple)):
        args = (args,)

    if not len(args) + len(__non_array_args__) == len(__arg_indices__):
        raise ValueError(
            "Total length of array and non-array args should match expected placement."
        )

    if all(
        isinstance(x, (awkward.contents.NumpyArray, awkward.contents.EmptyArray))
        or not isinstance(x, (awkward.contents.Content))
        for x in args
    ):
        result = None
        backend = awkward.backend(*args)

        if backend != "cpu" and backend != "typetracer":
            raise NotImplementedError("support for cupy/jax/etc. numpy extensions")

        # order args and __non_array_args__ correctly
        repacked_args = [None] * len(__arg_indices__)
        for iarg, arg in enumerate(args):
            if backend == "cpu":
                repacked_args[__arg_indices__[iarg]] = awkward.to_numpy(arg)
            elif backend == "typetracer":
                repacked_args[__arg_indices__[iarg]] = awkward.to_numpy(
                    awkward.typetracer.length_zero_if_typetracer(arg)
                )

        for inaarg, naarg in enumerate(__non_array_args__):
            repacked_args[__arg_indices__[inaarg + len(args)]] = naarg

        result = thelookup._evaluate(*repacked_args, **kwargs)
        out = awkward.contents.NumpyArray(result)
        if backend == "typetracer":
            return out.to_typetracer(forget_length=True)
        return out
    return None


class _LookupXformFn:
    def __init__(self, *args, arg_indices, **kwargs):
        self.getfunction = getfunction
        self.__non_array_args__ = args
        self.__arg_indices__ = arg_indices
        self.kwargs = kwargs

    def __call__(self, thelookup, *args):
        func = partial(
            self.getfunction,
            thelookup=thelookup,
            __non_array_args__=self.__non_array_args__,
            __arg_indices__=self.__arg_indices__,
            **self.kwargs,
        )
        return awkward.transform(func, *args)


class lookup_base:
    """Base class for all objects that do some sort of value or function lookup"""

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        dask_label = kwargs.pop("dask_label", None)

        actual_args = []
        actual_arg_indices = []
        delay_args = []
        delay_arg_indices = []
        for iarg, arg in enumerate(args):
            if isinstance(arg, (awkward.highlevel.Array, dask_awkward.Array)):
                actual_args.append(arg)
                actual_arg_indices.append(iarg)
            else:
                delay_args.append(arg)
                delay_arg_indices.append(iarg)
        arg_indices = tuple(actual_arg_indices + delay_arg_indices)
        actual_args = tuple(actual_args)
        delay_args = tuple(delay_args)

        tomap = _LookupXformFn(
            *delay_args,
            arg_indices=arg_indices,
            **kwargs,
        )

        # if our inputs are all dask_awkward arrays, then we should map_partitions
        if any(isinstance(x, (dask_awkward.Array)) for x in args):
            import dask.delayed

            out_meta = tomap(self, *tuple([arg._meta for arg in actual_args]))

            if not hasattr(self, "_delayed_corr"):
                setattr(self, "_delayed_corr", dask.delayed(self))

            return dask_awkward.map_partitions(
                tomap,
                self._delayed_corr,
                *actual_args,
                label=dask_label,
                meta=out_meta,
            )

        if all(isinstance(x, (numpy.ndarray, numbers.Number, str)) for x in args):
            return self._evaluate(*args, **kwargs)
        elif any(
            not isinstance(x, (awkward.highlevel.Array, numbers.Number, str))
            for x in args
        ):
            raise TypeError(
                "lookup base must receive high level awkward arrays,"
                " numpy arrays, strings, or numbers!"
            )

        return tomap(self, *actual_args)

    def _evaluate(self, *args, **kwargs):
        raise NotImplementedError
