import abc
import warnings
from typing import Dict, List, Set, Tuple

import awkward
import dask
import dask_awkward
import numpy
from dask.base import unpack_collections


class nonserializable_attribute:
    """
    Generalizing the container for non-serializable objects.

    For a given unserializable object of name "obj", on container "x"
    initialization, it will create a dummy instance of the "x._obj = None". On
    the first time that "x.obj" is called, the contents of "x._obj" will be
    replaced by the return value of "x._create_obj()" method. The corresponding
    "_create_obj" method must be able to return the correct object without
    additional function arguments, so all arguments required to create the
    object of interest needs to be stored in the container instance.

    Parameters
    ----------
        nonserial_list : List[str]
            A list of string for the names of the unserializable objects.
    """

    def __init__(self, nonserial_list: List[str]):
        self._nonserial_list = nonserial_list

        for name in nonserial_list:
            assert (
                name.isidentifier()
            ), f"Requested variable {name} cannot be used as variable name"
            assert hasattr(
                self, "_create_" + name
            ), f"Method _create_{name} needs to be implemented in by class"
            setattr(self, "_" + name, None)

    def __getattr__(self, name):
        """
        The method __getattr__ is only invoke if the attribute cannot be found
        with conventional method (such as with the not-explicitly defined method
        nonserializable_attribute.att_name with no "_" prefix)
        """
        if name in self._nonserial_list:
            if getattr(self, "_" + name) is None:
                setattr(self, "_" + name, getattr(self, "_create_" + name)())
            return getattr(self, "_" + name)
        else:
            raise AttributeError(f"{name} not defined for {self.__class__.__name__}!")

    def __getstate__(self):
        """
        Explicitly setting all the unserializable objects to be hidden from
        getstate requests. This ensures that wrapper objects can be pickled
        regardless of whether evaluations of the various objects has been
        carried out or not.
        """
        state = self.__dict__.copy()
        for name in state["_nonserial_list"]:
            state["_" + name] = None
        return state

    def __setstate__(self, d: dict):
        """
        Due to the overloading the of the __getattr__ and __getstate__ method,
        we need to explicitly define the __setstate__ method. Notice that due to
        the design of the __getstate__ method, this should never receive a
        dictionary that initializes the unserializable objects under normal
        operations.
        """
        self.__dict__.update(d)


class container_converter:
    """
    Running over the all arguments of arbitrary function inputs (*args,
    **kwargs), iterating through the base python containers and running a
    conversion function on each of the leaf entries.

    The method_map has types as the map key and a callable as the value, the
    callable will only be used on the matching type.

    A default conversion callable can also be provided to convert objects that
    is not explicitly listed. By default, if a type is not explicitly listed, an
    exception is raised.
    """

    def __init__(self, method_map, default_conv=None):
        self.method_map = method_map
        self.default_conv = default_conv
        if self.default_conv is None:
            self.default_conv = self.unrecognized

    def _convert(self, arg, maybe_backends):
        if isinstance(arg, Dict):
            return dict(
                {key: self.convert(val, maybe_backends) for key, val in arg.items()}
            )
        elif isinstance(arg, (List, Set, Tuple)):
            return arg.__class__(self.convert(val, maybe_backends) for val in arg)
        else:
            for itype, call in self.method_map.items():
                if isinstance(arg, itype):
                    if maybe_backends is not None and itype is awkward.highlevel.Array:
                        maybe_backends.add(awkward.backend(arg))
                    return call(arg)

            return self.default_conv(arg)

    def convert(self, arg, maybe_backends=None):
        out = self._convert(arg, maybe_backends)
        return out

    def __call__(self, *args, **kwargs) -> Tuple:
        backends = set()
        out_args = self.convert(args, backends)
        out_kwargs = self.convert(kwargs, backends)
        return (out_args, out_kwargs), backends

    @staticmethod
    def no_action(x):
        return x

    @staticmethod
    def unrecognized(x):
        raise ValueError(f"Unknown type {type(x)}")


class numpy_call_wrapper(abc.ABC):
    """
    Wrapper for awkward.to_numpy evaluations for dask_awkward array inputs.

    For tools outside the coffea package (like for ML inference), the inputs
    typically expect a numpy-like input. This class wraps up the user-level
    awkward->numpy data mangling and the underlying numpy evaluation calls to
    recognizable to dask.

    For the class to be fully functional, the user must overload these methods:

    - numpy_call: How the evaluation using all-numpy tool be performed
    - prepare_awkward: How awkward arrays should be translated to a numpy
      format that is compatible with the numpy_call

    Additionally, the following helper functions can be omitted, but will help
    assist with either code debugging or for data mangling niceties.

    - validate_numpy_input: makes sure the computation routine understand the
      input.
    - postprocess_awkward: Additional translation to convert numpy outputs to
      awkward (defaults to a simple `awkward.from_numpy` conversion)
    """

    # Commonly used helper classes so defining as static method
    _ak_to_np_ = container_converter(
        {awkward.Array: awkward.to_numpy}, default_conv=container_converter.no_action
    )
    _np_to_ak_ = container_converter(
        {numpy.ndarray: awkward.from_numpy},
        default_conv=container_converter.no_action,
    )

    def __init__(self):
        pass

    def validate_numpy_input(self, *args, **kwargs) -> None:
        """
        Validating that the numpy-like input arguments are compatible with the
        underlying evaluation calls. This function should raise an exception if
        invalid input values are found. The base method performs no checks but
        raises a warning that no checks were performed.
        """
        warnings.warn("No format checks were performed on input!")

    def _call_numpy(self, *args, **kwargs):
        """
        Thin wrapper such that the validate_numpy_inputs is called.
        """
        self.validate_numpy_input(*args, **kwargs)
        return self.numpy_call(*args, **kwargs)

    @abc.abstractmethod
    def numpy_call(self, *args, **kwargs):
        """
        Underlying numpy-like evaluation. This method should be reimplemented by
        the user or by tool-specialized classes.
        """
        pass

    @abc.abstractmethod
    def prepare_awkward(self, *args, **kwargs) -> Tuple:
        r"""
        Converting awkward-array like inputs into be numpy-compatible awkward-arrays
        compatible with the `numpy_call` method. The actual conversion to numpy is
        handled automatically. The return value should be (\*args, \*\*kwargs) pair
        that is compatible with the numpy_call.

        Consult the following documentation to find the awkward operations
        needed.
        https://awkward-array.org/doc/main/user-guide/how-to-restructure-pad.html
        """
        pass

    def get_awkward_lib(self, *args, **kwargs):
        all_args = [*args, *kwargs.values()]
        has_ak = any(isinstance(arg, awkward.Array) for arg in all_args)
        has_dak = any(isinstance(arg, dask_awkward.Array) for arg in all_args)
        if has_ak and has_dak:
            raise RuntimeError("Cannot mix awkward and dask_awkward in calculations")
        elif has_ak:
            return awkward
        elif has_dak:
            return dask_awkward
        else:
            return None

    def postprocess_awkward(self, return_array, *args, **kwargs):
        """
        Additional conversion from the numpy_call output back to awkward arrays.
        This method does not need to need to be overloaded, but can make the
        data-mangling that occurs outside the class cleaner (ex: additional
        awkward unflatten calls). To ensure that the data mangling can occur,
        the unformatted awkward-like inputs are also passed to this function.

        For the base method, we will simply iterate over the containers and call
        the default `awkward.from_numpy` conversion
        """
        return return_array

    def _call_awkward(self, *args, **kwargs):
        """
        The common routine of prepare_awkward conversion, numpy evaluation,
        then postprocess_awkward conversion.
        """
        ak_args, ak_kwargs = self.prepare_awkward(*args, **kwargs)
        (np_args, np_kwargs), _ = self._ak_to_np_(*ak_args, **ak_kwargs)
        np_rets = self._call_numpy(*np_args, **np_kwargs)
        np_rets = self._np_to_ak_.convert(np_rets)
        return self.postprocess_awkward(np_rets, *args, **kwargs)

    def _call_dask(self, *args, **kwargs):
        """
        Wrapper required for dask awkward calls.

        Here we create a new callable class (_callable_wrap) that packs the
        prepare_awkward/numpy_call/postprocess_awkward call routines to be
        passable to the dask_awkward.map_partition method.

        In addition, because map_partition by default expects the callable's
        return to be singular awkward array, we provide the additional format
        converters to translate numpy_calls that returns container of arrays.
        """

        def pack_ret_array(ret):
            """
            In case the return instance is not a singular array, we will need to
            pack the results in a way that it "looks" like a single awkward
            array up to dask.
            """
            if isinstance(ret, awkward.Array):
                return ret
            elif isinstance(ret, Dict):
                return awkward.zip(ret)
            else:
                # TODO: implement more potential containers?
                raise ValueError(f"Do not know how to pack array type {type(ret)}")

        def unpack_ret_array(ret):
            if len(ret.fields) != 0:
                # TODO: is this method robust?
                return {k: ret[k] for k in ret.fields}
            else:
                return ret

        class _callable_wrap:
            def __init__(self, inputs):
                """
                Here we need to also store the args_len and keys argument, as
                the map_partition method currently only works with *args like
                arguments. These containers are needed to properly translate the
                passed *args to a (*args, **kwargs) pair used by
                __call_awkward__.
                """
                assert len(inputs) == 2
                self.args_len = len(inputs[0])
                self.kwargs_keys = list(inputs[1].keys())
                # self.wrapper = wrapper

            def args_to_pair(self, *args) -> Tuple:
                """Converting *args to a (*args,**kwargs) pair"""
                ret_args = tuple(x for x in args[0 : self.args_len])
                ret_kwargs = {
                    k: v for k, v in zip(self.kwargs_keys, args[self.args_len :])
                }
                return ret_args, ret_kwargs

            def pair_to_args(self, *args, **kwargs) -> Tuple:
                """Converting (*args,**kwargs) pair to *args-like"""
                return [*args, *kwargs.values()]

            def get_backend(self, *args):
                for x in args:
                    if isinstance(x, awkward.Array):
                        return awkward.backend(x)
                    elif isinstance(x, dask_awkward.Array):
                        return awkward.backend(x)
                return None

            def __call__(self, wrapper, *args):
                """
                Mainly translating the input *args to the (*args, **kwarg) pair
                defined for the `__call_awkward__` method. Additional
                calculation routine defined to for the 'typetracer' backend for
                metadata scouting.
                """
                # This also touches input arrays in case of
                # type tracers, when it generates length-one
                # arrays
                ak_args, ak_kwargs = self.args_to_pair(*args)

                conv = container_converter(
                    {awkward.Array: awkward.typetracer.length_one_if_typetracer},
                    default_conv=container_converter.no_action,
                )

                (ak_args, ak_kwargs), backends = conv(*ak_args, **ak_kwargs)

                # Converting to numpy
                (np_args, np_kwargs), _ = numpy_call_wrapper._ak_to_np_(
                    *ak_args, **ak_kwargs
                )
                out = wrapper._call_numpy(*np_args, **np_kwargs)
                out = wrapper._np_to_ak_.convert(out)

                # Additional packing
                out = pack_ret_array(out)
                if "typetracer" in backends:
                    out = awkward.Array(
                        out.layout.to_typetracer(forget_length=True),
                        behavior=out.behavior,
                    )
                return out

        dak_args, dak_kwargs = self.prepare_awkward(*args, **kwargs)
        wrap = _callable_wrap((dak_args, dak_kwargs))
        packed_args = wrap.pair_to_args(*dak_args, **dak_kwargs)

        flattened_args, repack = unpack_collections(*packed_args, traverse=True)
        flattened_metas = tuple(arg._meta for arg in flattened_args)
        packed_metas = repack(flattened_metas)

        wrap_meta = wrap(self, *packed_metas)
        if not hasattr(self, "_delayed_wrapper"):
            setattr(self, "_delayed_wrapper", dask.delayed(self))
        arr = dask_awkward.lib.core.map_partitions(
            wrap,
            self._delayed_wrapper,
            *packed_args,
            label=f"numpy_call_{self.__class__.__name__}",
            meta=wrap_meta,
        )
        arr = unpack_ret_array(arr)
        return self.postprocess_awkward(arr, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        Highest level abstraction to be directly called by the user. Checks
        whether the inputs has any awkward arrays or dask_awkward arrays, and
        call the corresponding function if they are found. If no dask awkward or
        awkward arrays are found, calling the underlying _call_numpy method.
        """
        array_lib = self.get_awkward_lib(*args, **kwargs)

        if array_lib is awkward:
            return self._call_awkward(*args, **kwargs)
        elif array_lib is dask_awkward:
            return self._call_dask(*args, **kwargs)
        else:
            return self._call_numpy(*args, **kwargs)
