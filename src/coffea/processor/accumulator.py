import copy
import operator
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, MutableMapping, MutableSet
from typing import Optional, TypeVar, Union

try:
    from typing import Protocol, runtime_checkable  # type: ignore
except ImportError:
    from typing_extensions import Protocol  # type: ignore
    from typing import runtime_checkable

import awkward
import numpy

T = TypeVar("T")


@runtime_checkable
class Addable(Protocol):
    def __add__(self: T, other: T) -> T: ...


Accumulatable = Union[Addable, MutableSet, MutableMapping]


def add(a: Accumulatable, b: Accumulatable) -> Accumulatable:
    """Add two accumulatables together, without altering inputs

    This may make copies in certain situations
    """
    if isinstance(a, Addable) and isinstance(b, Addable):
        return operator.add(a, b)
    if isinstance(a, MutableSet) and isinstance(b, MutableSet):
        return operator.or_(a, b)
    elif isinstance(a, MutableMapping) and isinstance(b, MutableMapping):
        # capture type(X) by shallow copy and clear
        # since we don't know the signature of type(X).__init__
        if isinstance(b, type(a)):
            out = copy.copy(a)
        elif isinstance(a, type(b)):
            out = copy.copy(b)
        else:
            raise ValueError(
                f"Cannot add two mappings of incompatible type ({type(a)} vs. {type(b)})"
            )
        out.clear()
        lhs, rhs = set(a), set(b)
        # Keep the order of elements as far as possible
        for key in a:
            if key in rhs:
                out[key] = add(a[key], b[key])
            else:
                out[key] = copy.deepcopy(a[key])
        for key in b:
            if key not in lhs:
                out[key] = copy.deepcopy(b[key])
        return out
    raise ValueError(
        f"Cannot add accumulators of incompatible type ({type(a)} vs. {type(b)})"
    )


def iadd(a: Accumulatable, b: Accumulatable) -> Accumulatable:
    """Add two accumulatables together, assuming the first is mutable"""
    if isinstance(a, Addable) and isinstance(b, Addable):
        return operator.iadd(a, b)
    elif isinstance(a, MutableSet) and isinstance(b, MutableSet):
        return operator.ior(a, b)
    elif isinstance(a, MutableMapping) and isinstance(b, MutableMapping):
        if not isinstance(b, type(a)):
            raise ValueError(
                f"Cannot add two mappings of incompatible type ({type(a)} vs. {type(b)})"
            )
        lhs, rhs = set(a), set(b)
        # Keep the order of elements as far as possible
        for key in a:
            if key in rhs:
                a[key] = iadd(a[key], b[key])
        for key in b:
            if key not in lhs:
                a[key] = copy.deepcopy(b[key])
        return a
    raise ValueError(
        f"Cannot add accumulators of incompatible type ({type(a)} vs. {type(b)})"
    )


def accumulate(
    items: Iterable[Optional[Accumulatable]], accum: Optional[Accumulatable] = None
) -> Optional[Accumulatable]:
    gen = (x for x in items if x is not None)
    try:
        if accum is None:
            accum = next(gen)
            # we want to produce a new object so that the input is not mutated
            accum = add(accum, next(gen))
        while True:
            # subsequent additions can happen in-place, which may be more performant
            accum = iadd(accum, next(gen))
    except StopIteration:
        pass
    return accum


async def async_accumulate(result_stream):
    output = None
    async for results in result_stream:
        if output:
            output = iadd(output, results)
        else:
            output = results
        yield output


class AccumulatorABC(metaclass=ABCMeta):
    """Abstract base class for an accumulator

    Accumulators are abstract objects that enable the reduce stage of the typical map-reduce
    scaleout that we do in Coffea. One concrete example is a histogram. The idea is that an
    accumulator definition holds enough information to be able to create an empty accumulator
    (the ``identity()`` method) and add two compatible accumulators together (the ``add()`` method).
    The former is not strictly necessary, but helps with book-keeping. Here we show an example usage
    of a few accumulator types. An arbitrary-depth nesting of dictionary accumulators is supported, much
    like the behavior of directories in ROOT hadd.

    After defining an accumulator::

        from coffea.processor import dict_accumulator, column_accumulator, defaultdict_accumulator
        from coffea.hist import Hist, Bin
        import numpy as np

        adef = dict_accumulator({
            'cutflow': defaultdict_accumulator(int),
            'pt': Hist("counts", Bin("pt", "$p_T$", 100, 0, 100)),
            'final_pt': column_accumulator(np.zeros(shape=(0,))),
        })

    Notice that this function does not mutate ``adef``::

        def fill(n):
            ptvals = np.random.exponential(scale=30, size=n)
            cut = ptvals > 200.
            acc = adef.identity()
            acc['cutflow']['pt>200'] += cut.sum()
            acc['pt'].fill(pt=ptvals)
            acc['final_pt'] += column_accumulator(ptvals[cut])
            return acc

    As such, we can execute it several times in parallel and reduce the result::

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            outputs = executor.map(fill, [2000, 2000])

        combined = sum(outputs, adef.identity())


    Derived classes must implement
        - ``identity()``: returns a new object of same type as self,
          such that ``self + self.identity() == self``
        - ``add(other)``: adds an object of same type as self to self

    Concrete implementations are then provided for ``__add__``, ``__radd__``, and ``__iadd__``.
    """

    @abstractmethod
    def identity(self):
        """Identity of the accumulator

        A value such that any other value added to it will return
        the other value
        """
        pass

    @abstractmethod
    def add(self, other):
        """Add another accumulator to this one in-place"""
        pass

    def __add__(self, other):
        ret = self.identity()
        ret.add(self)
        ret.add(other)
        return ret

    def __radd__(self, other):
        ret = self.identity()
        ret.add(other)
        ret.add(self)
        return ret

    def __iadd__(self, other):
        self.add(other)
        return self


class value_accumulator(AccumulatorABC):
    """Holds a value of arbitrary type

    Parameters
    ----------
        default_factory : callable
            a function that returns an instance of the desired identity value
        initial : bool, optional
            an initial value, if the identity is not the desired initial value
    """

    def __init__(self, default_factory, initial=None):
        self.value = default_factory() if initial is None else initial
        self.default_factory = default_factory

    def __repr__(self):
        if type(self.default_factory) is type:
            defrepr = self.default_factory.__name__
        else:
            defrepr = repr(self.default_factory)
        return f"value_accumulator({defrepr}, {self.value!r})"

    def identity(self):
        return value_accumulator(self.default_factory)

    def add(self, other):
        if isinstance(other, value_accumulator):
            self.value = self.value + other.value
        else:
            self.value = self.value + other


class list_accumulator(list, AccumulatorABC):
    """A list with accumulator semantics

    See `list` for further info
    """

    def identity(self):
        return list()

    def add(self, other):
        """Add another accumulator to this one in-place"""
        if isinstance(other, list):
            list.extend(self, other)
        else:
            raise ValueError


class set_accumulator(set, AccumulatorABC):
    """A set with accumulator semantics

    See `set` for further info
    """

    def identity(self):
        return set_accumulator()

    def add(self, other):
        """Add another accumulator to this one in-place

        Note
        ----
        This replaces `set.add` behavior, unfortunately.
        A workaround is to use `set.update`, e.g. ``a.update({'val'})``
        """
        if isinstance(other, MutableSet):
            set.update(self, other)
        else:
            set.add(self, other)


class dict_accumulator(dict, AccumulatorABC):
    """A dictionary with accumulator semantics

    See `dict` for further info.
    It is assumed that the contents of the dict have accumulator semantics.
    """

    def identity(self):
        ret = dict_accumulator()
        for key, value in self.items():
            ret[key] = value.identity()
        return ret

    def add(self, other):
        if isinstance(other, MutableMapping):
            for key, value in other.items():
                if key not in self:
                    if isinstance(value, AccumulatorABC):
                        self[key] = value.identity()
                    else:
                        raise ValueError
                self[key] += value
        else:
            raise ValueError


class defaultdict_accumulator(defaultdict, AccumulatorABC):
    """A defaultdict with accumulator semantics

    See `collections.defaultdict` for further info.
    It is assumed that the contents of the dict have accumulator semantics
    """

    def identity(self):
        return defaultdict_accumulator(self.default_factory)

    def add(self, other):
        for key, value in other.items():
            self[key] += value


class column_accumulator(AccumulatorABC):
    """An appendable numpy.ndarray or ak.Array

    Parameters
    ----------
        value : numpy.ndarray or ak.Array
            The identity value array, which should be an empty ndarray
            with the desired row shape. The column dimension will correspond to
            the first index of `value` shape.

    Examples
    --------
    If a set of accumulators is defined as::

        a = column_accumulator(np.array([]))
        b = column_accumulator(np.array([1., 2., 3.]))
        c = column_accumulator(np.array([4., 5., 6.]))

    then:

    >>> a + b
    column_accumulator(array([1., 2., 3.]))
    >>> c + b + a
    column_accumulator(array([4., 5., 6., 1., 2., 3.]))
    """

    def __init__(self, value):
        if isinstance(value, numpy.ndarray):
            self._empty = numpy.empty(dtype=value.dtype, shape=(0,) + value.shape[1:])
        elif isinstance(value, awkward.Array):
            self._empty = awkward.Array(
                value.layout.form.length_zero_array(), behavior=value.behavior
            )
        else:
            raise ValueError(
                "column_accumulator only works with numpy arrays or awkward arrays"
            )
        self._value = value

    def __repr__(self):
        return "column_accumulator(%r)" % self.value

    def identity(self):
        return column_accumulator(self._empty)

    def add(self, other):
        if not isinstance(other, column_accumulator):
            raise ValueError("column_accumulator cannot be added to %r" % type(other))
        if isinstance(self._empty, numpy.ndarray) and isinstance(
            other._empty, awkward.Array
        ):
            raise ValueError(
                "Cannot add column_accumulator with numpy.ndarray to one with awkward.Array"
            )
        if isinstance(self._empty, awkward.Array) and isinstance(
            other._empty, numpy.ndarray
        ):
            raise ValueError(
                "Cannot add column_accumulator with awkward.Array to one with numpy.ndarray"
            )
        if isinstance(self._empty, numpy.ndarray) and isinstance(
            other._empty, numpy.ndarray
        ):
            if self._empty.shape != other._empty.shape:
                raise ValueError(
                    "Cannot add two column_accumulator objects of dissimilar shape (%r vs %r)"
                    % (self._empty.shape, other._empty.shape)
                )
        if isinstance(self._empty, awkward.Array) and isinstance(
            other._empty, awkward.Array
        ):
            if self._empty.ndim != other._empty.ndim:
                raise ValueError(
                    "Cannot add two column_accumulator objects of dissimilar dimensionality (%r vs %r)"
                    % (self._empty.ndim, other._empty.ndim)
                )
        self._value = numpy.concatenate((self._value, other._value), axis=0)

    @property
    def value(self):
        """The current value of the column

        Returns a numpy array where the first dimension is the column dimension
        """
        return self._value
