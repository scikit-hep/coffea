"""Basic NanoEvents and NanoCollection mixins"""
import awkward
import numpy
from abc import abstractmethod
from functools import partial
import re
from coffea.util import rewrap_recordarray, awkward_rewrap
from typing import (
    List,
    Union,
    Tuple,
    Callable,
)


behavior = {}


@awkward.mixin_class(behavior)
class Systematic:
    """A base mixin class to describe and build variations on a feature of an nanoevents object."""

    _systematic_kinds = set()

    @classmethod
    def add_kind(cls, kind: str):
        """
        Register a type of systematic variation, it must fullfil the base class interface.
        """
        cls._systematic_kinds.add(kind)

    def _ensure_systematics(self):
        """
        Make sure that the parent object always has a field called '__systematics__'.
        """
        if "__systematics__" not in awkward.fields(self):
            self["__systematics__"] = awkward.layout.RecordArray([], [], len(self))

    @property
    def systematics(self):
        """
        Return the list of all systematics attached to this object.
        """
        regex = re.compile(r"\_{2}.*\_{2}")
        self._ensure_systematics()
        fields = [
            f for f in awkward.fields(self["__systematics__"]) if not regex.match(f)
        ]
        return self["__systematics__"][fields]

    @abstractmethod
    def _build_variations(
        self,
        name: str,
        what: Union[str, List[str], Tuple[str]],
        varying_function: Callable,
    ):
        """
        name: str, name of the systematic variation / uncertainty source
        what: Union[str, List[str], Tuple[str]], name what gets varied,
              this could be a list or tuple of column names
        varying_function: Union[function, bound method, partial], a function that describes how 'what' is varied
        define how to manipulate the output of varying_function to produce all systematic variations. Varying function
        must close over all non-event-data arguments.
        """
        pass

    @abstractmethod
    def explodes_how(self):
        """
        This describes how a systematic uncertainty needs to be evaluated in the context of other systematic uncertainties.
        i.e. Do you iterate over this keeping all others fixed or do you need to have correlations with other (subsets of) systematics.
        """
        # this function contains decades of thinking about iterate over systematics variations
        # your opinions about systematics go here. :D
        pass

    @abstractmethod
    def describe_variations(self):
        """returns a list of variation names"""
        pass

    def add_systematic(
        self,
        name: str,
        kind: str,
        what: Union[str, List[str], Tuple[str]],
        varying_function: Callable,
    ):
        """
        name: str, name of the systematic variation / uncertainty source
        kind: str, the name of the kind of systematic variation
        what: Union[str, List[str], Tuple[str]], name what gets varied,
               this could be a list or tuple of column names
        varying_function: Union[function, bound method], a function that describes how 'what' is varied, it must close over all non-event-data arguments.
        """
        self._ensure_systematics()

        if name in awkward.fields(self["__systematics__"]):
            raise ValueError(f"{name} already exists as a systematic for this object!")

        if kind not in self._systematic_kinds:
            raise ValueError(
                f"{kind} is not an available systematics type, please add it and try again!"
            )

        wrap = partial(
            awkward_rewrap, like_what=self["__systematics__"], gfunc=rewrap_recordarray
        )
        flat = self if self.ndim == 1 else awkward.flatten(self)

        if what == "weight" and "__ones__" not in awkward.fields(
            flat["__systematics__"]
        ):
            flat["__systematics__", "__ones__"] = numpy.ones(
                len(flat), dtype=numpy.float32
            )

        rendered_type = flat.layout.parameters["__record__"]
        as_syst_type = awkward.with_parameter(flat, "__record__", kind)
        as_syst_type._build_variations(name, what, varying_function)
        variations = as_syst_type.describe_variations()

        flat["__systematics__", name] = awkward.zip(
            {
                v: getattr(as_syst_type, v)(name, what, rendered_type)
                for v in variations
            },
            depth_limit=1,
            with_name=f"{name}Systematics",
        )

        self["__systematics__"] = wrap(flat["__systematics__"])
        self.behavior[("__typestr__", f"{name}Systematics")] = f"{kind}"


behavior[("__typestr__", "Systematic")] = "Systematic"

# initialize all systematic variation types
from coffea.nanoevents.methods import systematics

for kind in systematics.__all__:
    Systematic.add_kind(kind)


@awkward.mixin_class(behavior)
class NanoEvents(Systematic):
    """NanoEvents mixin class

    This mixin class is used as the top-level type for NanoEvents objects.
    """

    @property
    def metadata(self):
        """Arbitrary metadata"""
        return self.layout.purelist_parameter("metadata")


behavior[("__typestr__", "NanoEvents")] = "event"


@awkward.mixin_class(behavior)
class NanoCollection:
    """A NanoEvents collection

    This mixin provides some helper methods useful for creating cross-references
    and other advanced mixin types.
    """

    def _collection_name(self):
        """The name of the collection (i.e. the field under events where it is found)"""
        return self.layout.purelist_parameter("collection_name")

    def _getlistarray(self):
        """Do some digging to find the initial listarray"""

        def descend(layout, depth):
            islistarray = isinstance(
                layout,
                (awkward.layout.ListOffsetArray32, awkward.layout.ListOffsetArray64),
            )
            if islistarray and layout.content.parameter("collection_name") is not None:
                return lambda: layout

        return awkward._util.recursively_apply(self.layout, descend)

    def _content(self):
        """Internal method to get jagged collection content

        This should only be called on the original unsliced collection array.
        Used with global indexes to resolve cross-references"""
        return self._getlistarray().content

    def _apply_global_index(self, index):
        """Internal method to take from a collection using a flat index

        This is often necessary to be able to still resolve cross-references on
        reduced arrays or single records.
        """
        if isinstance(index, int):
            out = self._content()[index]
            return awkward.Record(out, behavior=self.behavior)

        def flat_take(layout):
            idx = awkward.Array(layout)
            return self._content()[idx.mask[idx >= 0]]

        def descend(layout, depth):
            if layout.purelist_depth == 1:
                return lambda: flat_take(layout)

        (index,) = awkward.broadcast_arrays(index)
        out = awkward._util.recursively_apply(index.layout, descend)
        return awkward.Array(out, behavior=self.behavior)

    def _events(self):
        """Internal method to get the originally-constructed NanoEvents

        This can be called at any time from any collection, as long as
        the NanoEventsFactory instance exists."""
        return self.behavior["__events_factory__"].events()


__all__ = ["NanoCollection", "NanoEvents", "Systematic"]
