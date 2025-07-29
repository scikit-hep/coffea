"""Tools of general use for columnar analysis

These helper classes were previously part of ``coffea.processor``
but have been migrated and updated to be compatible with awkward-array 1.0
"""

import warnings
from collections import namedtuple
from functools import lru_cache

import awkward
import dask.array
import dask_awkward
import hist
import hist.dask
import numpy
from dask_awkward.lib.core import compatible_partitions
from dask_awkward.utils import IncompatiblePartitions

import coffea.processor
import coffea.util


def _generate_slices(array_length, max_elements=128):
    """Generate slices to split an array into chunks of at most `max_elements` elements

    Parameters
    ----------
    array_length : int
        The length of the array to split
    max_elements : int, optional
        The maximum number of elements in each chunk. Default is 128.

    Returns
    -------
    slices : list of slice objects
        A list of slice objects to iterate over and split the array into chunks with at most `max_elements` elements per slice
    """
    slices = []
    for start in range(0, array_length, max_elements):
        end = min(start + max_elements, array_length)
        slices.append(slice(start, end))
    return slices


def boolean_masks_to_categorical_integers(
    masks,
    insert_unmasked_as_zeros=False,
    insert_commonmask_as_zeros=None,
    return_mask=False,
):
    """Converts a list of boolean masks to irregular arrays of enumerated categorical integers

    Parameters
    ----------
    masks : list of boolean numpy.ndarray, awkward.Array or dask_awkward.lib.core.Array objects
        The boolean mask arrays to convert to categorical integers
    insert_unmasked_as_zeros : bool, optional
        Whether to insert a zero entry representing an 'unmasked' state, equivalent to the first mask satisfying `ak.all(mask == True)`. Default is False.
    insert_commonmask_as_zeros : boolean numpy.ndarray, awkward.Array, or dask_awkward.lib.core.Array, optional
        If not None, insert a zero entry representing a 'commonmasked' state. Default is None. Not compatible with insert_unmasked_as_zeros=True.
    return_mask : bool, optional
        Whether to return the intermediate concatenated mask array instead of the ragged array of categorical integers. Default is False.

    Returns
    -------
    irregular_categories : awkward.Array or dask_awkward.lib.core.Array containing integers representing whether an entry contained a True value in the corresponding mask

        >>> pt = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6]])
        >>> at_least_one = (ak.num(pt, axis=1) >= 1)
        >>> at_least_two = (ak.num(pt, axis=1) >= 2)
        >>> at_least_three = (ak.num(pt, axis=1) >= 3)
        >>> something_over_four = ak.any(pt > 4, axis=1)
        >>> masks = [at_least_one, at_least_two, at_least_three, something_over_four]

        >>> print(boolean_masks_to_categorical_integers(masks, insert_unmasked_as_zeros=False, return_mask=False))
        [[0, 1, 2], [], [0, 1, 3], [0, 3]]
        >>> print(boolean_masks_to_categorical_integers(masks, insert_unmasked_as_zeros=False, return_mask=True))
        [[True, True, True, False], [False, ...], ..., [True, False, False, True]]
        >>> print(boolean_masks_to_categorical_integers(masks, insert_unmasked_as_zeros=True, return_mask=False))
        [[0, 1, 2, 3], [0], [0, 1, 2, 4], [0, 1, 4]]
        >>> print(boolean_masks_to_categorical_integers(masks, insert_unmasked_as_zeros=True, return_mask=True))
        [[True, True, True, True, False], [...], ..., [True, True, False, False, True]]
    """
    mask_inputs = [mask[:, None] for mask in masks]
    if insert_unmasked_as_zeros and insert_commonmask_as_zeros is not None:
        raise ValueError(
            "insert_unmasked_as_zeros and insert_commonmask_as_zeros cannot be used together"
        )
    if insert_unmasked_as_zeros:
        mask_inputs.insert(0, awkward.ones_like(mask_inputs[0], dtype=bool))
    if insert_commonmask_as_zeros is not None:
        mask_inputs.insert(0, insert_commonmask_as_zeros[:, None])
    irregular_masks = []
    # TODO: _generate_slices is used to work around the issue addressed in awkward PR https://github.com/scikit-hep/awkward/pull/3312
    # which was merged in awkward v2.7.2 (https://github.com/scikit-hep/awkward/releases/tag/v2.7.2) and this can be removed when it becomes the minimum version for coffea
    for slc in _generate_slices(len(mask_inputs), max_elements=128):
        # create subarrays of the masks to concatenate, to work around issue prior to awkward v2.7.2
        irregular_masks.append(
            awkward.from_regular(awkward.concatenate(mask_inputs[slc], axis=1), axis=1)
        )
    if len(irregular_masks) == 1:
        # unwrap the new concatenated (irregular) masks if there is only one
        irregular_mask = irregular_masks[0]
    else:
        # if multiple irregular masks were created, concatenate them a final time
        irregular_mask = awkward.concatenate(irregular_masks, axis=1)
    if return_mask:
        return irregular_mask
    # convert the boolean masks to categorical integers by calling local index and remove elements whose mask entry was false
    irregular_values = awkward.local_index(irregular_mask, axis=1)[irregular_mask]
    return irregular_values


class WeightStatistics:
    """
    Container for statistics about the weight, including the sum of squared weights
    and number of entries.

    Parameters
    ----------
        sumw: float
            The sum of weights
        sumw2: float
            The sum of squared weights
        minw: float
            The minimum weight
        maxw: float
            The maximum weight
        n: int
            The number of entries
    """

    def __init__(self, sumw=0.0, sumw2=0.0, minw=numpy.inf, maxw=-numpy.inf, n=0):
        self.sumw = sumw
        self.sumw2 = sumw2
        self.minw = minw
        self.maxw = maxw
        self.n = n

    def __repr__(self):
        return f"WeightStatistics(sumw={self.sumw}, sumw2={self.sumw2}, minw={self.minw}, maxw={self.maxw}, n={self.n})"

    def identity(self):
        return WeightStatistics()

    def add(self, other):
        """Add two WeightStatistics objects together.

        Adds the sum of weights, the sum of squared weights, and the number of entries.
        Takes the minimum and maximum across the two WeightStatistics objects. Modifies
        this object in place.

        Parameters
        ----------
            other: WeightStatistics
                The other WeightStatistics object to add to this one
        """
        self.sumw += other.sumw
        self.sumw2 += other.sumw2
        self.minw = min(self.minw, other.minw)
        self.maxw = max(self.maxw, other.maxw)
        self.n += other.n

    def __add__(self, other):
        temp = WeightStatistics(self.sumw, self.sumw2, self.minw, self.maxw, self.n)
        return temp.add(other)

    def __iadd__(self, other):
        return self.add(other)


class Weights:
    """Container for event weights and associated systematic shifts

    This container keeps track of correction factors and systematic
    effects that can be encoded as multiplicative modifiers to the event weight.
    All weights are stored in vector form.

    Parameters
    ----------
        size : int | None
            size of the weight arrays to be handled (i.e. the number of events / instances).
            If None then we expect to operate in delayed mode.
        storeIndividual : bool, optional
            store not only the total weight + variations, but also each individual weight.
            Default is false.
    """

    def __init__(self, size, storeIndividual=False):
        self._weight = None if size is None else numpy.ones(size)
        self._weights = {}
        self._modifiers = {}
        self._weightStats = {}
        self._storeIndividual = storeIndividual
        self._names = []

    @property
    def weightStatistics(self):
        """Statistics about the weight, including the sum of squared weights
        and number of entries."""
        return self._weightStats

    def __add_eager(self, name, weight, weightUp, weightDown, shift):
        """Add a new weight with eager calculation"""
        if isinstance(weight, numpy.ma.MaskedArray):
            # TODO what to do with option-type? is it representative of unknown weight
            # and we default to one or is it an invalid weight and we should never use this
            # event in the first place (0) ?
            weight = weight.filled(1.0)
        self._weight = self._weight * weight
        if self._storeIndividual:
            self._weights[name] = weight
        self.__add_variation(name, weight, weightUp, weightDown, shift)
        if weight.size == 0:
            dtype = weight.dtype
            if dtype in (
                numpy.int8,
                numpy.int16,
                numpy.int32,
                numpy.int64,
                numpy.uint8,
                numpy.uint16,
                numpy.uint32,
                numpy.uint64,
            ):
                min = numpy.iinfo(dtype).max
                max = numpy.iinfo(dtype).min
            else:
                min = numpy.inf
                max = -numpy.inf
        else:
            min = weight.min()
            max = weight.max()
        self._weightStats[name] = WeightStatistics(
            weight.sum(),
            (weight**2).sum(),
            min,
            max,
            weight.size,
        )
        self._names.append(name)

    def __add_delayed(self, name, weight, weightUp, weightDown, shift):
        """Add a new weight with delayed calculation"""
        if isinstance(dask_awkward.type(weight), awkward.types.OptionType):
            # TODO what to do with option-type? is it representative of unknown weight
            # and we default to one or is it an invalid weight and we should never use this
            # event in the first place (0) ?
            weight = dask_awkward.fill_none(weight, 1.0)
        if self._weight is None:
            self._weight = weight
        else:
            self._weight = self._weight * weight
        if self._storeIndividual:
            self._weights[name] = weight
        self.__add_variation(name, weight, weightUp, weightDown, shift)
        self._weightStats[name] = {
            "sumw": dask_awkward.sum(weight),
            "sumw2": dask_awkward.sum(weight**2),
            "minw": dask_awkward.min(weight, mask_identity=False),
            "maxw": dask_awkward.max(weight, mask_identity=False),
            "n": dask_awkward.num(weight, axis=0),
        }
        self._names.append(name)

    def add(self, name, weight, weightUp=None, weightDown=None, shift=False):
        """Add a new weight

        Adds a named correction to the event weight, and optionally also associated
        systematic uncertainties.

        Parameters
        ----------
            name : str
                name of correction
            weight : numpy.ndarray
                the nominal event weight associated with the correction
            weightUp : numpy.ndarray, optional
                weight with correction uncertainty shifted up (if available)
            weightDown : numpy.ndarray, optional
                weight with correction uncertainty shifted down. If ``weightUp`` is supplied, and
                the correction uncertainty is symmetric, this can be set to None to auto-calculate
                the down shift as ``1 / weightUp``.
            shift : bool, optional
                if True, interpret weightUp and weightDown as a relative difference (additive) to the
                nominal value

        .. note:: ``weightUp`` and ``weightDown`` are assumed to be rvalue-like and may be modified in-place by this function
        """
        if name in self._names:
            raise ValueError(f"Weight '{name}' already exists")
        if name.endswith("Up") or name.endswith("Down"):
            raise ValueError(
                "Avoid using 'Up' and 'Down' in weight names, instead pass appropriate shifts to add() call"
            )
        weight = coffea.util._ensure_flat(weight, allow_missing=True)
        if isinstance(weight, numpy.ndarray) and isinstance(
            self._weight, numpy.ndarray
        ):
            self.__add_eager(name, weight, weightUp, weightDown, shift)
        elif isinstance(weight, dask_awkward.Array) and isinstance(
            self._weight, (dask_awkward.Array, type(None))
        ):
            self.__add_delayed(name, weight, weightUp, weightDown, shift)
        else:
            raise ValueError(
                f"Incompatible weights: self._weight={type(self._weight)}, weight={type(weight)}"
            )

    def __add_multivariation_eager(
        self, name, weight, modifierNames, weightsUp, weightsDown, shift=False
    ):
        """Add a new weight with multiple variations in eager mode"""
        if isinstance(weight, numpy.ma.MaskedArray):
            # TODO what to do with option-type? is it representative of unknown weight
            # and we default to one or is it an invalid weight and we should never use this
            # event in the first place (0) ?
            weight = weight.filled(1.0)
        self._weight = self._weight * weight
        if self._storeIndividual:
            self._weights[name] = weight
        # Now loop on the variations
        if len(modifierNames) > 0:
            if len(modifierNames) != len(weightsUp) or len(modifierNames) != len(
                weightsDown
            ):
                raise ValueError(
                    "Provide the same number of modifier names related to the list of modified weights"
                )
        for modifier, weightUp, weightDown in zip(
            modifierNames, weightsUp, weightsDown
        ):
            systName = f"{name}_{modifier}"
            self.__add_variation(systName, weight, weightUp, weightDown, shift)
        if weight.size == 0:
            dtype = weight.dtype
            if dtype in (
                numpy.int8,
                numpy.int16,
                numpy.int32,
                numpy.int64,
                numpy.uint8,
                numpy.uint16,
                numpy.uint32,
                numpy.uint64,
            ):
                min = numpy.iinfo(dtype).max
                max = numpy.iinfo(dtype).min
            else:
                min = numpy.inf
                max = -numpy.inf
        else:
            min = weight.min()
            max = weight.max()
        self._weightStats[name] = WeightStatistics(
            weight.sum(),
            (weight**2).sum(),
            min,
            max,
            weight.size,
        )
        self._names.append(name)

    def __add_multivariation_delayed(
        self, name, weight, modifierNames, weightsUp, weightsDown, shift=False
    ):
        """Add a new weight with multiple variations in delayed mode"""
        if isinstance(weight, awkward.types.OptionType):
            # TODO what to do with option-type? is it representative of unknown weight
            # and we default to one or is it an invalid weight and we should never use this
            # event in the first place (0) ?
            weight = dask_awkward.fill_none(weight, 1.0)
        if self._weight is None:
            self._weight = weight
        else:
            self._weight = self._weight * weight
        if self._storeIndividual:
            self._weights[name] = weight
        # Now loop on the variations
        if len(modifierNames) > 0:
            if len(modifierNames) != len(weightsUp) or len(modifierNames) != len(
                weightsDown
            ):
                raise ValueError(
                    "Provide the same number of modifier names related to the list of modified weights"
                )
        for modifier, weightUp, weightDown in zip(
            modifierNames, weightsUp, weightsDown
        ):
            systName = f"{name}_{modifier}"
            self.__add_variation(systName, weight, weightUp, weightDown, shift)
        self._weightStats[name] = {
            "sumw": dask_awkward.sum(weight),
            "sumw2": dask_awkward.sum(weight**2),
            "minw": dask_awkward.min(weight, mask_identity=False),
            "maxw": dask_awkward.max(weight, mask_identity=False),
            "n": dask_awkward.num(weight, axis=0),
        }
        self._names.append(name)

    def add_multivariation(
        self, name, weight, modifierNames, weightsUp, weightsDown, shift=False
    ):
        """Add a new weight with multiple variations

        Each variation of a single weight is given a different modifier name.
        This is particularly useful e.g. for btag SF variations.

        Parameters
        ----------
            name : str
                name of correction
            weight : numpy.ndarray
                the nominal event weight associated with the correction
            modifierNames: list of str
                list of modifiers for each set of weights variation
            weightsUp : list of numpy.ndarray
                weight with correction uncertainty shifted up (if available)
            weightsDown : list of numpy.ndarray
                weight with correction uncertainty shifted down. If ``weightUp`` is supplied, and
                the correction uncertainty is symmetric, this can be set to None to auto-calculate
                the down shift as ``1 / weightUp``.
            shift : bool, optional
                if True, interpret weightUp and weightDown as a relative difference (additive) to the
                nominal value

        .. note:: ``weightUp`` and ``weightDown`` are assumed to be rvalue-like and may be modified in-place by this function
        """
        if name in self._names:
            raise ValueError(f"Weight '{name}' already exists")
        if name.endswith("Up") or name.endswith("Down"):
            raise ValueError(
                "Avoid using 'Up' and 'Down' in weight names, instead pass appropriate shifts to add() call"
            )
        weight = coffea.util._ensure_flat(weight, allow_missing=True)
        if isinstance(weight, numpy.ndarray) and isinstance(
            self._weight, numpy.ndarray
        ):
            self.__add_multivariation_eager(
                name, weight, modifierNames, weightsUp, weightsDown, shift
            )
        elif isinstance(weight, dask_awkward.Array) and isinstance(
            self._weight, (dask_awkward.Array, type(None))
        ):
            self.__add_multivariation_delayed(
                name, weight, modifierNames, weightsUp, weightsDown, shift
            )
        else:
            raise ValueError(
                f"Incompatible weights: self._weight={type(self._weight)}, weight={type(weight)}"
            )

    def __add_variation_eager(self, name, weight, weightUp, weightDown, shift):
        """Helper function to add an eagerly calculated weight variation."""
        if weightUp is not None:
            weightUp = coffea.util._ensure_flat(weightUp, allow_missing=True)
            if isinstance(weightUp, numpy.ma.MaskedArray):
                weightUp = weightUp.filled(1.0)
            if shift:
                weightUp += weight
            weightUp[weight != 0.0] /= weight[weight != 0.0]
            self._modifiers[name + "Up"] = weightUp
        if weightDown is not None:
            weightDown = coffea.util._ensure_flat(weightDown, allow_missing=True)
            if isinstance(weightDown, numpy.ma.MaskedArray):
                weightDown = weightDown.filled(1.0)
            if shift:
                weightDown = weight - weightDown
            weightDown[weight != 0.0] /= weight[weight != 0.0]
            self._modifiers[name + "Down"] = weightDown

    def __add_variation_delayed(self, name, weight, weightUp, weightDown, shift):
        """Helper function to add a delayed-calculation weight variation."""
        if weightUp is not None:
            weightUp = coffea.util._ensure_flat(weightUp, allow_missing=True)
            if isinstance(dask_awkward.type(weightUp), awkward.types.OptionType):
                weightUp = dask_awkward.fill_none(weightUp, 1.0)
            if shift:
                weightUp = weightUp + weight
            weightUp = dask_awkward.where(weight != 0.0, weightUp / weight, weightUp)
            self._modifiers[name + "Up"] = weightUp
        if weightDown is not None:
            weightDown = coffea.util._ensure_flat(weightDown, allow_missing=True)
            if isinstance(dask_awkward.type(weightDown), awkward.types.OptionType):
                weightDown = dask_awkward.fill_none(weightDown, 1.0)
            if shift:
                weightDown = weight - weightDown
            weightDown = dask_awkward.where(
                weight != 0.0, weightDown / weight, weightDown
            )
            self._modifiers[name + "Down"] = weightDown

    def __add_variation(
        self, name, weight, weightUp=None, weightDown=None, shift=False
    ):
        """Helper function to add a weight variation.

        Parameters
        ----------
            name : str
                name of systematic variation (just the name of the weight if only
                one variation is added, or `name_syst` for multiple variations)
            weight : numpy.ndarray
                the nominal event weight associated with the correction
            weightUp : numpy.ndarray, optional
                weight with correction uncertainty shifted up (if available)
            weightDown : numpy.ndarray, optional
                weight with correction uncertainty shifted down. If ``weightUp`` is supplied, and
                the correction uncertainty is symmetric, this can be set to None to auto-calculate
                the down shift as ``1 / weightUp``.
            shift : bool, optional
                if True, interpret weightUp and weightDown as a relative difference (additive) to the
                nominal value

        .. note:: ``weightUp`` and ``weightDown`` are assumed to be rvalue-like and may be modified in-place by this function
        """
        if isinstance(weight, numpy.ndarray):
            self.__add_variation_eager(name, weight, weightUp, weightDown, shift)
        elif isinstance(weight, dask_awkward.Array):
            self.__add_variation_delayed(name, weight, weightUp, weightDown, shift)

    def weight(self, modifier=None):
        """Returns the current event weight vector

        Parameters
        ----------
            modifier : str, optional
                if supplied, provide event weight corresponding to a particular
                systematic uncertainty shift, of form ``str(name + 'Up')`` or (Down)

        Returns
        -------
            weight : numpy.ndarray
                The weight vector, possibly modified by the effect of a given systematic variation.
        """
        if modifier is None:
            return self._weight
        elif "Down" in modifier and modifier not in self._modifiers:
            return self._weight / self._modifiers[modifier.replace("Down", "Up")]
        return self._weight * self._modifiers[modifier]

    def partial_weight(self, include=[], exclude=[], modifier=None):
        """Partial event weight vector

        Return a partial weight by multiplying a subset of all weights.
        Can be operated either by specifying weights to include or
        weights to exclude, but not both at the same time. The method
        can only be used if the individual weights are stored via the
        ``storeIndividual`` argument in the `Weights` initializer.


        Parameters
        ----------
            include : list
                Weight names to include, defaults to []
            exclude : list
                Weight names to exclude, defaults to []
            modifier : str, optional
                if supplied, provide event weight corresponding to a particular
                systematic uncertainty shift, of form ``str(name + 'Up')`` or (Down)
        Returns
        -------
            weight : numpy.ndarray
                The weight vector, corresponding to only the effect of the
                corrections specified.
        """
        return self._partial_weight(
            include=tuple(include), exclude=tuple(exclude), modifier=modifier
        )

    def _partial_weight(self, include, exclude, modifier=None):
        if not self._storeIndividual:
            raise ValueError(
                "To be able to request weight exclusion, use storeIndividual=True when creating Weights object."
            )
        if (include and exclude) or not (include or exclude):
            raise ValueError(
                "Need to specify exactly one of the 'exclude' or 'include' arguments."
            )

        names = set(self._weights.keys())
        if include:
            names = names & set(include)
        if exclude:
            names = names - set(exclude)

        w = None
        if isinstance(self._weight, numpy.ndarray):
            w = numpy.ones(self._weight.size)
        elif isinstance(self._weight, dask_awkward.Array):
            w = dask_awkward.ones_like(self._weight)

        for name in names:
            w = w * self._weights[name]

        if modifier is None:
            return w
        elif modifier.replace("Down", "").replace("Up", "") not in names:
            raise ValueError(
                f"Modifier {modifier} is not in the list of included weights"
            )
        elif "Down" in modifier and modifier not in self._modifiers:
            return w / self._modifiers[modifier.replace("Down", "Up")]
        return w * self._modifiers[modifier]

    @property
    def variations(self):
        """List of available modifiers"""
        keys = set(self._modifiers.keys())
        # add any missing 'Down' variation
        for k in self._modifiers.keys():
            keys.add(k.replace("Up", "Down"))
        return keys


class NminusOneToNpz:
    """Object to be returned by NminusOne.to_npz()"""

    def __init__(self, file, labels, nev, masks, saver):
        self._file = file
        self._labels = labels
        self._nev = nev
        self._masks = masks
        self._saver = saver

    def __repr__(self):
        return f"NminusOneToNpz(file={self._file}), labels={self._labels})"

    @property
    def file(self):
        return self._file

    @property
    def labels(self):
        return self._labels

    @property
    def nev(self):
        return self._nev

    @property
    def masks(self):
        return self._masks

    def compute(self):
        self._nev = list(dask.compute(*self._nev))
        self._masks = list(dask.compute(*self._masks))
        self._saver(self._file, labels=self._labels, nev=self._nev, masks=self._masks)


class CutflowToNpz:
    """Object to be returned by Cutflow.to_npz()

    Parameters
    ----------
        includeweights : bool, optional
            Whether to include the weights in the saved npz file. Default is None, which includes the weights if the Cutflow was instantiated with weights
            and excludes them otherwise.
    """

    def __init__(
        self,
        file,
        labels,
        nevonecut,
        nevcutflow,
        masksonecut,
        maskscutflow,
        saver,
        commonmask=None,
        wgtevonecut=None,
        wgtevcutflow=None,
        weights=None,
        weightsmodifier=None,
        includeweights=None,
    ):
        self._file = file
        self._labels = labels
        self._nevonecut = nevonecut
        self._nevcutflow = nevcutflow
        self._masksonecut = masksonecut
        self._maskscutflow = maskscutflow
        self._saver = saver
        self._commonmask = commonmask
        self._wgtevonecut = wgtevonecut
        self._wgtevcutflow = wgtevcutflow
        self._weights = weights if includeweights is not False else None
        self._weightsmodifier = weightsmodifier
        self._commonmasked = self.commonmask is not None
        self._weighted = (self._wgtevonecut is not None) and (
            self._wgtevcutflow is not None
        )

    def __repr__(self):
        return f"CutflowToNpz(file={self._file}), labels={self._labels}, commonmasked={self._commonmasked}, weighted={self._weighted}, weightsmodifier={self._weightsmodifier})"

    @property
    def file(self):
        return self._file

    @property
    def labels(self):
        return self._labels

    @property
    def nevonecut(self):
        return self._nevonecut

    @property
    def nevcutflow(self):
        return self._nevcutflow

    @property
    def masksonecut(self):
        return self._masksonecut

    @property
    def maskscutflow(self):
        return self._maskscutflow

    @property
    def commonmask(self):
        return self._commonmask

    @property
    def wgtevonecut(self):
        return self._wgtevonecut

    @property
    def wgtevcutflow(self):
        return self._wgtevcutflow

    @property
    def weights(self):
        return self._weights

    @property
    def weightsmodifier(self):
        return self._weightsmodifier

    def compute(self):
        # Weights has no compute method, ergo it will pass through uncomputed, i.e. as a delayed object
        # self._weights = list(self._weights) if isinstance(self._weights, (tuple, list)) else self._weights
        (
            self._nevonecut,
            self._nevcutflow,
            self._commonmask,
            self._wgtevonecut,
            self._wgtevcutflow,
            self._masksonecut,
            self._maskscutflow,
            self._weights_wmodifier,
        ) = dask.compute(
            self._nevonecut,
            self._nevcutflow,
            self._commonmask,
            self._wgtevonecut,
            self._wgtevcutflow,
            self._masksonecut,
            self._maskscutflow,
            (
                self._weights.weight(self._weightsmodifier)
                if self._weights is not None
                else None
            ),
        )
        self._nevonecut = list(self._nevonecut)
        self._nevcutflow = list(self._nevcutflow)
        self._masksonecut = list(self._masksonecut)
        self._maskscutflow = list(self._maskscutflow)
        self._commonmask = list(self._commonmask) if self._commonmasked else None
        self._wgtevonecut = list(self._wgtevonecut) if self._weighted else None
        self._wgtevcutflow = list(self._wgtevcutflow) if self._weighted else None
        self._weights_wmodifier = (
            list(self._weights_wmodifier) if self._weights is not None else None
        )
        to_save = {
            "labels": self._labels,
            "nevonecut": self._nevonecut,
            "nevcutflow": self._nevcutflow,
            "masksonecut": self._masksonecut,
            "maskscutflow": self._maskscutflow,
        }
        if self._commonmask is not None:
            to_save["commonmask"] = self._commonmask
        if self._weighted:
            to_save["wgtevonecut"] = self._wgtevonecut
            to_save["wgtevcutflow"] = self._wgtevcutflow
        if self._weights is not None:
            to_save["weights"] = self._weights_wmodifier
        self._saver(self._file, **to_save)


class NminusOne:
    """Object to be returned by PackedSelection.nminusone()"""

    def __init__(self, names, nev, masks, delayed_mode):
        self._names = names
        self._nev = nev
        self._masks = masks
        self._delayed_mode = delayed_mode

    def __repr__(self):
        return f"NminusOne(selections={self._names})"

    def result(self):
        """Returns the results of the N-1 selection as a namedtuple

        Returns
        -------
            result : NminusOneResult
                A namedtuple with the following attributes:

                nev : list of integers or dask_awkward.lib.core.Scalar objects
                    The number of events in each step of the N-1 selection as a list of integers or delayed integers
                masks : list of boolean numpy.ndarray or dask_awkward.lib.core.Array objects
                    The boolean mask vectors of which events pass the N-1 selection each time as a list of materialized or delayed boolean arrays

        """
        NminusOneResult = namedtuple("NminusOneResult", ["labels", "nev", "masks"])
        labels = ["initial"] + [f"N - {i}" for i in self._names] + ["N"]
        return NminusOneResult(labels, self._nev, self._masks)

    def to_npz(self, file, compressed=False, compute=False):
        """Saves the results of the N-1 selection to a .npz file

        Parameters
        ----------
            file : str or file
                Either the filename (string) or an open file (file-like object)
                where the data will be saved. If file is a string or a Path, the
                ``.npz`` extension will be appended to the filename if it is not
                already there.
            compressed : bool, optional
                If True, the data will be compressed in the ``.npz`` file.
                Default is False.
            compute : bool, optional
                Whether to immediately start writing or to return an object
                that the user can choose when to start writing by calling compute().
                Default is False.

        Returns
        -------
            out : NminusOneToNpz or None
                If ``compute=True``, returns None. Otherwise, returns an object
                that can be used to start writing the data by calling compute().
        """
        labels, nev, masks = self.result()

        if compressed:
            saver = numpy.savez_compressed
        else:
            saver = numpy.savez

        out = NminusOneToNpz(file, labels, nev, masks, saver)
        if compute:
            out.compute()
            return None
        else:
            return out

    def print(self):
        """Prints the statistics of the N-1 selection"""

        if self._delayed_mode:
            warnings.warn(
                "Printing the N-1 selection statistics is going to compute dask_awkward objects."
            )
            self._nev = list(dask.compute(*self._nev))

        nev = self._nev
        print("N-1 selection stats:")
        for i, name in enumerate(self._names):
            stats = (
                f"Ignoring {name:<20}"
                f"pass = {nev[i+1]:<20}"
                f"all = {nev[0]:<20}"
                f"-- eff = {nev[i+1]*100/nev[0]:.1f} %"
            )
            print(stats)

        stats_all = (
            f"All cuts {'':<20}"
            f"pass = {nev[-1]:<20}"
            f"all = {nev[0]:<20}"
            f"-- eff = {nev[-1]*100/nev[0]:.1f} %"
        )
        print(stats_all)

    def yieldhist(self):
        """Returns the N-1 selection yields as a ``hist.Hist`` object

        Returns
        -------
            h : hist.Hist or hist.dask.Hist
                Histogram of the number of events surviving the N-1 selection
            labels : list of strings
                The bin labels of the histogram
        """
        labels = ["initial"] + [f"N - {i}" for i in self._names] + ["N"]
        if not self._delayed_mode:
            h = hist.Hist(hist.axis.Integer(0, len(labels), name="N-1"))
            h.fill(numpy.arange(len(labels), dtype=int), weight=self._nev)

        else:
            h = hist.dask.Hist(hist.axis.Integer(0, len(labels), name="N-1"))
            for i, weight in enumerate(self._masks, 1):
                h.fill(dask_awkward.full_like(weight, i, dtype=int), weight=weight)
            h.fill(dask_awkward.zeros_like(weight, dtype=int))

        return h, labels

    def plot_vars(
        self,
        vars,
        axes=None,
        bins=None,
        start=None,
        stop=None,
        edges=None,
        transform=None,
    ):
        """Plot the histograms of variables for each step of the N-1 selection

        Parameters
        ----------
            vars : dict
                A dictionary in the form ``{name: array}`` where ``name`` is the name of the variable,
                and ``array`` is the corresponding array of values.
                The arrays must be the same length as each mask of the N-1 selection.
            axes : list of hist.axis objects, optional
                The axes objects to histogram the variables on. This will override all the following arguments that define axes.
                Must be the same length as ``vars``.
            bins : iterable of integers or Nones, optional
                The number of bins for each variable histogram. If not specified, it defaults to 20.
                Must be the same length as ``vars``.
            start : iterable of floats or integers or Nones, optional
                The lower edge of the first bin for each variable histogram. If not specified, it defaults to the minimum value of the variable array.
                Must be the same length as ``vars``.
            stop : iterable of floats or integers or Nones, optional
                The upper edge of the last bin for each variable histogram. If not specified, it defaults to the maximum value of the variable array.
                Must be the same length as ``vars``.
            edges : list of iterables of floats or integers, optional
                The bin edges for each variable histogram. This overrides ``bins``, ``start``, and ``stop`` if specified.
                Must be the same length as ``vars``.
            transform : iterable of hist.axis.transform objects or Nones, optional
                The transforms to apply to each variable histogram axis. If not specified, it defaults to None.
                Must be the same length as ``vars``.

        Returns
        -------
            hists : list of hist.Hist or hist.dask.Hist objects
                A list of 2D histograms of the variables for each step of the N-1 selection.
                The first axis is the variable, the second axis is the N-1 selection step.
            labels : list of strings
                The bin labels of y axis of the histogram.
        """
        if self._delayed_mode:
            for name, var in vars.items():
                if not compatible_partitions(var, self._masks[0]):
                    raise IncompatiblePartitions("plot_vars", var, self._masks[0])
        else:
            for name, var in vars.items():
                if len(var) != len(self._masks[0]):
                    raise ValueError(
                        f"The variable '{name}' has length '{len(var)}', but the masks have length '{len(self._masks[0])}'"
                    )

        hists = []
        labels = ["initial"] + [f"N - {i}" for i in self._names] + ["N"]

        bins = [None] * len(vars) if bins is None else bins
        start = [None] * len(vars) if start is None else start
        stop = [None] * len(vars) if stop is None else stop
        edges = [None] * len(vars) if edges is None else edges
        transform = [None] * len(vars) if transform is None else transform

        if axes is not None:
            axes = axes
        else:
            axes = []
            for (name, var), b, s1, s2, e, t in zip(
                vars.items(), bins, start, stop, edges, transform
            ):
                ax = coffea.util._gethistogramaxis(
                    name, var, b, s1, s2, e, t, self._delayed_mode
                )
                axes.append(ax)

        checklengths = [
            len(x) == len(vars) for x in (axes, bins, start, stop, edges, transform)
        ]
        if not all(checklengths):
            raise ValueError(
                "vars, axes, bins, start, stop, edges, and transform must be the same length"
            )

        if not self._delayed_mode:
            for (name, var), axis in zip(vars.items(), axes):
                h = hist.Hist(
                    axis,
                    hist.axis.Integer(0, len(labels), name="N-1"),
                )
                arr = awkward.flatten(var)
                h.fill(arr, awkward.zeros_like(arr, dtype=int))
                for i, mask in enumerate(self.result().masks, 1):
                    arr = awkward.flatten(var[mask])
                    h.fill(arr, awkward.full_like(arr, i, dtype=int))
                hists.append(h)

        else:
            for (name, var), axis in zip(vars.items(), axes):
                h = hist.dask.Hist(
                    axis,
                    hist.axis.Integer(0, len(labels), name="N-1"),
                )
                arr = dask_awkward.flatten(var)
                h.fill(arr, dask_awkward.zeros_like(arr, dtype=int))
                for i, mask in enumerate(self.result().masks, 1):
                    arr = dask_awkward.flatten(var[mask])
                    h.fill(arr, dask_awkward.full_like(arr, i, dtype=int))
                hists.append(h)

        return hists, labels


class Cutflow:
    """Object to be returned by PackedSelection.cutflow()"""

    def __init__(
        self,
        names,
        nevonecut,
        nevcutflow,
        masksonecut,
        maskscutflow,
        delayed_mode,
        commonmask=None,
        wgtevonecut=None,
        wgtevcutflow=None,
        weights=None,
        weightsmodifier=None,
    ):
        self._names = names
        self._nevonecut = nevonecut
        self._nevcutflow = nevcutflow
        self._masksonecut = masksonecut
        self._maskscutflow = maskscutflow
        self._delayed_mode = delayed_mode
        self._commonmask = commonmask
        self._wgtevonecut = wgtevonecut
        self._wgtevcutflow = wgtevcutflow
        self._weights = weights
        self._weightsmodifier = weightsmodifier
        self._commonmasked = self._commonmask is not None
        self._weighted = self._weights is not None

    def __repr__(self):
        return f"Cutflow(selections={self._names}, commonmasked={self._commonmasked}, weighted={self._weighted}, weightsmodifier={self._weightsmodifier})"

    def result(self, includeweights=None):
        """Returns the results of the cutflow as a namedtuple

        Parameters
        ----------
            includeweights : bool, optional
                Whether to include the weights in the result. Default is None, which includes the weights if the Cutflow was instantiated with weights
                and excludes them otherwise.

        Returns
        -------
            result : CutflowResult
                A namedtuple with the following attributes:

                nevonecut : list of integers or dask_awkward.lib.core.Scalar objects
                    The number of events that survive each cut alone as a list of integers or delayed integers
                nevcutflow : list of integers or dask_awkward.lib.core.Scalar objects
                    The number of events that survive the cumulative cutflow as a list of integers or delayed integers
                masksonecut : list of boolean numpy.ndarray or dask_awkward.lib.core.Array objects
                    The boolean mask vectors of which events pass each cut alone as a list of materialized or delayed boolean arrays
                maskscutflow : list of boolean numpy.ndarray or dask_awkward.lib.core.Array objects
                    The boolean mask vectors of which events pass the cumulative cutflow a list of materialized or delayed boolean arrays

            result: ExtendedCutflowResult
                A namedtuple with the CutflowResult properties and additionally the following:

                commonmask : boolean numpy.ndarray or dask_awkward.lib.core.Array object, or None if no common mask was provided
                wgtevonecut : list of floats or dask_awkward.lib.core.Scalar objects, or None if no weights were provided
                    The weighted number of events that survive each cut alone as a list of floats or delayed floats
                wgtevcutflow : list of floats or dask_awkward.lib.core.Scalar objects, or None if no weights were provided
                    The weighted number of events that survive the cumulative cutflow as a list of floats or delayed floats
                weights : float numpy.ndarray or dask_awkward.lib.core.Array object, or None if no weights were provided
                    The Weights.weight(modifier) array provided as input. Must be masked by masksonecut or maskscutflow to get the corresponding weights
                weightsmodifier : str or None
                    The modifier passed to Weights.weight([modifier]) if weights were provided

        """
        _include_weights = self._weighted if includeweights is None else includeweights
        CutflowResult = namedtuple(
            "CutflowResult",
            [
                "labels",
                "nevonecut",
                "nevcutflow",
                "masksonecut",
                "maskscutflow",
            ],
        )
        ExtendedCutflowResult = namedtuple(
            "ExtendedCutflowResult",
            [
                "labels",
                "nevonecut",
                "nevcutflow",
                "masksonecut",
                "maskscutflow",
                "commonmask",
                "wgtevonecut",
                "wgtevcutflow",
                "weights",
                "weightsmodifier",
            ],
        )
        labels = ["initial"] + list(self._names)
        if self._weighted or self._commonmasked:
            return ExtendedCutflowResult(
                labels,
                self._nevonecut,
                self._nevcutflow,
                self._masksonecut,
                self._maskscutflow,
                self._commonmask,
                self._wgtevonecut,
                self._wgtevcutflow,
                self._weights if _include_weights else None,
                self._weightsmodifier if _include_weights else None,
            )
        else:
            return CutflowResult(
                labels,
                self._nevonecut,
                self._nevcutflow,
                self._masksonecut,
                self._maskscutflow,
            )

    def to_npz(self, file, compressed=False, compute=False, includeweights=None):
        """Saves the results of the cutflow to a .npz file

        Parameters
        ----------
            file : str or file
                Either the filename (string) or an open file (file-like object)
                where the data will be saved. If file is a string or a Path, the
                ``.npz`` extension will be appended to the filename if it is not
                already there.
            compressed : bool, optional
                If True, the data will be compressed in the ``.npz`` file.
                Default is False.
            compute : bool, optional
                Whether to immediately start writing or to return an object
                that the user can choose when to start writing by calling compute().
                Default is False.
            includeweights : bool, optional
                Whether to save the weights in the npz file. Default is None, which saves the weights if the Cutflow was instantiated with weights

        Returns
        -------
            out : CutflowToNpz or None
                If ``compute=True``, returns None. Otherwise, returns an object
                that can be used to start writing the data by calling compute().
        """
        (
            labels,
            nevonecut,
            nevcutflow,
            masksonecut,
            maskscutflow,
            *packed_info,
        ) = self.result(includeweights=includeweights)

        if self._weighted or self._commonmasked:
            (
                commonmask,
                wgtevonecut,
                wgtevcutflow,
                weights,
                weightsmodifier,
            ) = packed_info
        else:
            (
                commonmask,
                wgtevonecut,
                wgtevcutflow,
                weights,
                weightsmodifier,
            ) = (None, None, None, None, None)
        if compressed:
            saver = numpy.savez_compressed
        else:
            saver = numpy.savez

        out = CutflowToNpz(
            file,
            labels,
            nevonecut,
            nevcutflow,
            masksonecut,
            maskscutflow,
            saver,
            commonmask,
            wgtevonecut,
            wgtevcutflow,
            weights,
            weightsmodifier,
            includeweights=includeweights,
        )
        if compute:
            out.compute()
            return None
        else:
            return out

    def print(self, weighted=None, scale=None):
        """Prints the statistics of the Cutflow

        Parameters
        ----------
            weighted : bool, optional
                Whether to print the weighted statistics. Default is None, which prints the weighted statistics
                if the cutflow was instantiated with weights and unweighted statistics otherwise.
            scale : float, optional
                A scale factor to apply to the cutflow statistics. Default is None, which does not apply any scaling.
        """
        do_weighted = self._weighted if weighted is None else weighted
        do_scaled = scale is not None
        if do_scaled:
            if isinstance(scale, (int, float)):
                pass
            else:
                raise ValueError(
                    f"The scale must be an integer or a float, {scale} (type {type(scale)}) was provided."
                )

        if self._delayed_mode:
            warnings.warn(
                "Printing the cutflow statistics is going to compute dask_awkward objects."
            )
            self._nevonecut, self._nevcutflow, self._wgtevonecut, self._wgtevcutflow = (
                dask.compute(
                    self._nevonecut,
                    self._nevcutflow,
                    self._wgtevonecut,
                    self._wgtevcutflow,
                )
            )

        xevonecut = self._nevonecut if not do_weighted else self._wgtevonecut
        xevcutflow = self._nevcutflow if not do_weighted else self._wgtevcutflow

        header = "Cutflow stats:"
        if do_weighted:
            header += " (weighted)"
        if do_scaled:
            header += f" (scaled by {scale})"
            xevonecut = [x * scale for x in xevonecut]
            xevcutflow = [x * scale for x in xevcutflow]
        print(header)
        for i, name in enumerate(self._names):
            stats = (
                f"Cut {name:<20}:"
                f"pass = {xevonecut[i+1]:<20}"
                f"cumulative pass = {xevcutflow[i+1]:<20}"
                f"all = {xevonecut[0]:<20}"
                f"-- eff = {xevonecut[i+1]*100/xevonecut[0]:.1f} %{'':<20}"
                f"-- cumulative eff = {xevcutflow[i+1]*100/xevcutflow[0]:.1f} %"
            )
            print(stats)

    def yieldhist(self, weighted=None, scale=None, categorical=None):
        """Returns the cutflow yields as ``hist.Hist`` objects

        Parameters
        ----------
            weighted : bool, optional
                Whether to fill the histograms with weights. Default is None, which applies the weights
                if the cutflow was instantiated with weights and unweighted statistics otherwise.
            categorical : dict, optional
                A dictionary with the following keys:
                    axis : hist.axis object
                        The axis to be used as a categorical axis
                    values : list
                        The array to be filled in the categorical axis, must be the same length as the masks
                    labels : list
                        The labels corresponding to the values in the categorical axis
                Default is None, which does not apply any categorical axis.

        Returns
        -------
            honecut : hist.Hist or hist.dask.Hist
                Histogram of the number of events surviving each cut alone
            hcutflow : hist.Hist or hist.dask.Hist
                Histogram of the number of events surviving the cumulative cutflow
            labels : list of strings
                The bin labels of the onecut/cutflow histograms
            catlabels : list of strings
                The labels of the categorical axis
        """
        do_weighted = self._weighted if weighted is None else weighted
        do_categorical = categorical is not None
        do_commonmasked = self._commonmasked
        do_scaled = scale is not None
        if do_scaled:
            if isinstance(scale, (int, float)):
                pass
            else:
                raise ValueError(
                    f"The scale must be an integer or a float, {scale} (type {type(scale)}) was provided."
                )
        Hist = hist.Hist if not self._delayed_mode else hist.dask.Hist
        ak_or_dak = awkward if not self._delayed_mode else dask_awkward
        labels = ["initial"] + list(self._names)
        axes = [hist.axis.Integer(0, len(labels), name="onecut")]
        if do_categorical:
            catax = categorical.get("axis")
            catvar = categorical.get("values")
            catlabels = categorical.get("labels")
            axes.append(catax)
        else:
            catlabels = None
        if do_weighted:
            axes.append(hist.storage.Weight())
        if not self._delayed_mode and not do_categorical:
            if categorical is not None:
                raise NotImplementedError(
                    "yieldhist is not implemented for non-delayed mode (v1) with categorical"
                )
            honecut = hist.Hist(*axes)
            hcutflow = honecut.copy()
            hcutflow.axes.name = ("cutflow",)
            weightonecut = self._wgtevonecut if do_weighted else self._nevonecut
            weightcutflow = self._wgtevcutflow if do_weighted else self._nevcutflow
            if do_scaled:
                weightonecut = [wgt * scale for wgt in weightonecut]
                weightcutflow = [wgt * scale for wgt in weightcutflow]
            honecut.fill(numpy.arange(len(labels), dtype=int), weight=weightonecut)
            hcutflow.fill(numpy.arange(len(labels), dtype=int), weight=weightcutflow)
        elif self._delayed_mode and not do_categorical:
            if categorical is not None:
                raise NotImplementedError(
                    "yieldhist is not implemented for non-delayed mode (v1) with categorical"
                )
            honecut = hist.dask.Hist(*axes)
            hcutflow = honecut.copy()
            hcutflow.axes.name = ("cutflow",)

            for i, mask in enumerate(self._masksonecut, 1):
                weight = (
                    self._weights.weight(self._weightsmodifier)[mask]
                    if do_weighted
                    else mask
                )
                if do_scaled:
                    weight = weight * scale
                honecut.fill(
                    dask_awkward.full_like(weight, i, dtype=int), weight=weight
                )
            weight = (
                self._weights.weight(self._weightsmodifier)
                if do_weighted
                else dask_awkward.ones_like(self._masksonecut[0], dtype=bool)
            )
            if do_scaled:
                weight = weight * scale
            if do_commonmasked:
                weight = weight[self._commonmask]
            honecut.fill(dask_awkward.zeros_like(weight, dtype=int), weight=weight)

            for i, mask in enumerate(self._maskscutflow, 1):
                weight = (
                    self._weights.weight(self._weightsmodifier)[mask]
                    if do_weighted
                    else mask
                )
                if do_scaled:
                    weight = weight * scale
                hcutflow.fill(
                    dask_awkward.full_like(weight, i, dtype=int), weight=weight
                )
            weight = (
                self._weights.weight(self._weightsmodifier)
                if do_weighted
                else dask_awkward.ones_like(self._maskscutflow[0], dtype=bool)
            )
            if do_scaled:
                weight = weight * scale
            if do_commonmasked:
                weight = weight[self._commonmask]
            hcutflow.fill(dask_awkward.zeros_like(weight, dtype=int), weight=weight)
        else:
            honecut = Hist(*axes)
            hcutflow = honecut.copy()
            hcutflow.axes.name = ("cutflow", *honecut.axes[1:].name)

            weight = (
                self._weights.weight(self._weightsmodifier)
                if do_weighted
                else ak_or_dak.ones_like(self._masksonecut[0], dtype=numpy.float32)
            )
            if do_scaled:
                weight = weight * scale
            if self._commonmasked:
                to_broadcastonecut = {
                    "onecut": boolean_masks_to_categorical_integers(
                        self._masksonecut, insert_commonmask_as_zeros=self._commonmask
                    )
                }
                to_broadcastcutflow = {
                    "cutflow": boolean_masks_to_categorical_integers(
                        self._maskscutflow, insert_commonmask_as_zeros=self._commonmask
                    )
                }
            else:
                to_broadcastonecut = {
                    "onecut": boolean_masks_to_categorical_integers(
                        self._masksonecut, insert_unmasked_as_zeros=True
                    )
                }
                to_broadcastcutflow = {
                    "cutflow": boolean_masks_to_categorical_integers(
                        self._maskscutflow, insert_unmasked_as_zeros=True
                    )
                }
            if do_categorical:
                to_broadcastonecut[catax.name] = catvar
                to_broadcastcutflow[catax.name] = catvar
            to_broadcastonecut["weight"] = weight
            to_broadcastcutflow["weight"] = weight
            broadcastedonecut = zip(
                to_broadcastonecut.keys(),
                ak_or_dak.broadcast_arrays(*to_broadcastonecut.values()),
            )
            broadcastedcutflow = zip(
                to_broadcastcutflow.keys(),
                ak_or_dak.broadcast_arrays(*to_broadcastcutflow.values()),
            )
            onecutargs = {
                k: ak_or_dak.flatten(arr, axis=None) for k, arr in broadcastedonecut
            }
            cutflowargs = {
                k: ak_or_dak.flatten(arr, axis=None) for k, arr in broadcastedcutflow
            }
            honecut.fill(**onecutargs)
            hcutflow.fill(**cutflowargs)

        if do_categorical:
            return honecut, hcutflow, labels, catlabels
        else:
            return honecut, hcutflow, labels

    def plot_vars(
        self,
        vars,
        axes=None,
        bins=None,
        start=None,
        stop=None,
        edges=None,
        transform=None,
        weighted=None,
        scale=None,
        categorical=None,
    ):
        """Plot the histograms of variables for each step of the N-1 selection

        Parameters
        ----------
            vars : dict
                A dictionary in the form ``{name: array}`` where ``name`` is the name of the variable,
                and ``array`` is the corresponding array of values.
                The arrays must be the same length as each mask of the cutflow.
            axes : list of hist.axis objects, optional
                The axes objects to histogram the variables on. This will override all the following arguments that define axes.
                Must be the same length as ``vars``.
            bins : iterable of integers or Nones, optional
                The number of bins for each variable histogram. If not specified, it defaults to 20.
                Must be the same length as ``vars``.
            start : iterable of floats or integers or Nones, optional
                The lower edge of the first bin for each variable histogram. If not specified, it defaults to the minimum value of the variable array.
                Must be the same length as ``vars``.
            stop : iterable of floats or integers or Nones, optional
                The upper edge of the last bin for each variable histogram. If not specified, it defaults to the maximum value of the variable array.
                Must be the same length as ``vars``.
            edges : list of iterables of floats or integers, optional
                The bin edges for each variable histogram. This overrides ``bins``, ``start``, and ``stop`` if specified.
                Must be the same length as ``vars``.
            transform : iterable of hist.axis.transform objects or Nones, optional
                The transforms to apply to each variable histogram axis. If not specified, it defaults to None.
                Must be the same length as ``vars``.
            weighted : bool, optional
                Whether to fill the histograms with weights. Default is None, which applies the weights
                if the cutflow was instantiated with weights and unweighted distributions otherwise.

        Returns
        -------
            histsonecut : list of hist.Hist or hist.dask.Hist objects
                A list of 1D histograms of the variables of events surviving each cut alone.
                The first axis is the variable, the second axis is the cuts.
            histscutflow : list of hist.Hist or hist.dask.Hist objects
                A list of 1D histograms of the variables of events surviving the cumulative cutflow.
                The first axis is the variable, the second axis is the cuts.
            labels : list of strings
                The bin labels of the y axis of the histograms.
            catlabels : list of strings, optional
                The labels of the categorical axis
        """
        do_weighted = self._weighted if weighted is None else weighted
        do_categorical = categorical is not None
        do_scaled = scale is not None
        if do_scaled:
            if isinstance(scale, (int, float)):
                pass
            else:
                raise ValueError(
                    f"The scale must be an integer or a float, {scale} (type {type(scale)}) was provided."
                )
        Hist = hist.dask.Hist if self._delayed_mode else hist.Hist
        ak_or_dak = dask_awkward if self._delayed_mode else awkward
        if do_categorical:
            catax = categorical.get("axis")
            catvar = categorical.get("values")
            catlabels = categorical.get("labels")
        if self._delayed_mode:
            for name, var in vars.items():
                if not compatible_partitions(var, self._masksonecut[0]):
                    raise IncompatiblePartitions("plot_vars", var, self._masksonecut[0])
            if do_categorical:
                if not compatible_partitions(catvar, self._masksonecut[0]):
                    raise IncompatiblePartitions(
                        "plot_vars (categorical values)", catvar, self._masksonecut[0]
                    )
        else:
            for name, var in vars.items():
                if len(var) != len(self._masksonecut[0]):
                    raise ValueError(
                        f"The variable '{name}' has length '{len(var)}', but the masks have length '{len(self._masksonecut[0])}'"
                    )

        histsonecut, histscutflow = [], []
        labels = ["initial"] + list(self._names)

        bins = [None] * len(vars) if bins is None else bins
        start = [None] * len(vars) if start is None else start
        stop = [None] * len(vars) if stop is None else stop
        edges = [None] * len(vars) if edges is None else edges
        transform = [None] * len(vars) if transform is None else transform

        if axes is not None:
            axes = axes
        else:
            axes = []
            for (name, var), b, s1, s2, e, t in zip(
                vars.items(), bins, start, stop, edges, transform
            ):
                ax = coffea.util._gethistogramaxis(
                    name, var, b, s1, s2, e, t, self._delayed_mode
                )
                axes.append(ax)

        checklengths = [
            len(x) == len(vars) for x in (axes, bins, start, stop, edges, transform)
        ]
        if not all(checklengths):
            raise ValueError(
                "vars, axes, bins, start, stop, edges, and transform must be the same length"
            )

        for (name, var), axis in zip(vars.items(), axes):
            constructor_args = [axis, hist.axis.Integer(0, len(labels), name="onecut")]
            fill_args = {name: var}
            if do_categorical:
                constructor_args.append(catax)
                fill_args[catax.name] = catvar
            if do_weighted:
                constructor_args.append(hist.storage.Weight())
            fill_args["weight"] = (
                self._weights.weight(self._weightsmodifier)
                if do_weighted
                else ak_or_dak.ones_like(self._masksonecut[0], dtype=numpy.float32)
            )
            if do_scaled:
                fill_args["weight"] = fill_args["weight"] * scale
            honecut = Hist(*constructor_args)
            hcutflow = honecut.copy()
            hcutflow.axes.name = (name, "cutflow", *honecut.axes[2:].name)

            # initial fill is special, needs to have commonmask applied if it exists
            to_fill_initial = (
                {k: v[self.result().commonmask] for k, v in fill_args.items()}
                if self._commonmasked
                else fill_args
            )
            to_fill_initial = dict(
                zip(
                    to_fill_initial.keys(),
                    [
                        ak_or_dak.flatten(arr)
                        for arr in ak_or_dak.broadcast_arrays(*to_fill_initial.values())
                    ],
                )
            )
            honecut.fill(
                onecut=ak_or_dak.zeros_like(to_fill_initial[name], dtype=int),
                **to_fill_initial,
            )

            for i, mask in enumerate(self.result().masksonecut, 1):
                to_fill_iter = {k: v[mask] for k, v in fill_args.items()}
                to_fill_iter = dict(
                    zip(
                        to_fill_iter.keys(),
                        [
                            ak_or_dak.flatten(arr)
                            for arr in ak_or_dak.broadcast_arrays(
                                *to_fill_iter.values()
                            )
                        ],
                    )
                )
                honecut.fill(
                    onecut=ak_or_dak.full_like(to_fill_iter[name], i, dtype=int),
                    **to_fill_iter,
                )
            histsonecut.append(honecut)

            hcutflow.fill(
                cutflow=ak_or_dak.zeros_like(to_fill_initial[name], dtype=int),
                **to_fill_initial,
            )
            for i, mask in enumerate(self.result().maskscutflow, 1):
                to_fill_iter = {k: v[mask] for k, v in fill_args.items()}
                to_fill_iter = dict(
                    zip(
                        to_fill_iter.keys(),
                        [
                            ak_or_dak.flatten(arr)
                            for arr in ak_or_dak.broadcast_arrays(
                                *to_fill_iter.values()
                            )
                        ],
                    )
                )
                hcutflow.fill(
                    cutflow=ak_or_dak.full_like(to_fill_iter[name], i, dtype=int),
                    **to_fill_iter,
                )
            histscutflow.append(hcutflow)

        if do_categorical:
            return histsonecut, histscutflow, labels, catlabels
        else:
            return histsonecut, histscutflow, labels


class PackedSelection:
    """Store several boolean arrays in a compact manner

    This class can store several boolean arrays in a memory-efficient mannner
    and evaluate arbitrary combinations of boolean requirements in an CPU-efficient way.
    Supported inputs are 1D numpy or awkward arrays.

    Parameters
    ----------
        dtype : numpy.dtype or str
            internal bitwidth of the packed array, which governs the maximum
            number of selections storable in this object. The default value
            is ``uint32``, which allows up to 32 booleans to be stored, but
            if a smaller or larger number of selections needs to be stored,
            one can choose ``uint16`` or ``uint64`` instead.
    """

    _supported_types = {
        numpy.dtype("uint16"): 16,
        numpy.dtype("uint32"): 32,
        numpy.dtype("uint64"): 64,
    }

    def __init__(self, dtype="uint32"):
        self._dtype = numpy.dtype(dtype)
        if self._dtype not in PackedSelection._supported_types:
            raise ValueError(f"dtype {dtype} is not supported")
        self._names = []
        self._data = None

    def __repr__(self):
        delayed_mode = None if self._data is None else self.delayed_mode
        return f"PackedSelection(selections={tuple(self._names)}, delayed_mode={delayed_mode}, items={len(self._names)}, maxitems={self.maxitems})"

    @property
    def names(self):
        """Current list of mask names available"""
        return self._names

    @property
    def delayed_mode(self):
        """
        Is the PackedSelection in delayed mode?

        Returns
        -------
            res: bool
                True if the PackedSelection is in delayed mode
        """
        if isinstance(self._data, dask_awkward.Array):
            return True
        elif isinstance(self._data, numpy.ndarray):
            return False
        else:
            warnings.warn(
                "PackedSelection hasn't been initialized with a boolean array yet!"
            )
            return False

    @property
    def maxitems(self):
        """
        What is the maximum supported number of selections in this PackedSelection?

        Returns
        -------
            res: bool
                The maximum supported number of selections
        """
        return PackedSelection._supported_types[self._dtype]

    def __add_delayed(self, name, selection, fill_value):
        """Add a new delayed boolean array"""
        selection = coffea.util._ensure_flat(selection, allow_missing=True)
        sel_type = dask_awkward.type(selection)
        if isinstance(sel_type, awkward.types.OptionType):
            selection = dask_awkward.fill_none(selection, fill_value)
            sel_type = dask_awkward.type(selection)
        if sel_type.primitive != "bool":
            raise ValueError(f"Expected a boolean array, received {sel_type.primitive}")
        if len(self._names) == 0:
            self._data = dask_awkward.zeros_like(selection, dtype=self._dtype)
        if isinstance(selection, dask_awkward.Array) and not self.delayed_mode:
            raise ValueError(
                f"New selection '{name}' is not eager while PackedSelection is!"
            )
        elif len(self._names) == self.maxitems:
            raise RuntimeError(
                f"Exhausted all slots in PackedSelection: {self}, consider a larger dtype or fewer selections"
            )
        elif not dask_awkward.lib.core.compatible_partitions(self._data, selection):
            raise ValueError(
                f"New selection '{name}' has a different partition structure than existing selections"
            )
        self._data = numpy.bitwise_or(
            self._data,
            selection * self._dtype.type(1 << len(self._names)),
        )
        self._names.append(name)

    def __add_eager(self, name, selection, fill_value):
        """Add a new eager boolean array"""
        selection = coffea.util._ensure_flat(selection, allow_missing=True)
        if isinstance(selection, numpy.ma.MaskedArray):
            selection = selection.filled(fill_value)
        if selection.dtype != bool:
            raise ValueError(f"Expected a boolean array, received {selection.dtype}")
        if len(self._names) == 0:
            self._data = numpy.zeros(len(selection), dtype=self._dtype)
        if isinstance(selection, numpy.ndarray) and self.delayed_mode:
            raise ValueError(
                f"New selection '{name}' is not delayed while PackedSelection is!"
            )
        elif len(self._names) == self.maxitems:
            raise RuntimeError(
                f"Exhausted all slots in PackedSelection: {self}, consider a larger dtype or fewer selections"
            )
        elif self._data.shape != selection.shape:
            raise ValueError(
                f"New selection '{name}' has a different shape than existing selections ({selection.shape} vs. {self._data.shape})"
            )
        numpy.bitwise_or(
            self._data,
            self._dtype.type(1 << len(self._names)),
            where=selection,
            out=self._data,
        )
        self._names.append(name)

    def add(self, name, selection, fill_value=False):
        """Add a new boolean array

        Parameters
        ----------
            name : str
                name of the selection
            selection : numpy.ndarray or awkward.Array
                a flat array of type ``bool`` or ``?bool``.
                If this is not the first selection added, it must also have
                the same shape as previously added selections. If the array
                is option-type, null entries will be filled with ``fill_value``.
            fill_value : bool, optional
                All masked entries will be filled as specified (default: ``False``)
        """
        if name in self._names:
            raise ValueError(f"Selection '{name}' already exists")
        if isinstance(selection, dask.array.Array):
            raise ValueError(
                "Dask arrays are not supported, please convert them to dask_awkward.Array by using dask_awkward.from_dask_array()"
            )
        selection = coffea.util._ensure_flat(selection, allow_missing=True)
        if isinstance(selection, numpy.ndarray):
            self.__add_eager(name, selection, fill_value)
        elif isinstance(selection, dask_awkward.Array):
            self.__add_delayed(name, selection, fill_value)

    def add_multiple(self, selections, fill_value=False):
        """Add multiple boolean arrays at once, see ``add`` for details

        Parameters
        ----------
            selections : dict
                a dictionary of selections, in the form ``{name: selection}``
            fill_value : bool, optional
                All masked entries will be filled as specified (default: ``False``)
        """
        for name, selection in selections.items():
            self.add(name, selection, fill_value)

    @lru_cache
    def require(self, **names):
        """Return a mask vector corresponding to specific requirements

        Specify an exact requirement on an arbitrary subset of the masks

        Parameters
        ----------
            ``**names`` : kwargs
                Each argument to require specific value for, in form ``arg=True``
                or ``arg=False``.

        Examples
        --------
        If

        >>> selection.names
        ['cut1', 'cut2', 'cut3']

        then

        >>> selection.require(cut1=True, cut2=False)
        array([True, False, True, ...])

        returns a boolean array where an entry is True if the corresponding entries
        ``cut1 == True``, ``cut2 == False``, and ``cut3`` arbitrary.
        """
        for cut, v in names.items():
            if not isinstance(cut, str) or cut not in self._names:
                raise ValueError(
                    "All arguments must be strings that refer to the names of existing selections"
                )

        consider = 0
        require = 0
        for name, val in names.items():
            val = bool(val)
            idx = self._names.index(name)
            consider |= 1 << idx
            require |= int(val) << idx
        return (self._data & self._dtype.type(consider)) == require

    def all(self, *names):
        """Shorthand for `require`, where all the values are True.
        If no arguments are given, all the added selections are required to be True.
        """
        if names:
            return self.require(**{name: True for name in names})
        return self.require(**{name: True for name in self._names})

    def allfalse(self, *names):
        """Shorthand for `require`, where all the values are False.
        If no arguments are given, all the added selections are required to be False.
        """
        if names:
            return self.require(**{name: False for name in names})
        return self.require(**{name: False for name in self._names})

    def any(self, *names):
        """Return a mask vector corresponding to an inclusive OR of requirements

        Parameters
        ----------
            ``*names`` : args
                The named selections to allow

        Examples
        --------
        If

        >>> selection.names
        ['cut1', 'cut2', 'cut3']

        then

        >>> selection.any("cut1", "cut2")
        array([True, False, True, ...])

        returns a boolean array where an entry is True if the corresponding entries
        ``cut1 == True`` or ``cut2 == False``, and ``cut3`` arbitrary.
        """
        for cut in names:
            if not isinstance(cut, str) or cut not in self._names:
                raise ValueError(
                    "All arguments must be strings that refer to the names of existing selections"
                )
        consider = 0
        for name in names:
            idx = self._names.index(name)
            consider |= 1 << idx
        return (self._data & self._dtype.type(consider)) != 0

    def nminusone(self, *names):
        """Compute the "N-1" style selection for a set of selections

        The N-1 style selection for a set of selections, returns an object which can return a list of the number of events
        that pass all the other selections ignoring one at a time. The first element of the returned list
        is the total number of events before any selections are applied.
        The last element is the final number of events that pass if all selections are applied.
        It also returns a list of boolean mask vectors of which events pass the N-1 selection each time.
        Can also return a histogram as a ``hist.Hist`` object where the bin heights are the number of events of the N-1 selection list.
        If the PackedSelection is in delayed mode, the elements of those lists will be dask_awkward Arrays that can be computed whenever the user wants.
        If the histogram is requested, the delayed arrays of the number of events list will be computed in the process in order to set the bin heights.

        Parameters
        ----------
            ``*names`` : args
                The named selections to use, need to be a subset of the selections already added

        Returns
        -------
            res: coffea.analysis_tools.NminusOne
                A wrapper class for the results, see the documentation for that class for more details
        """
        for cut in names:
            if not isinstance(cut, str) or cut not in self._names:
                raise ValueError(
                    "All arguments must be strings that refer to the names of existing selections"
                )

        masks = []
        for i, cut in enumerate(names):
            mask = self.all(*(names[:i] + names[i + 1 :]))
            masks.append(mask)
        mask = self.all(*names)
        masks.append(mask)

        if not self.delayed_mode:
            nev = [len(self._data)]
            nev.extend(numpy.sum(masks, axis=1))

        else:
            nev = [dask_awkward.count(self._data, axis=0)]
            nev.extend([dask_awkward.sum(mask) for mask in masks])

        return NminusOne(names, nev, masks, self.delayed_mode)

    def cutflow(self, *names, commonmask=None, weights=None, weightsmodifier=None):
        """Compute the cutflow for a set of selections

        Returns an object which can return a list of the number of events that pass all the previous selections including the current one
        after each named selection is applied consecutively. The first element
        of the returned list is the total number of events before any selections are applied.
        The last element is the final number of events that pass after all the selections are applied.
        Can also return a cutflow histogram as a ``hist.Hist`` object where the bin heights are the number of events of the cutflow list.
        If the PackedSelection is in delayed mode, the elements of the list will be dask_awkward Arrays that can be computed whenever the user wants.
        If the histogram is requested, those delayed arrays will be computed in the process in order to set the bin heights.

        Parameters
        ----------
            ``*names`` : args
                The named selections to use, need to be a subset of the selections already added
            commonmask : boolean numpy.ndarray or dask_awkward.lib.core.Array, optional
                A common mask which is applied for all the selections, including the initial one. Default is None.
            weights : coffea.analysis_tools.Weights instance, optional
                The Weights object to use for the cutflow. If not provided, the cutflow will be unweighted.
            modifier : str, optional
                The modifier to use for the weights. Default is None which results in Weights.weight() being called without a modifier.

        Returns
        -------
            res: coffea.analysis_tools.Cutflow
                A wrapper class for the results, see the documentation for that class for more details
        """
        for cut in names:
            if not isinstance(cut, str) or cut not in self._names:
                raise ValueError(
                    "All arguments must be strings that refer to the names of existing selections"
                )

        masksonecut, maskscutflow = [], []
        if weights is not None:
            weightsonecut, weightscutflow = [], []
        else:
            weightsonecut, weightscutflow, wgtevonecut, wgtevcutflow = (
                None,
                None,
                None,
                None,
            )
        for i, cut in enumerate(names):
            mask1 = self.any(cut)
            mask2 = self.all(*(names[: i + 1]))
            if commonmask is not None:
                mask1 = mask1 & commonmask
                mask2 = mask2 & commonmask
            masksonecut.append(mask1)
            maskscutflow.append(mask2)
            if weights is not None:
                weights1 = weights.weight(weightsmodifier)[mask1]
                weights2 = weights.weight(weightsmodifier)[mask2]
                weightsonecut.append(weights1)
                weightscutflow.append(weights2)

        if not self.delayed_mode:
            nevonecut = [
                numpy.sum(commonmask) if commonmask is not None else len(self._data)
            ]
            nevcutflow = [
                numpy.sum(commonmask) if commonmask is not None else len(self._data)
            ]
            nevonecut.extend(numpy.sum(masksonecut, axis=1))
            nevcutflow.extend(numpy.sum(maskscutflow, axis=1))
            if weights is not None:
                if commonmask is not None:
                    wgtevonecut = [
                        numpy.sum(weights.weight(weightsmodifier)[commonmask])
                    ]
                    wgtevcutflow = [
                        numpy.sum(weights.weight(weightsmodifier)[commonmask])
                    ]
                else:
                    wgtevonecut = [numpy.sum(weights.weight(weightsmodifier))]
                    wgtevcutflow = [numpy.sum(weights.weight(weightsmodifier))]
                wgtevonecut.extend([numpy.sum(wgt1) for wgt1 in weightsonecut])
                wgtevcutflow.extend([numpy.sum(wgt2) for wgt2 in weightscutflow])

        else:
            nevonecut = [
                (
                    dask_awkward.sum(commonmask)
                    if commonmask is not None
                    else dask_awkward.count(self._data, axis=0)
                )
            ]
            nevcutflow = [
                (
                    dask_awkward.sum(commonmask)
                    if commonmask is not None
                    else dask_awkward.count(self._data, axis=0)
                )
            ]
            nevonecut.extend([dask_awkward.sum(mask1) for mask1 in masksonecut])
            nevcutflow.extend([dask_awkward.sum(mask2) for mask2 in maskscutflow])
            if weights is not None:
                if commonmask is not None:
                    wgtevonecut = [
                        dask_awkward.sum(weights.weight(weightsmodifier)[commonmask])
                    ]
                    wgtevcutflow = [
                        dask_awkward.sum(weights.weight(weightsmodifier)[commonmask])
                    ]
                else:
                    wgtevonecut = [dask_awkward.sum(weights.weight(weightsmodifier))]
                    wgtevcutflow = [dask_awkward.sum(weights.weight(weightsmodifier))]
                wgtevonecut.extend([dask_awkward.sum(wgt1) for wgt1 in weightsonecut])
                wgtevcutflow.extend([dask_awkward.sum(wgt2) for wgt2 in weightscutflow])

        return Cutflow(
            names,
            nevonecut,
            nevcutflow,
            masksonecut,
            maskscutflow,
            self.delayed_mode,
            commonmask,
            wgtevonecut,
            wgtevcutflow,
            weights,
            weightsmodifier,
        )
