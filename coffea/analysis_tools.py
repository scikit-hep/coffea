"""Tools of general use for columnar analysis

These helper classes were previously part of ``coffea.processor``
but have been migrated and updated to be compatible with awkward-array 1.0
"""
import awkward
import numpy
import coffea.util
import coffea.processor


class WeightStatistics(coffea.processor.AccumulatorABC):
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
        self.sumw += other.sumw
        self.sumw2 += other.sumw2
        self.minw = min(self.minw, other.minw)
        self.maxw = max(self.maxw, other.maxw)
        self.n += other.n


class Weights:
    """Container for event weights and associated systematic shifts

    This container keeps track of correction factors and systematic
    effects that can be encoded as multiplicative modifiers to the event weight.
    All weights are stored in vector form.

    Parameters
    ----------
        size : int
            size of the weight arrays to be handled (i.e. the number of events / instances).
        storeIndividual : bool, optional
            store not only the total weight + variations, but also each individual weight.
            Default is false.
    """

    def __init__(self, size, storeIndividual=False):
        self._weight = numpy.ones(size)
        self._weights = {}
        self._modifiers = {}
        self._weightStats = coffea.processor.dict_accumulator()
        self._storeIndividual = storeIndividual

    @property
    def weightStatistics(self):
        return self._weightStats

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
                if True, interpret weightUp and weightDown as a realtive difference (additive) to the
                nominal value

        .. note:: ``weightUp`` and ``weightDown`` are assumed to be rvalue-like and may be modified in-place by this function
        """
        if name.endswith("Up") or name.endswith("Down"):
            raise ValueError(
                "Avoid using 'Up' and 'Down' in weight names, instead pass appropriate shifts to add() call"
            )
        weight = coffea.util._ensure_flat(weight, allow_missing=True)
        if isinstance(weight, numpy.ma.MaskedArray):
            # TODO what to do with option-type? is it representative of unknown weight
            # and we default to one or is it an invalid weight and we should never use this
            # event in the first place (0) ?
            weight = weight.filled(1.0)
        self._weight = self._weight * weight
        if self._storeIndividual:
            self._weights[name] = weight
        self.__add_variation(name, weight, weightUp, weightDown, shift)
        self._weightStats[name] = WeightStatistics(
            weight.sum(),
            (weight**2).sum(),
            awkward.min(weight, mask_identity=False),
            awkward.max(weight, mask_identity=False),
            weight.size,
        )

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
                if True, interpret weightUp and weightDown as a realtive difference (additive) to the
                nominal value

        .. note:: ``weightUp`` and ``weightDown`` are assumed to be rvalue-like and may be modified in-place by this function
        """
        if name.endswith("Up") or name.endswith("Down"):
            raise ValueError(
                "Avoid using 'Up' and 'Down' in weight names, instead pass appropriate shifts to add() call"
            )
        weight = coffea.util._ensure_flat(weight, allow_missing=True)
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
        self._weightStats[name] = WeightStatistics(
            weight.sum(),
            (weight**2).sum(),
            weight.min(initial=numpy.inf),
            weight.max(initial=-numpy.inf),
            weight.size,
        )

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
                if True, interpret weightUp and weightDown as a realtive difference (additive) to the
                nominal value

        .. note:: ``weightUp`` and ``weightDown`` are assumed to be rvalue-like and may be modified in-place by this function
        """
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

    def weight(self, modifier=None):
        """Current event weight vector

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

    def partial_weight(self, include=[], exclude=[]):
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
        Returns
        -------
            weight : numpy.ndarray
                The weight vector, corresponding to only the effect of the
                corrections specified.
        """
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

        w = numpy.ones(self._weight.size)
        for name in names:
            w *= self._weights[name]

        return w

    @property
    def variations(self):
        """List of available modifiers"""
        keys = set(self._modifiers.keys())
        # add any missing 'Down' variation
        for k in self._modifiers.keys():
            keys.add(k.replace("Up", "Down"))
        return keys


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

    @property
    def names(self):
        """Current list of mask names available"""
        return self._names

    @property
    def maxitems(self):
        return PackedSelection._supported_types[self._dtype]

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
        selection = coffea.util._ensure_flat(selection, allow_missing=True)
        if isinstance(selection, numpy.ma.MaskedArray):
            selection = selection.filled(fill_value)
        if selection.dtype != bool:
            raise ValueError(f"Expected a boolean array, received {selection.dtype}")
        if len(self._names) == 0:
            self._data = numpy.zeros(len(selection), dtype=self._dtype)
        elif len(self._names) == self.maxitems:
            raise RuntimeError(
                f"Exhausted all slots in {self}, consider a larger dtype or fewer selections"
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
        consider = 0
        require = 0
        for name, val in names.items():
            val = bool(val)
            idx = self._names.index(name)
            consider |= 1 << idx
            require |= int(val) << idx
        return (self._data & consider) == require

    def all(self, *names):
        """Shorthand for `require`, where all the values are True"""
        return self.require(**{name: True for name in names})

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
        consider = 0
        for name in names:
            idx = self._names.index(name)
            consider |= 1 << idx
        return (self._data & consider) != 0
