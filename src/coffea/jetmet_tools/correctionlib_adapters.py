"""Adapter classes that wrap correctionlib (JSON-POG) corrections for use with
:class:`CorrectedJetsFactory`.

The existing ``CorrectedJetsFactory`` is **not** modified at all.  Instead
these thin adapters implement the same duck-typed interface that the factory
already expects from ``JECStack`` and its component correctors
(``FactorizedJetCorrector``, ``JetResolution``, ``JetResolutionScaleFactor``,
``JetCorrectionUncertainty``).
"""

import awkward
import correctionlib
import dask_awkward
import numpy

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_flat_numpy(arr):
    """Convert an awkward array (flat or jagged) to a flat numpy float32 array."""
    flat = awkward.flatten(arr, axis=None)
    return numpy.asarray(flat, dtype=numpy.float32)


def _is_typetracer(*args):
    """Return ``True`` if any of *args* lives on the typetracer backend."""
    return any(awkward.backend(a) == "typetracer" for a in args)


def _typetracer_1d(ref):
    """Return a 1-D float32 typetracer array shaped like *ref*."""
    out = awkward.Array(awkward.typetracer.length_zero_if_typetracer(ref))
    return awkward.Array(
        out.layout.to_typetracer(forget_length=True), behavior=out.behavior
    )


def _typetracer_nd(ref, inner_shape):
    """Return an N-D float32 typetracer array with the given inner shape."""
    n = 0  # zero-length
    arr = numpy.zeros((n, *inner_shape), dtype=numpy.float32)
    out = awkward.Array(arr)
    return awkward.Array(
        out.layout.to_typetracer(forget_length=True), behavior=out.behavior
    )


# ---------------------------------------------------------------------------
# Callable objects for dask_awkward.map_partitions
# ---------------------------------------------------------------------------


class _CorrectionLibEvalFn:
    """Callable that evaluates a ``correctionlib.Correction`` on awkward arrays.

    Designed to be used as the *func* argument to
    ``dask_awkward.map_partitions``.
    """

    def __init__(self, correction, input_names):
        self._correction = correction
        self._input_names = input_names

    def __call__(self, *args):
        if _is_typetracer(*args):
            return _typetracer_1d(args[0])
        np_args = [_to_flat_numpy(a) for a in args]
        result = self._correction.evaluate(*np_args)
        return awkward.Array(result.astype(numpy.float32))


class _CompoundCorrectionLibEvalFn:
    """Like :class:`_CorrectionLibEvalFn` but for ``CompoundCorrection``."""

    def __init__(self, correction, input_names):
        self._correction = correction
        self._input_names = input_names

    def __call__(self, *args):
        if _is_typetracer(*args):
            return _typetracer_1d(args[0])
        np_args = [_to_flat_numpy(a) for a in args]
        result = self._correction.evaluate(*np_args)
        return awkward.Array(result.astype(numpy.float32))


class _JERSFEvalFn:
    """Evaluates a JER scale-factor correction for three systematics and stacks
    them into an (N, 3) array ``[nom, up, down]``."""

    def __init__(self, correction, input_names):
        self._correction = correction
        self._input_names = input_names

    def __call__(self, *args):
        if _is_typetracer(*args):
            return _typetracer_nd(args[0], (3,))
        np_args = [_to_flat_numpy(a) for a in args]
        nom = self._correction.evaluate(*np_args, "nom").astype(numpy.float32)
        up = self._correction.evaluate(*np_args, "up").astype(numpy.float32)
        down = self._correction.evaluate(*np_args, "down").astype(numpy.float32)
        stacked = numpy.stack([nom, up, down], axis=1)
        return awkward.Array(stacked)


class _JUNCSourceEvalFn:
    """Evaluates a single JES uncertainty source and returns an (N, 2) array
    ``[1+delta, 1-delta]``."""

    def __init__(self, correction, input_names):
        self._correction = correction
        self._input_names = input_names

    def __call__(self, *args):
        if _is_typetracer(*args):
            return _typetracer_nd(args[0], (2,))
        np_args = [_to_flat_numpy(a) for a in args]
        delta = self._correction.evaluate(*np_args).astype(numpy.float32)
        stacked = numpy.stack([1.0 + delta, 1.0 - delta], axis=1).astype(numpy.float32)
        return awkward.Array(stacked)


# ---------------------------------------------------------------------------
# Helpers for the dask / eager dispatch
# ---------------------------------------------------------------------------


def _dispatch(func_cls, correction, input_names, kwargs):
    """Route evaluation through ``dask_awkward.map_partitions`` when the first
    kwarg value is a dask array, otherwise evaluate eagerly."""
    first_arg = kwargs[input_names[0]]
    args = tuple(kwargs[name] for name in input_names)

    if isinstance(first_arg, dask_awkward.Array):
        func = func_cls(correction, input_names)
        # Compute zero-length meta
        zl_args = tuple(
            awkward.Array(
                a.layout.form.length_zero_array(highlevel=False),
                behavior=a.behavior,
            )
            for a in args
        )
        zl_out = func(*zl_args)
        meta = awkward.Array(
            zl_out.layout.to_typetracer(forget_length=True),
            behavior=zl_out.behavior,
        )
        return dask_awkward.map_partitions(
            func,
            *args,
            label="correctionlib",
            meta=meta,
        )
    else:
        func = func_cls(correction, input_names)
        return func(*args)


# ---------------------------------------------------------------------------
# Public adapter classes
# ---------------------------------------------------------------------------


class CorrectionLibJEC:
    """Adapter for a compound (or single) JEC correction from correctionlib.

    Parameters
    ----------
    correction : correctionlib.Correction or correctionlib.CompoundCorrection
        The correction object (e.g. ``cset.compound["...L1L2L3Res..."]``).
    """

    def __init__(self, correction):
        self._correction = correction
        self._signature = [inp.name for inp in correction.inputs]
        self._is_compound = (
            isinstance(correction, correctionlib.highlevel.CorrectionSet)
            or type(correction).__name__ == "CompoundCorrection"
        )

    @property
    def signature(self):
        return self._signature

    def getCorrection(self, **kwargs):
        fn_cls = (
            _CompoundCorrectionLibEvalFn if self._is_compound else _CorrectionLibEvalFn
        )
        return _dispatch(fn_cls, self._correction, self._signature, kwargs)


class CorrectionLibJER:
    """Adapter for a JER pt-resolution correction from correctionlib.

    Parameters
    ----------
    correction : correctionlib.Correction
        The JER pt-resolution correction.
    """

    def __init__(self, correction):
        self._correction = correction
        self._signature = [inp.name for inp in correction.inputs]

    @property
    def signature(self):
        return self._signature

    def getResolution(self, **kwargs):
        return _dispatch(
            _CorrectionLibEvalFn, self._correction, self._signature, kwargs
        )


class CorrectionLibJERSF:
    """Adapter for a JER scale-factor correction from correctionlib.

    The correctionlib correction takes a ``"systematic"`` input that selects
    between ``"nom"``, ``"up"``, and ``"down"``.  This adapter evaluates all
    three and stacks them into an ``(N, 3)`` array so that
    ``CorrectedJetsFactory`` can index as ``jersf[:, variation]``.

    Parameters
    ----------
    correction : correctionlib.Correction
        The JER scale-factor correction.
    """

    def __init__(self, correction):
        self._correction = correction
        # Signature exposed to the factory excludes "systematic"
        self._signature = [
            inp.name for inp in correction.inputs if inp.name != "systematic"
        ]
        # Full input names (without "systematic") for the eval function
        self._eval_inputs = self._signature

    @property
    def signature(self):
        return self._signature

    def getScaleFactor(self, **kwargs):
        return _dispatch(_JERSFEvalFn, self._correction, self._eval_inputs, kwargs)


class CorrectionLibJUNC:
    """Adapter for JES uncertainty sources from correctionlib.

    ``getUncertainty`` returns an iterable of ``(name, (N, 2) array)`` tuples
    where column 0 is ``1 + delta`` (up) and column 1 is ``1 - delta`` (down),
    matching the convention expected by ``CorrectedJetsFactory.build()``.

    Parameters
    ----------
    sources : list[tuple[str, correctionlib.Correction]]
        Each element is ``(source_name, correction)`` where *correction*
        evaluates the signed uncertainty delta for that source.
    """

    def __init__(self, sources):
        self._sources = list(sources)
        # Signature comes from the first source's inputs
        if len(self._sources) == 0:
            self._signature = []
        else:
            self._signature = [inp.name for inp in self._sources[0][1].inputs]

    @property
    def signature(self):
        return self._signature

    @property
    def levels(self):
        return [name for name, _ in self._sources]

    def getUncertainty(self, **kwargs):
        results = []
        for name, corr in self._sources:
            unc = _dispatch(_JUNCSourceEvalFn, corr, self._signature, kwargs)
            results.append((name, unc))
        return results


class CorrectionLibJECStack:
    """Drop-in replacement for :class:`JECStack` backed by correctionlib.

    Parameters
    ----------
    jec : CorrectionLibJEC, optional
    junc : CorrectionLibJUNC, optional
    jer : CorrectionLibJER, optional
    jersf : CorrectionLibJERSF, optional
    """

    def __init__(self, jec=None, junc=None, jer=None, jersf=None):
        self._jec = jec
        self._junc = junc
        self._jer = jer
        self._jersf = jersf

        if (self._jer is None) != (self._jersf is None):
            raise ValueError(
                "Cannot apply JER-SF without an input JER, and vice-versa!"
            )

    @property
    def jec(self):
        return self._jec

    @property
    def junc(self):
        return self._junc

    @property
    def jer(self):
        return self._jer

    @property
    def jersf(self):
        return self._jersf

    @property
    def blank_name_map(self):
        out = {
            "massRaw",
            "ptRaw",
            "JetMass",
            "JetPt",
            "METpt",
            "METphi",
            "JetPhi",
            "UnClusteredEnergyDeltaX",
            "UnClusteredEnergyDeltaY",
        }
        if self._jec is not None:
            for name in self._jec.signature:
                out.add(name)
        if self._junc is not None:
            for name in self._junc.signature:
                out.add(name)
        if self._jer is not None:
            for name in self._jer.signature:
                out.add(name)
        if self._jersf is not None:
            for name in self._jersf.signature:
                out.add(name)
        return {name: None for name in out}

    @classmethod
    def from_file(
        cls,
        path,
        jec_tag,
        data_type,
        jet_type,
        jec_level="L1L2L3Res",
        unc_sources=None,
        jer_tag=None,
    ):
        """Construct a :class:`CorrectionLibJECStack` from a JSON-POG file.

        Parameters
        ----------
        path : str
            Path to the JSON-POG ``.json`` or ``.json.gz`` file.
        jec_tag : str
            JEC campaign tag, e.g. ``"Summer24Prompt24_V2"``.
        data_type : str
            ``"MC"`` or ``"DATA"``.
        jet_type : str
            Jet algorithm, e.g. ``"AK4PFPuppi"``.
        jec_level : str
            Compound correction level, e.g. ``"L1L2L3Res"``.
        unc_sources : list[str], optional
            Uncertainty source names, e.g.
            ``["Regrouped_FlavorQCD", "Regrouped_AbsoluteMPFBias"]``.
        jer_tag : str, optional
            JER campaign tag.  If ``None``, JER/JERSF are not loaded.

        Examples
        --------
        Build a correctionlib-backed JEC stack and apply it to jets::

            from coffea.jetmet_tools import CorrectionLibJECStack, CorrectedJetsFactory

            # Load corrections from a JSON-POG file
            jec_stack = CorrectionLibJECStack.from_file(
                "jet_jerc.json.gz",
                jec_tag="Summer24Prompt24_V2",
                data_type="MC",
                jet_type="AK4PFPuppi",
                jec_level="L1L2L3Res",
                unc_sources=["Regrouped_FlavorQCD", "Regrouped_AbsoluteMPFBias"],
                jer_tag="Summer23BPixPrompt23_RunD_JRV1",
            )

            # Build the name map (maps correction input names to jet field names)
            name_map = jec_stack.blank_name_map
            name_map["JetPt"] = "pt"
            name_map["JetMass"] = "mass"
            name_map["JetEta"] = "eta"
            name_map["JetA"] = "area"
            name_map["ptRaw"] = "pt_raw"
            name_map["massRaw"] = "mass_raw"
            name_map["Rho"] = "rho"
            name_map["ptGenJet"] = "pt_gen"
            name_map["METpt"] = "met_pt"
            name_map["METphi"] = "met_phi"
            name_map["JetPhi"] = "phi"
            name_map["UnClusteredEnergyDeltaX"] = "MetUnclustEnUpDeltaX"
            name_map["UnClusteredEnergyDeltaY"] = "MetUnclustEnUpDeltaY"

            # Create factory and apply corrections (same API as txt-based JECStack)
            factory = CorrectedJetsFactory(name_map, jec_stack)
            corrected_jets = factory.build(jets)

            # Access uncertainties
            for unc in factory.uncertainties():
                print(unc, corrected_jets[unc].up.pt, corrected_jets[unc].down.pt)
        """
        cset = correctionlib.CorrectionSet.from_file(path)

        # JEC (compound correction)
        jec_name = f"{jec_tag}_{data_type}_{jec_level}_{jet_type}"
        jec_adapter = CorrectionLibJEC(cset.compound[jec_name])

        # JUNC
        junc_adapter = None
        if unc_sources:
            source_pairs = []
            for source in unc_sources:
                unc_name = f"{jec_tag}_{data_type}_{source}_{jet_type}"
                source_pairs.append((source, cset[unc_name]))
            junc_adapter = CorrectionLibJUNC(source_pairs)

        # JER / JERSF
        jer_adapter = None
        jersf_adapter = None
        if jer_tag is not None:
            jer_name = f"{jer_tag}_{data_type}_PtResolution_{jet_type}"
            jer_adapter = CorrectionLibJER(cset[jer_name])

            jersf_name = f"{jer_tag}_{data_type}_ScaleFactor_{jet_type}"
            jersf_adapter = CorrectionLibJERSF(cset[jersf_name])

        return cls(
            jec=jec_adapter,
            junc=junc_adapter,
            jer=jer_adapter,
            jersf=jersf_adapter,
        )
