import awkward
import numpy

from coffea.util import _isinstance, maybe_map_partitions


def corrected_polar_met(
    met_pt, met_phi, jet_pt, jet_phi, jet_pt_orig, positive=None, dx=None, dy=None
):
    sj, cj = numpy.sin(jet_phi), numpy.cos(jet_phi)
    x = met_pt * numpy.cos(met_phi) - awkward.sum((jet_pt - jet_pt_orig) * cj, axis=1)
    y = met_pt * numpy.sin(met_phi) - awkward.sum((jet_pt - jet_pt_orig) * sj, axis=1)
    if positive is not None and dx is not None and dy is not None:
        x = x + dx if positive else x - dx
        y = y + dy if positive else y - dy

    return awkward.zip(
        {"pt": numpy.hypot(x, y), "phi": numpy.arctan2(y, x)}, depth_limit=1
    )


def corrected_type1_met(
    raw_met_pt, raw_met_phi, delta_px, delta_py, positive=None, dx=None, dy=None
):
    """Compute Type-1 corrected MET from pre-computed jet delta sums.

    Parameters
    ----------
    raw_met_pt, raw_met_phi : array
        Raw (uncorrected) MET pt and phi.
    delta_px, delta_py : array
        Per-event sum of (pt_noMuL1L2L3 - pt_noMuL1) * cos/sin(phi_noMuRaw)
        from both Jet and CorrT1METJet collections.
    positive : bool or None
        If not None, shift by unclustered energy (True=up, False=down).
    dx, dy : array or None
        Unclustered energy delta x/y.
    """
    x = raw_met_pt * numpy.cos(raw_met_phi) - delta_px
    y = raw_met_pt * numpy.sin(raw_met_phi) - delta_py
    if positive is not None and dx is not None and dy is not None:
        x = x + dx if positive else x - dx
        y = y + dy if positive else y - dy

    return awkward.zip(
        {"pt": numpy.hypot(x, y), "phi": numpy.arctan2(y, x)}, depth_limit=1
    )


# Required name_map keys for Type-1 mode (Jet collection). The raw jet pT is
# read from ``ptRaw`` (an always-required key); ``JetRawFactor`` is optional and
# only consulted to derive it when the ``ptRaw`` field is missing from the jets.
_TYPE1_JET_KEYS = [
    "RawMETpt",
    "RawMETphi",
    "JetMuonSubtrFactor",
    "JetMuonSubtrDeltaPhi",
    "JetChEmEF",
    "JetNeEmEF",
]

# Required name_map keys when CorrT1METJet is used
_TYPE1_CORRT1_KEYS = [
    "CorrT1JetPt",
    "CorrT1JetPhi",
    "CorrT1JetEta",
    "CorrT1JetArea",
    "CorrT1JetMuonSubtrFactor",
    "CorrT1JetMuonSubtrDeltaPhi",
    "CorrT1JetEmEF",
]


def _ensure_jet_raw_pt_field(jets, name_map):
    """Return ``jets`` with the ``ptRaw`` field present.

    Unlike CorrT1METJet, the Jet collection has no native raw pT; it is normally
    added before jet correction (the ``pt_raw = (1 - rawFactor) * pt`` idiom) and
    preserved by ``CorrectedJetsFactory``. If the field is absent it is derived
    from ``JetRawFactor`` and the uncorrected pT (``<JetPt>_orig`` when the jets
    came from ``CorrectedJetsFactory``, else ``JetPt``). An existing field is
    never overwritten.

    Returns
    -------
    awkward.Array
        ``jets`` with a ``name_map["ptRaw"]`` field.
    """
    raw_field = name_map["ptRaw"]
    if raw_field in jets.fields:
        return jets

    raw_factor_key = name_map.get("JetRawFactor")
    if raw_factor_key is None or raw_factor_key not in jets.fields:
        raise ValueError(
            f"Type-1 MET needs a raw jet pT: add a {raw_field!r} field "
            "(e.g. 'pt_raw = (1 - rawFactor) * pt' before jet correction) or "
            "provide a 'JetRawFactor' name mapping so it can be derived."
        )

    orig_key = name_map["JetPt"] + "_orig"
    base_pt = jets[orig_key] if orig_key in jets.fields else jets[name_map["JetPt"]]
    return awkward.with_field(jets, base_pt * (1.0 - jets[raw_factor_key]), raw_field)


def _compute_jec_factors(jets, name_map, jec_L1, jec_L1L2L3):
    """Compute L1 and L1L2L3 JEC factors for the Jet collection.

    The correctors accept jagged arrays directly, so inputs keep their event
    structure. The raw pT is read from the ``ptRaw`` field.

    Returns
    -------
    factor_L1, factor_L1L2L3 : awkward.Array
        Jagged arrays of JEC factors matching ``jets``.
    """
    jet_pt_raw = jets[name_map["ptRaw"]]

    def jec_inputs(corrector):
        return {
            k: (jet_pt_raw if k == "JetPt" else jets[name_map[k]])
            for k in corrector.signature
        }

    factor_L1 = awkward.values_astype(
        jec_L1.getCorrection(**jec_inputs(jec_L1)), numpy.float32
    )
    factor_L1L2L3 = awkward.values_astype(
        jec_L1L2L3.getCorrection(**jec_inputs(jec_L1L2L3)), numpy.float32
    )

    return factor_L1, factor_L1L2L3


def _compute_corrt1_jec_factors(corrt1jets, name_map, jec_L1, jec_L1L2L3):
    """Compute L1 and L1L2L3 JEC factors for the CorrT1METJet collection.

    CorrT1METJet stores raw pT directly (``rawPt``); the generic JEC input names
    are remapped onto its fields.

    Returns
    -------
    factor_L1, factor_L1L2L3 : awkward.Array
        Jagged arrays of JEC factors matching ``corrt1jets``.
    """
    key_remap = {
        "JetPt": "CorrT1JetPt",
        "JetEta": "CorrT1JetEta",
        "JetA": "CorrT1JetArea",
    }

    def jec_inputs(corrector):
        return {
            k: corrt1jets[name_map[key_remap.get(k, k)]] for k in corrector.signature
        }

    factor_L1 = awkward.values_astype(
        jec_L1.getCorrection(**jec_inputs(jec_L1)), numpy.float32
    )
    factor_L1L2L3 = awkward.values_astype(
        jec_L1L2L3.getCorrection(**jec_inputs(jec_L1L2L3)), numpy.float32
    )

    return factor_L1, factor_L1L2L3


def _compute_jet_type1_deltas_with_factors(
    jets, name_map, factor_L1, factor_L1L2L3, pt_scale_factor=None
):
    """Compute per-jet Type-1 MET correction deltas using pre-computed JEC factors.

    Returns an awkward record with fields ``delta_px`` and ``delta_py``
    (per-event sums from selected jets).

    Parameters
    ----------
    jets : awkward.Array
        The corrected jets array (jagged).
    name_map : dict
        Name map with Jet field mappings.
    factor_L1, factor_L1L2L3 : awkward.Array
        Pre-computed JEC factors (jagged, matching jets shape).
    pt_scale_factor : awkward.Array or None
        If provided, scale the full L1L2L3-corrected jet pT (for JES/JER
        variations). The nominal no-muon pT is still used for jet selection.
    """
    # Step 1: muon-subtracted raw pT and phi
    jet_pt_raw = jets[name_map["ptRaw"]]
    muon_substr_factor = jets[name_map["JetMuonSubtrFactor"]]
    muon_substr_dphi = jets[name_map["JetMuonSubtrDeltaPhi"]]
    jet_phi = jets[name_map["JetPhi"]]

    # Match types to CMSJMECalculators
    pt_noMuRaw = awkward.values_astype(
        jet_pt_raw * (1.0 - muon_substr_factor), numpy.float64
    )
    muon_pt = awkward.values_astype(jet_pt_raw * muon_substr_factor, numpy.float64)
    phi_noMuRaw = muon_substr_dphi + jet_phi

    # Step 2: apply pre-computed JEC factors
    pt_noMuL1 = pt_noMuRaw * factor_L1
    pt_noMuL1L2L3 = pt_noMuRaw * factor_L1L2L3

    # Step 3: selection cuts
    chEmEF = jets[name_map["JetChEmEF"]]
    neEmEF = jets[name_map["JetNeEmEF"]]
    mask = (
        (pt_noMuL1L2L3 > 15.0)
        & (numpy.abs(jets[name_map["JetEta"]]) < 5.2)  # JERC tutorial adds this
        & ((chEmEF + neEmEF) < 0.9)
    )

    # Step 4: vectorial sum of (pt_noMuL1L2L3 - pt_noMuL1) for selected jets
    if pt_scale_factor is None:
        diff_pt = pt_noMuL1L2L3 - pt_noMuL1
    else:
        pt_L1 = pt_noMuL1 + muon_pt
        pt_L1L2L3 = pt_noMuL1L2L3 + muon_pt
        diff_pt = pt_L1L2L3 * pt_scale_factor - pt_L1

    diff_pt = awkward.where(mask, diff_pt, 0.0)
    delta_px = awkward.sum(diff_pt * numpy.cos(phi_noMuRaw), axis=1)
    delta_py = awkward.sum(diff_pt * numpy.sin(phi_noMuRaw), axis=1)

    return awkward.zip({"delta_px": delta_px, "delta_py": delta_py}, depth_limit=1)


def _compute_corrt1_type1_deltas_with_factors(
    corrt1jets, name_map, factor_L1, factor_L1L2L3, pt_scale_factor=None
):
    """Compute per-jet Type-1 MET correction deltas for CorrT1METJet using pre-computed JEC factors.

    Returns an awkward record with fields ``delta_px`` and ``delta_py``.
    """
    # Step 1: muon-subtracted raw pT and phi
    raw_pt = corrt1jets[name_map["CorrT1JetPt"]]
    muon_substr_factor = corrt1jets[name_map["CorrT1JetMuonSubtrFactor"]]
    muon_substr_dphi = corrt1jets[name_map["CorrT1JetMuonSubtrDeltaPhi"]]
    jet_phi = corrt1jets[name_map["CorrT1JetPhi"]]

    # Match types to CMSJMECalculators
    pt_noMuRaw = awkward.values_astype(
        raw_pt * (1.0 - muon_substr_factor), numpy.float64
    )
    phi_noMuRaw = muon_substr_dphi + jet_phi
    muon_pt = awkward.values_astype(raw_pt * muon_substr_factor, numpy.float64)

    # Step 2: apply pre-computed JEC factors
    pt_noMuL1 = pt_noMuRaw * factor_L1
    pt_noMuL1L2L3 = pt_noMuRaw * factor_L1L2L3

    # Step 3: selection cuts
    emEF = corrt1jets[name_map["CorrT1JetEmEF"]]
    mask = (
        (pt_noMuL1L2L3 > 15.0)
        & (numpy.abs(corrt1jets[name_map["CorrT1JetEta"]]) < 5.2)
        & (emEF < 0.9)
    )

    # Step 4: vectorial sum
    if pt_scale_factor is None:
        diff_pt = pt_noMuL1L2L3 - pt_noMuL1
    else:
        pt_L1 = pt_noMuL1 + muon_pt
        pt_L1L2L3 = pt_noMuL1L2L3 + muon_pt
        diff_pt = pt_L1L2L3 * pt_scale_factor - pt_L1

    diff_pt = awkward.where(mask, diff_pt, 0.0)
    delta_px = awkward.sum(diff_pt * numpy.cos(phi_noMuRaw), axis=1)
    delta_py = awkward.sum(diff_pt * numpy.sin(phi_noMuRaw), axis=1)

    return awkward.zip({"delta_px": delta_px, "delta_py": delta_py}, depth_limit=1)


def _evaluate_jet_jes_uncertainty(
    jets, name_map, jes_uncertainty, pt_L1L2L3, is_corrt1=False
):
    """Evaluate a JES uncertainty source on CorrT1METJet.

    CMSJMECalculators evaluates JES uncertainty sources at the full corrected jet pT
    (no-muon corrected pT plus the uncorrected muon component).
    """

    if is_corrt1:
        key_remap = {
            "JetEta": "CorrT1JetEta",
            "JetA": "CorrT1JetArea",
        }
    else:
        key_remap = {"JetEta": "JetEta", "JetA": "JetA"}

    inputs = {}
    for key in jes_uncertainty.signature:
        if key == "JetPt":
            inputs[key] = pt_L1L2L3
        else:
            inputs[key] = jets[name_map[key_remap.get(key, key)]]

    return awkward.values_astype(jes_uncertainty.getCorrection(**inputs), numpy.float32)


def _compute_type1_jes_deltas_with_factors(
    jets,
    name_map,
    factor_L1,
    factor_L1L2L3,
    jes_uncertainty,
    nominal_smear_factor=None,
    *,
    is_corrt1=False,
):
    """Compute Jet or CorrT1METJet Type-1 JES up/down deltas.

    The Type-1 jet selection is made using nominal no-muon L1L2L3 pT.
    The JES source is then evaluated at the full Type-1 corrected jet pT and
    applied as ``1 +/- delta``, matching CMSJMECalculators.
    """
    if is_corrt1:
        jet_pt_raw = jets[name_map["CorrT1JetPt"]]
        muon_subtr_factor = jets[name_map["CorrT1JetMuonSubtrFactor"]]
        muon_subtr_dphi = jets[name_map["CorrT1JetMuonSubtrDeltaPhi"]]
        jet_phi = jets[name_map["CorrT1JetPhi"]]
        jet_eta = jets[name_map["CorrT1JetEta"]]
        emEF = jets[name_map["CorrT1JetEmEF"]]
    else:
        jet_pt_raw = jets[name_map["ptRaw"]]
        muon_subtr_factor = jets[name_map["JetMuonSubtrFactor"]]
        muon_subtr_dphi = jets[name_map["JetMuonSubtrDeltaPhi"]]
        jet_phi = jets[name_map["JetPhi"]]
        jet_eta = jets[name_map["JetEta"]]
        emEF = jets[name_map["JetChEmEF"]] + jets[name_map["JetNeEmEF"]]

    # Match types to CMSJMECalculators
    pt_noMuRaw = awkward.values_astype(
        jet_pt_raw * (1.0 - muon_subtr_factor), numpy.float64
    )
    muon_pt = awkward.values_astype(jet_pt_raw * muon_subtr_factor, numpy.float64)
    phi_noMuRaw = muon_subtr_dphi + jet_phi

    pt_noMuL1 = pt_noMuRaw * factor_L1
    pt_noMuL1L2L3 = pt_noMuRaw * factor_L1L2L3
    pt_L1 = pt_noMuL1 + muon_pt
    pt_L1L2L3 = pt_noMuL1L2L3 + muon_pt

    mask = (pt_noMuL1L2L3 > 15.0) & (numpy.abs(jet_eta) < 5.2) & (emEF < 0.9)

    delta = _evaluate_jet_jes_uncertainty(
        jets,
        name_map,
        jes_uncertainty,
        pt_L1L2L3,
        is_corrt1=is_corrt1,
    )

    if nominal_smear_factor is None:
        nominal_smear_factor = 1.0
    # CMSJMECalculators offsets the L1 reference by the nominal JER contribution before
    # constructing JES branches. JES and JER are additive around the smeared
    # nominal rather than multiplying the JES delta by the JER factor.
    pt_L1_reference = pt_L1 + pt_L1L2L3 * (1.0 - nominal_smear_factor)
    diff_pt_up = awkward.where(mask, pt_L1L2L3 * (1.0 + delta) - pt_L1_reference, 0.0)
    diff_pt_down = awkward.where(mask, pt_L1L2L3 * (1.0 - delta) - pt_L1_reference, 0.0)

    cos_phi = numpy.cos(phi_noMuRaw)
    sin_phi = numpy.sin(phi_noMuRaw)

    return awkward.zip(
        {
            "up_delta_px": awkward.sum(diff_pt_up * cos_phi, axis=1),
            "up_delta_py": awkward.sum(diff_pt_up * sin_phi, axis=1),
            "down_delta_px": awkward.sum(diff_pt_down * cos_phi, axis=1),
            "down_delta_py": awkward.sum(diff_pt_down * sin_phi, axis=1),
        },
        depth_limit=1,
    )


class CorrectedMETFactory:
    """
    Factory class for propagating corrections made to jets into a corrected value
    of MET. This includes organizing different variations associated with uncertainties
    in MET from unclustered energy.

    Once the ``CorrectedMETFactory`` is constructed, an array of corrected MET values and
    variations can be produced with the `build` method, which requires an array of
    uncorrected MET and an array of corrected jets.

    Parameters
    ----------
        name_map : dict[str, str]
            Keys must include at least the following:

                - METpt
                - METphi
                - JetPt
                - JetPhi
                - ptRaw
                - UnClusteredEnergyDeltaX
                - UnClusteredEnergyDeltaY

            and each of those must be mapped to the corresponding field name of the input
            arrays ``in_MET`` and ``in_corrected_jets`` for the ``build`` method.

            When ``jec_L1L2L3`` and ``jec_L1`` are provided (Type-1 mode), additional
            keys are required. See the class documentation for details.
        jec_L1L2L3 : corrector or None
            Full JEC corrector (L1L2L3). Must have ``.signature`` and
            ``.getCorrection(**kwargs)``. Can be ``FactorizedJetCorrector`` or
            ``CorrectionLibJEC``.
        jec_L1 : corrector or None
            L1-only JEC corrector. Same interface as ``jec_L1L2L3``.
        jes_uncertainties : dict[str, corrector] or None
            Optional mapping from corrected-jet uncertainty field names to
            JES uncertainty correctors evaluated directly in Type-1 mode.
        is_t1_smeared_met : bool
            If true, include nominal and varied JER factors in Type-1 MET.
    """

    def __init__(
        self,
        name_map,
        jec_L1L2L3=None,
        jec_L1=None,
        jes_uncertainties=None,
        is_t1_smeared_met=False,
    ):
        # Validate that both or neither JEC corrector is provided
        if (jec_L1L2L3 is None) != (jec_L1 is None):
            raise ValueError(
                "Both jec_L1L2L3 and jec_L1 must be provided together, or neither."
            )

        self.type1_mode = jec_L1L2L3 is not None
        self.jec_L1L2L3 = jec_L1L2L3
        self.jec_L1 = jec_L1
        self.jes_uncertainties = dict(jes_uncertainties or {})
        self.is_t1_smeared_met = is_t1_smeared_met

        # Always require legacy keys
        for name in [
            "METpt",
            "METphi",
            "JetPt",
            "JetPhi",
            "ptRaw",
            "UnClusteredEnergyDeltaX",
            "UnClusteredEnergyDeltaY",
        ]:
            if name not in name_map or name_map[name] is None:
                raise ValueError(
                    f"There is no name mapping for {name}, which is needed for CorrectedMETFactory"
                )

        # In Type-1 mode, validate additional required keys
        if self.type1_mode:
            for name in _TYPE1_JET_KEYS:
                if name not in name_map or name_map[name] is None:
                    raise ValueError(
                        f"There is no name mapping for {name}, which is needed for "
                        f"CorrectedMETFactory in Type-1 mode"
                    )

        self.name_map = name_map

    def build(self, in_MET, in_corrected_jets, in_RawMET=None, in_CorrT1METJets=None):
        """
        Produce an array of corrected MET values from an array of uncorrected MET
        values and an array of corrected jets.

        Parameters
        ----------
            in_MET : awkward.Array or dask_awkward.Array
                An array of (uncorrected) MET values.
            in_corrected_jets : awkward.Array or dask_awkward.Array
                An array of corrected jets, as produced by `CorrectedJetsFactory`.
            in_RawMET : awkward.Array or dask_awkward.Array, optional
                Raw (uncorrected) MET array. Required in Type-1 mode.
            in_CorrT1METJets : awkward.Array or dask_awkward.Array, optional
                CorrT1METJet collection. Optional even in Type-1 mode.

        Returns
        -------
            awkward.Array or dask_awkward.Array
                Array of corrected MET values with shape matching ``in_MET``.
        """
        if not _isinstance(
            in_MET, "awkward.highlevel.Array", "dask_awkward.lib.core.Array"
        ) or not _isinstance(
            in_corrected_jets,
            "awkward.highlevel.Array",
            "dask_awkward.lib.core.Array",
        ):
            raise Exception(
                "'MET' and 'corrected_jets' must be an (dask_)awkward array of some kind!"
            )

        if self.type1_mode:
            if in_RawMET is None:
                raise ValueError(
                    "in_RawMET is required when CorrectedMETFactory is in Type-1 mode "
                    "(jec_L1L2L3 and jec_L1 were provided)."
                )
            if in_CorrT1METJets is not None:
                # Validate CorrT1 name_map keys
                for name in _TYPE1_CORRT1_KEYS:
                    if name not in self.name_map or self.name_map[name] is None:
                        raise ValueError(
                            f"There is no name mapping for {name}, which is needed "
                            f"when in_CorrT1METJets is provided"
                        )
            return self._build_type1(
                in_MET, in_corrected_jets, in_RawMET, in_CorrT1METJets
            )

        # --- Legacy path (unchanged) ---
        return self._build_legacy(in_MET, in_corrected_jets)

    def _build_legacy(self, in_MET, in_corrected_jets):
        """Legacy MET correction path — identical to the original implementation."""
        MET = in_MET
        corrected_jets = in_corrected_jets

        def switch_properties(raw_met, corrected_jets, dx, dy, positive, save_orig):
            variation = corrected_polar_met(
                raw_met[self.name_map["METpt"]],
                raw_met[self.name_map["METphi"]],
                corrected_jets[self.name_map["JetPt"]],
                corrected_jets[self.name_map["JetPhi"]],
                corrected_jets[self.name_map["JetPt"] + "_orig"],
                positive=positive,
                dx=dx,
                dy=dy,
            )
            out = awkward.with_field(raw_met, variation.pt, self.name_map["METpt"])
            out = awkward.with_field(out, variation.phi, self.name_map["METphi"])
            if save_orig:
                out = awkward.with_field(
                    out,
                    raw_met[self.name_map["METpt"]],
                    self.name_map["METpt"] + "_orig",
                )
                out = awkward.with_field(
                    out,
                    raw_met[self.name_map["METphi"]],
                    self.name_map["METphi"] + "_orig",
                )

            return out

        def create_variants(raw_met, corrected_jets_or_variants, dx, dy):
            if dx is not None and dy is not None:
                return awkward.zip(
                    {
                        "up": switch_properties(
                            raw_met,
                            corrected_jets_or_variants,
                            dx,
                            dy,
                            True,
                            False,
                        ),
                        "down": switch_properties(
                            raw_met,
                            corrected_jets_or_variants,
                            dx,
                            dy,
                            False,
                            False,
                        ),
                    },
                    depth_limit=1,
                    with_name="METSystematic",
                )
            else:
                return awkward.zip(
                    {
                        "up": switch_properties(
                            raw_met,
                            corrected_jets_or_variants.up,
                            dx,
                            dy,
                            True,
                            False,
                        ),
                        "down": switch_properties(
                            raw_met,
                            corrected_jets_or_variants.down,
                            None,
                            None,
                            None,
                            False,
                        ),
                    },
                    depth_limit=1,
                    with_name="METSystematic",
                )

        out = maybe_map_partitions(
            switch_properties,
            MET,
            corrected_jets,
            None,
            None,
            None,
            True,
            label="nominal_corrected_met",
        )

        out_dict = {field: out[field] for field in awkward.fields(out)}

        out_dict["MET_UnclusteredEnergy"] = maybe_map_partitions(
            create_variants,
            MET,
            corrected_jets,
            MET[self.name_map["UnClusteredEnergyDeltaX"]],
            MET[self.name_map["UnClusteredEnergyDeltaY"]],
            label="UnclusteredEnergy_met",
        )

        for unc in filter(
            lambda x: x.startswith(("JER", "JES")), awkward.fields(corrected_jets)
        ):
            out_dict[unc] = maybe_map_partitions(
                create_variants,
                MET,
                corrected_jets[unc],
                None,
                None,
                label=f"{unc}_met",
            )

        out_parms = out.layout.parameters
        out = awkward.zip(
            out_dict, depth_limit=1, parameters=out_parms, behavior=out.behavior
        )

        return out

    def _build_type1(self, in_MET, in_corrected_jets, in_RawMET, in_CorrT1METJets):
        """Type-1 MET correction path."""
        MET = in_MET
        corrected_jets = in_corrected_jets
        raw_met = in_RawMET
        corrt1jets = in_CorrT1METJets

        # Ensure the jets carry a raw pT field (derive it if the user did not).
        if self.name_map["ptRaw"] not in corrected_jets.fields:

            def ensure_raw_pt(jets):
                return _ensure_jet_raw_pt_field(jets, self.name_map)

            corrected_jets = maybe_map_partitions(
                ensure_raw_pt, corrected_jets, label="type1_ensure_raw_pt"
            )

        # --- Compute JEC factors once (reused for all variations) ---
        def compute_jet_jec_factors(jets):
            f_L1, f_L1L2L3 = _compute_jec_factors(
                jets, self.name_map, self.jec_L1, self.jec_L1L2L3
            )
            return awkward.zip(
                {"factor_L1": f_L1, "factor_L1L2L3": f_L1L2L3}, depth_limit=1
            )

        jet_jec = maybe_map_partitions(
            compute_jet_jec_factors,
            corrected_jets,
            label="type1_jet_jec_factors",
        )
        jet_factor_L1 = jet_jec.factor_L1
        jet_factor_L1L2L3 = jet_jec.factor_L1L2L3

        # --- Compute nominal Jet deltas using pre-computed factors ---
        jet_nominal_smear_factor = None
        if self.is_t1_smeared_met:

            def get_nominal_jet_smear_factor(jets):
                if "jet_energy_resolution_correction" in jets.fields:
                    return jets["jet_energy_resolution_correction"]
                jec_pt_field = self.name_map["JetPt"] + "_jec"
                if jec_pt_field not in jets.fields:
                    raise ValueError(
                        "Smeared Type-1 MET requires either a "
                        "jet_energy_resolution_correction or a JEC-only pT field"
                    )
                safe_jec_pt = awkward.where(
                    jets[jec_pt_field] != 0, jets[jec_pt_field], 1.0
                )
                return jets[self.name_map["JetPt"]] / safe_jec_pt

            jet_nominal_smear_factor = maybe_map_partitions(
                get_nominal_jet_smear_factor,
                corrected_jets,
                label="type1_nominal_jet_smear_factor",
            )

        def compute_nominal_jet_deltas(jets, f_L1, f_L1L2L3, smear_factor):
            return _compute_jet_type1_deltas_with_factors(
                jets,
                self.name_map,
                f_L1,
                f_L1L2L3,
                pt_scale_factor=smear_factor,
            )

        if jet_nominal_smear_factor is None:
            jet_deltas = maybe_map_partitions(
                compute_nominal_jet_deltas,
                corrected_jets,
                jet_factor_L1,
                jet_factor_L1L2L3,
                None,
                label="type1_jet_deltas",
            )
        else:
            jet_deltas = maybe_map_partitions(
                compute_nominal_jet_deltas,
                corrected_jets,
                jet_factor_L1,
                jet_factor_L1L2L3,
                jet_nominal_smear_factor,
                label="type1_jet_deltas",
            )
        jet_dpx = jet_deltas.delta_px
        jet_dpy = jet_deltas.delta_py

        # Total deltas start with Jet contribution
        total_dpx = jet_dpx
        total_dpy = jet_dpy

        # --- Compute CorrT1METJet deltas (if provided) ---
        corrt1_factor_L1 = None
        corrt1_factor_L1L2L3 = None
        corrt1_dpx = None
        corrt1_dpy = None
        corrt1_nominal_smear_factor = None

        if corrt1jets is not None:

            def compute_corrt1_jec_factors(ct1jets):
                f_L1, f_L1L2L3 = _compute_corrt1_jec_factors(
                    ct1jets,
                    self.name_map,
                    self.jec_L1,
                    self.jec_L1L2L3,
                )
                return awkward.zip(
                    {"factor_L1": f_L1, "factor_L1L2L3": f_L1L2L3},
                    depth_limit=1,
                )

            corrt1_jec = maybe_map_partitions(
                compute_corrt1_jec_factors,
                corrt1jets,
                label="type1_corrt1_jec_factors",
            )
            corrt1_factor_L1 = corrt1_jec.factor_L1
            corrt1_factor_L1L2L3 = corrt1_jec.factor_L1L2L3

            corrt1_nominal_smear_factor = None
            if self.is_t1_smeared_met:

                def get_corrt1_nominal_smear_factor(ct1jets):
                    field = self.name_map.get("CorrT1JetJERSmearFactor")
                    if field is not None and field in ct1jets.fields:
                        return ct1jets[field]
                    return awkward.ones_like(ct1jets[self.name_map["CorrT1JetPt"]])

                corrt1_nominal_smear_factor = maybe_map_partitions(
                    get_corrt1_nominal_smear_factor,
                    corrt1jets,
                    label="type1_corrt1_nominal_smear_factor",
                )

            def compute_corrt1_deltas(ct1jets, f_L1, f_L1L2L3, smear_factor):
                return _compute_corrt1_type1_deltas_with_factors(
                    ct1jets,
                    self.name_map,
                    f_L1,
                    f_L1L2L3,
                    pt_scale_factor=smear_factor,
                )

            corrt1_deltas = maybe_map_partitions(
                compute_corrt1_deltas,
                corrt1jets,
                corrt1_factor_L1,
                corrt1_factor_L1L2L3,
                corrt1_nominal_smear_factor,
                label="type1_corrt1_deltas",
            )
            corrt1_dpx = corrt1_deltas.delta_px
            corrt1_dpy = corrt1_deltas.delta_py
            total_dpx = total_dpx + corrt1_dpx
            total_dpy = total_dpy + corrt1_dpy

        # --- Nominal corrected MET ---
        def build_nominal(met_record, rmet, dpx, dpy):
            raw_pt = rmet[self.name_map["RawMETpt"]]
            raw_phi = rmet[self.name_map["RawMETphi"]]
            variation = corrected_type1_met(raw_pt, raw_phi, dpx, dpy)
            out = awkward.with_field(met_record, variation.pt, self.name_map["METpt"])
            out = awkward.with_field(out, variation.phi, self.name_map["METphi"])
            out = awkward.with_field(
                out,
                raw_pt,
                self.name_map["METpt"] + "_orig",
            )
            out = awkward.with_field(
                out,
                raw_phi,
                self.name_map["METphi"] + "_orig",
            )
            return out

        out = maybe_map_partitions(
            build_nominal,
            MET,
            raw_met,
            total_dpx,
            total_dpy,
            label="type1_nominal_met",
        )

        out_dict = {field: out[field] for field in awkward.fields(out)}

        # --- Unclustered energy systematics ---
        def build_unclustered_variants(met_record, rmet, dpx, dpy, dx, dy):
            raw_pt = rmet[self.name_map["RawMETpt"]]
            raw_phi = rmet[self.name_map["RawMETphi"]]
            var_up = corrected_type1_met(
                raw_pt, raw_phi, dpx, dpy, positive=True, dx=dx, dy=dy
            )
            var_down = corrected_type1_met(
                raw_pt, raw_phi, dpx, dpy, positive=False, dx=dx, dy=dy
            )

            up_out = awkward.with_field(met_record, var_up.pt, self.name_map["METpt"])
            up_out = awkward.with_field(up_out, var_up.phi, self.name_map["METphi"])
            down_out = awkward.with_field(
                met_record, var_down.pt, self.name_map["METpt"]
            )
            down_out = awkward.with_field(
                down_out, var_down.phi, self.name_map["METphi"]
            )
            return awkward.zip(
                {"up": up_out, "down": down_out},
                depth_limit=1,
                with_name="METSystematic",
            )

        out_dict["MET_UnclusteredEnergy"] = maybe_map_partitions(
            build_unclustered_variants,
            MET,
            raw_met,
            total_dpx,
            total_dpy,
            MET[self.name_map["UnClusteredEnergyDeltaX"]],
            MET[self.name_map["UnClusteredEnergyDeltaY"]],
            label="type1_UnclusteredEnergy_met",
        )

        # --- JES/JER systematics ---
        # For JES sources with a supplied payload, evaluate the source directly
        # on both Jet and CorrT1METJet at the Type-1 full corrected pT.  This
        # matches CMSJMECalculators and avoids importing a scale factor evaluated at the
        # standard corrected-jet pT, which is not identical when a jet contains
        # a muon.  JER (and unmapped JES sources) retain the pre-built varied-pT
        # fallback.
        for unc in filter(
            lambda x: x.startswith(("JER", "JES")),
            awkward.fields(corrected_jets),
        ):
            corrt1_up_dpx = corrt1_dpx
            corrt1_up_dpy = corrt1_dpy
            corrt1_down_dpx = corrt1_dpx
            corrt1_down_dpy = corrt1_dpy

            if (
                corrt1jets is not None
                and unc.startswith("JER")
                and unc in corrt1jets.fields
            ):
                smear_field = self.name_map.get("CorrT1JetJERSmearFactor")
                if smear_field is not None:

                    def compute_corrt1_jer_variants(
                        ct1jets, f_L1, f_L1L2L3, scale_up, scale_down
                    ):
                        up = _compute_corrt1_type1_deltas_with_factors(
                            ct1jets,
                            self.name_map,
                            f_L1,
                            f_L1L2L3,
                            pt_scale_factor=scale_up,
                        )
                        down = _compute_corrt1_type1_deltas_with_factors(
                            ct1jets,
                            self.name_map,
                            f_L1,
                            f_L1L2L3,
                            pt_scale_factor=scale_down,
                        )
                        return awkward.zip(
                            {
                                "up_delta_px": up.delta_px,
                                "up_delta_py": up.delta_py,
                                "down_delta_px": down.delta_px,
                                "down_delta_py": down.delta_py,
                            },
                            depth_limit=1,
                        )

                    corrt1_jer = maybe_map_partitions(
                        compute_corrt1_jer_variants,
                        corrt1jets,
                        corrt1_factor_L1,
                        corrt1_factor_L1L2L3,
                        corrt1jets[unc].up[smear_field],
                        corrt1jets[unc].down[smear_field],
                        label=f"type1_corrt1_{unc}_deltas",
                    )
                    corrt1_up_dpx = corrt1_jer.up_delta_px
                    corrt1_up_dpy = corrt1_jer.up_delta_py
                    corrt1_down_dpx = corrt1_jer.down_delta_px
                    corrt1_down_dpy = corrt1_jer.down_delta_py

            jes_uncertainty = self.jes_uncertainties.get(unc)
            if corrt1jets is not None and jes_uncertainty is not None:

                def compute_corrt1_jes_variants(
                    ct1jets,
                    f_L1,
                    f_L1L2L3,
                    smear_factor,
                    _jes=jes_uncertainty,
                ):
                    return _compute_type1_jes_deltas_with_factors(
                        ct1jets,
                        self.name_map,
                        f_L1,
                        f_L1L2L3,
                        _jes,
                        nominal_smear_factor=smear_factor,
                        is_corrt1=True,
                    )

                corrt1_jes = maybe_map_partitions(
                    compute_corrt1_jes_variants,
                    corrt1jets,
                    corrt1_factor_L1,
                    corrt1_factor_L1L2L3,
                    corrt1_nominal_smear_factor,
                    label=f"type1_corrt1_{unc}_deltas",
                )
                corrt1_up_dpx = corrt1_jes.up_delta_px
                corrt1_up_dpy = corrt1_jes.up_delta_py
                corrt1_down_dpx = corrt1_jes.down_delta_px
                corrt1_down_dpy = corrt1_jes.down_delta_py

            def build_jes_jer_variant(
                met_record,
                rmet,
                jets_nominal,
                jets_var_up,
                jets_var_down,
                f_L1,
                f_L1L2L3,
                nominal_smear_factor,
                ct1_up_dpx,
                ct1_up_dpy,
                ct1_down_dpx,
                ct1_down_dpy,
                _jes_source=jes_uncertainty,
            ):
                raw_pt = rmet[self.name_map["RawMETpt"]]
                raw_phi = rmet[self.name_map["RawMETphi"]]

                jes_source = _jes_source

                if jes_source is not None:
                    # Exact CMSJMECalculators JES propagation for ordinary jets.
                    jet_jes = _compute_type1_jes_deltas_with_factors(
                        jets_nominal,
                        self.name_map,
                        f_L1,
                        f_L1L2L3,
                        jes_source,
                        nominal_smear_factor=nominal_smear_factor,
                    )
                    up_total_dpx = jet_jes.up_delta_px
                    up_total_dpy = jet_jes.up_delta_py
                    down_total_dpx = jet_jes.down_delta_px
                    down_total_dpy = jet_jes.down_delta_py
                else:
                    # Fallback for JER or JES sources for which no direct
                    # uncertainty payload was supplied.
                    nominal_pt = jets_nominal[self.name_map["JetPt"]]
                    up_pt = jets_var_up[self.name_map["JetPt"]]
                    down_pt = jets_var_down[self.name_map["JetPt"]]

                    reference_pt = nominal_pt
                    if self.is_t1_smeared_met:
                        jec_pt_field = self.name_map["JetPt"] + "_jec"
                        if jec_pt_field in jets_nominal.fields:
                            reference_pt = jets_nominal[jec_pt_field]
                    safe_reference = awkward.where(reference_pt != 0, reference_pt, 1.0)
                    scale_up = up_pt / safe_reference
                    scale_down = down_pt / safe_reference

                    up_deltas = _compute_jet_type1_deltas_with_factors(
                        jets_nominal,
                        self.name_map,
                        f_L1,
                        f_L1L2L3,
                        pt_scale_factor=scale_up,
                    )
                    down_deltas = _compute_jet_type1_deltas_with_factors(
                        jets_nominal,
                        self.name_map,
                        f_L1,
                        f_L1L2L3,
                        pt_scale_factor=scale_down,
                    )

                    up_total_dpx = up_deltas.delta_px
                    up_total_dpy = up_deltas.delta_py
                    down_total_dpx = down_deltas.delta_px
                    down_total_dpy = down_deltas.delta_py

                if ct1_up_dpx is not None:
                    up_total_dpx = up_total_dpx + ct1_up_dpx
                    up_total_dpy = up_total_dpy + ct1_up_dpy
                    down_total_dpx = down_total_dpx + ct1_down_dpx
                    down_total_dpy = down_total_dpy + ct1_down_dpy

                var_up = corrected_type1_met(
                    raw_pt, raw_phi, up_total_dpx, up_total_dpy
                )
                var_down = corrected_type1_met(
                    raw_pt, raw_phi, down_total_dpx, down_total_dpy
                )

                up_out = awkward.with_field(
                    met_record, var_up.pt, self.name_map["METpt"]
                )
                up_out = awkward.with_field(up_out, var_up.phi, self.name_map["METphi"])
                down_out = awkward.with_field(
                    met_record, var_down.pt, self.name_map["METpt"]
                )
                down_out = awkward.with_field(
                    down_out, var_down.phi, self.name_map["METphi"]
                )
                return awkward.zip(
                    {"up": up_out, "down": down_out},
                    depth_limit=1,
                    with_name="METSystematic",
                )

            out_dict[unc] = maybe_map_partitions(
                build_jes_jer_variant,
                MET,
                raw_met,
                corrected_jets,
                corrected_jets[unc].up,
                corrected_jets[unc].down,
                jet_factor_L1,
                jet_factor_L1L2L3,
                jet_nominal_smear_factor,
                corrt1_up_dpx,
                corrt1_up_dpy,
                corrt1_down_dpx,
                corrt1_down_dpy,
                label=f"type1_{unc}_met",
            )

        out_parms = out.layout.parameters
        out = awkward.zip(
            out_dict, depth_limit=1, parameters=out_parms, behavior=out.behavior
        )

        return out

    def uncertainties(self):
        """
        Returns a list of the sources of uncertainty included in the stack.

        Returns
        -------
            list[str]
                A list of the sources of uncertainty.
        """
        return ["MET_UnclusteredEnergy"]
