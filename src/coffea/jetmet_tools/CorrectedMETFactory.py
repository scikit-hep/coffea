import awkward
import dask_awkward
import numpy

from coffea.util import maybe_map_partitions


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


# Required name_map keys for Type-1 mode (Jet collection)
_TYPE1_JET_KEYS = [
    "RawMETpt",
    "RawMETphi",
    "JetRawFactor",
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


def _compute_jec_factors(jets, name_map, jec_L1, jec_L1L2L3):
    """Compute L1 and L1L2L3 JEC factors for the Jet collection.

    Flattens jagged arrays before calling JEC correctors (required for
    CorrectionLibJEC compatibility), then unflattens back.

    Returns
    -------
    factor_L1, factor_L1L2L3 : awkward.Array
        Jagged arrays of JEC factors matching jets shape.
    """
    jet_pt = jets[name_map["JetPt"]]
    raw_factor = jets[name_map["JetRawFactor"]]
    jet_pt_raw = jet_pt * (1.0 - raw_factor)

    counts = awkward.num(jets)

    # Build and flatten JEC inputs for L1
    jec_inputs_L1 = {}
    for k in jec_L1.signature:
        if k == "JetPt":
            jec_inputs_L1[k] = awkward.flatten(jet_pt_raw)
        else:
            jec_inputs_L1[k] = awkward.flatten(jets[name_map[k]])
    flat_factor_L1 = jec_L1.getCorrection(**jec_inputs_L1)
    factor_L1 = awkward.unflatten(flat_factor_L1, counts)

    # Build and flatten JEC inputs for L1L2L3
    jec_inputs_L1L2L3 = {}
    for k in jec_L1L2L3.signature:
        if k == "JetPt":
            jec_inputs_L1L2L3[k] = awkward.flatten(jet_pt_raw)
        else:
            jec_inputs_L1L2L3[k] = awkward.flatten(jets[name_map[k]])
    flat_factor_L1L2L3 = jec_L1L2L3.getCorrection(**jec_inputs_L1L2L3)
    factor_L1L2L3 = awkward.unflatten(flat_factor_L1L2L3, counts)

    return factor_L1, factor_L1L2L3


def _compute_corrt1_jec_factors(corrt1jets, name_map, jec_L1, jec_L1L2L3):
    """Compute L1 and L1L2L3 JEC factors for the CorrT1METJet collection.

    Flattens jagged arrays before calling JEC correctors (required for
    CorrectionLibJEC compatibility), then unflattens back.

    Returns
    -------
    factor_L1, factor_L1L2L3 : awkward.Array
        Jagged arrays of JEC factors matching corrt1jets shape.
    """
    raw_pt = corrt1jets[name_map["CorrT1JetPt"]]
    counts = awkward.num(corrt1jets)

    # Build and flatten JEC inputs for L1
    jec_inputs_L1 = {}
    for k in jec_L1.signature:
        if k == "JetPt":
            jec_inputs_L1[k] = awkward.flatten(raw_pt)
        elif k == "JetEta":
            jec_inputs_L1[k] = awkward.flatten(corrt1jets[name_map["CorrT1JetEta"]])
        elif k == "JetA":
            jec_inputs_L1[k] = awkward.flatten(corrt1jets[name_map["CorrT1JetArea"]])
        elif k == "Rho":
            jec_inputs_L1[k] = awkward.flatten(corrt1jets[name_map["Rho"]])
        else:
            jec_inputs_L1[k] = awkward.flatten(corrt1jets[name_map[k]])
    flat_factor_L1 = jec_L1.getCorrection(**jec_inputs_L1)
    factor_L1 = awkward.unflatten(flat_factor_L1, counts)

    # Build and flatten JEC inputs for L1L2L3
    jec_inputs_L1L2L3 = {}
    for k in jec_L1L2L3.signature:
        if k == "JetPt":
            jec_inputs_L1L2L3[k] = awkward.flatten(raw_pt)
        elif k == "JetEta":
            jec_inputs_L1L2L3[k] = awkward.flatten(corrt1jets[name_map["CorrT1JetEta"]])
        elif k == "JetA":
            jec_inputs_L1L2L3[k] = awkward.flatten(corrt1jets[name_map["CorrT1JetArea"]])
        elif k == "Rho":
            jec_inputs_L1L2L3[k] = awkward.flatten(corrt1jets[name_map["Rho"]])
        else:
            jec_inputs_L1L2L3[k] = awkward.flatten(corrt1jets[name_map[k]])
    flat_factor_L1L2L3 = jec_L1L2L3.getCorrection(**jec_inputs_L1L2L3)
    factor_L1L2L3 = awkward.unflatten(flat_factor_L1L2L3, counts)

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
        If provided, multiply pt_noMuL1L2L3 by this factor (for JES/JER variations).
    """
    # Step 1: muon-subtracted raw pT and phi
    jet_pt = jets[name_map["JetPt"]]
    raw_factor = jets[name_map["JetRawFactor"]]
    muon_substr_factor = jets[name_map["JetMuonSubtrFactor"]]
    muon_substr_dphi = jets[name_map["JetMuonSubtrDeltaPhi"]]
    jet_phi = jets[name_map["JetPhi"]]

    jet_pt_raw = jet_pt * (1.0 - raw_factor)
    pt_noMuRaw = jet_pt_raw * (1.0 - muon_substr_factor)
    phi_noMuRaw = muon_substr_dphi + jet_phi

    # Step 2: apply pre-computed JEC factors
    pt_noMuL1 = pt_noMuRaw * factor_L1
    pt_noMuL1L2L3 = pt_noMuRaw * factor_L1L2L3

    # Apply variation scale factor if provided
    if pt_scale_factor is not None:
        pt_noMuL1L2L3 = pt_noMuL1L2L3 * pt_scale_factor

    # Step 3: selection cuts
    chEmEF = jets[name_map["JetChEmEF"]]
    neEmEF = jets[name_map["JetNeEmEF"]]
    mask = (pt_noMuL1L2L3 > 15.0) & ((chEmEF + neEmEF) < 0.9)

    # Step 4: vectorial sum of (pt_noMuL1L2L3 - pt_noMuL1) for selected jets
    diff_pt = awkward.where(mask, pt_noMuL1L2L3 - pt_noMuL1, 0.0)
    delta_px = awkward.sum(diff_pt * numpy.cos(phi_noMuRaw), axis=1)
    delta_py = awkward.sum(diff_pt * numpy.sin(phi_noMuRaw), axis=1)

    return awkward.zip({"delta_px": delta_px, "delta_py": delta_py}, depth_limit=1)


def _compute_corrt1_type1_deltas_with_factors(
    corrt1jets, name_map, factor_L1, factor_L1L2L3
):
    """Compute per-jet Type-1 MET correction deltas for CorrT1METJet using pre-computed JEC factors.

    Returns an awkward record with fields ``delta_px`` and ``delta_py``.
    """
    # Step 1: muon-subtracted raw pT and phi
    raw_pt = corrt1jets[name_map["CorrT1JetPt"]]
    muon_substr_factor = corrt1jets[name_map["CorrT1JetMuonSubtrFactor"]]
    muon_substr_dphi = corrt1jets[name_map["CorrT1JetMuonSubtrDeltaPhi"]]
    jet_phi = corrt1jets[name_map["CorrT1JetPhi"]]

    pt_noMuRaw = raw_pt * (1.0 - muon_substr_factor)
    phi_noMuRaw = muon_substr_dphi + jet_phi

    # Step 2: apply pre-computed JEC factors
    pt_noMuL1 = pt_noMuRaw * factor_L1
    pt_noMuL1L2L3 = pt_noMuRaw * factor_L1L2L3

    # Step 3: selection cuts
    emEF = corrt1jets[name_map["CorrT1JetEmEF"]]
    mask = (pt_noMuL1L2L3 > 15.0) & (emEF < 0.9)

    # Step 4: vectorial sum
    diff_pt = awkward.where(mask, pt_noMuL1L2L3 - pt_noMuL1, 0.0)
    delta_px = awkward.sum(diff_pt * numpy.cos(phi_noMuRaw), axis=1)
    delta_py = awkward.sum(diff_pt * numpy.sin(phi_noMuRaw), axis=1)

    return awkward.zip({"delta_px": delta_px, "delta_py": delta_py}, depth_limit=1)


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
    """

    def __init__(self, name_map, jec_L1L2L3=None, jec_L1=None):
        # Validate that both or neither JEC corrector is provided
        if (jec_L1L2L3 is None) != (jec_L1 is None):
            raise ValueError(
                "Both jec_L1L2L3 and jec_L1 must be provided together, or neither."
            )

        self.type1_mode = jec_L1L2L3 is not None
        self.jec_L1L2L3 = jec_L1L2L3
        self.jec_L1 = jec_L1

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
        if not isinstance(
            in_MET, (awkward.highlevel.Array, dask_awkward.Array)
        ) or not isinstance(
            in_corrected_jets, (awkward.highlevel.Array, dask_awkward.Array)
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
        def compute_nominal_jet_deltas(jets, f_L1, f_L1L2L3):
            return _compute_jet_type1_deltas_with_factors(
                jets, self.name_map, f_L1, f_L1L2L3
            )

        jet_deltas = maybe_map_partitions(
            compute_nominal_jet_deltas,
            corrected_jets,
            jet_factor_L1,
            jet_factor_L1L2L3,
            label="type1_jet_deltas",
        )
        jet_dpx = jet_deltas.delta_px
        jet_dpy = jet_deltas.delta_py

        # Total deltas start with Jet contribution
        total_dpx = jet_dpx
        total_dpy = jet_dpy

        # --- Compute CorrT1METJet deltas (if provided) ---
        corrt1_dpx = None
        corrt1_dpy = None
        if corrt1jets is not None:

            def compute_corrt1_deltas(ct1jets):
                f_L1, f_L1L2L3 = _compute_corrt1_jec_factors(
                    ct1jets, self.name_map, self.jec_L1, self.jec_L1L2L3
                )
                return _compute_corrt1_type1_deltas_with_factors(
                    ct1jets, self.name_map, f_L1, f_L1L2L3
                )

            corrt1_deltas = maybe_map_partitions(
                compute_corrt1_deltas,
                corrt1jets,
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

            up_out = awkward.with_field(
                met_record, var_up.pt, self.name_map["METpt"]
            )
            up_out = awkward.with_field(
                up_out, var_up.phi, self.name_map["METphi"]
            )
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
        # For each JES/JER source, the varied jet pT comes from corrected_jets[unc].up/down.
        # We compute the ratio varied_pt / nominal_pt and apply it to pt_noMuL1L2L3.
        # CorrT1METJet contribution stays nominal.
        # JEC factors are reused from the nominal computation (no recomputation).
        for unc in filter(
            lambda x: x.startswith(("JER", "JES")), awkward.fields(corrected_jets)
        ):

            def build_jes_jer_variant(
                met_record,
                rmet,
                jets_nominal,
                jets_var_up,
                jets_var_down,
                f_L1,
                f_L1L2L3,
                corrt1_dpx_val,
                corrt1_dpy_val,
                _unc=unc,
            ):
                raw_pt = rmet[self.name_map["RawMETpt"]]
                raw_phi = rmet[self.name_map["RawMETphi"]]

                # Compute scale factors: varied_pt / nominal_pt
                nominal_pt = jets_nominal[self.name_map["JetPt"]]
                up_pt = jets_var_up[self.name_map["JetPt"]]
                down_pt = jets_var_down[self.name_map["JetPt"]]

                # Protect against division by zero
                safe_nominal = awkward.where(nominal_pt > 0, nominal_pt, 1.0)
                scale_up = up_pt / safe_nominal
                scale_down = down_pt / safe_nominal

                # Recompute Jet deltas with varied scale, reusing JEC factors
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
                up_jet_dpx = up_deltas.delta_px
                up_jet_dpy = up_deltas.delta_py
                down_jet_dpx = down_deltas.delta_px
                down_jet_dpy = down_deltas.delta_py

                # Add CorrT1 nominal contribution (stays constant)
                if corrt1_dpx_val is not None:
                    up_total_dpx = up_jet_dpx + corrt1_dpx_val
                    up_total_dpy = up_jet_dpy + corrt1_dpy_val
                    down_total_dpx = down_jet_dpx + corrt1_dpx_val
                    down_total_dpy = down_jet_dpy + corrt1_dpy_val
                else:
                    up_total_dpx = up_jet_dpx
                    up_total_dpy = up_jet_dpy
                    down_total_dpx = down_jet_dpx
                    down_total_dpy = down_jet_dpy

                var_up = corrected_type1_met(
                    raw_pt, raw_phi, up_total_dpx, up_total_dpy
                )
                var_down = corrected_type1_met(
                    raw_pt, raw_phi, down_total_dpx, down_total_dpy
                )

                up_out = awkward.with_field(
                    met_record, var_up.pt, self.name_map["METpt"]
                )
                up_out = awkward.with_field(
                    up_out, var_up.phi, self.name_map["METphi"]
                )
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
                corrt1_dpx,
                corrt1_dpy,
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
