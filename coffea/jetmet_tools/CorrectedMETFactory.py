import awkward
import numpy
from copy import copy


def corrected_polar_met(met_pt, met_phi, jet_pt, jet_phi, jet_pt_orig, deltas=None):
    sj, cj = numpy.sin(jet_phi), numpy.cos(jet_phi)
    x = met_pt * numpy.cos(met_phi) - awkward.sum(
        jet_pt * cj - jet_pt_orig * cj, axis=1
    )
    y = met_pt * numpy.sin(met_phi) - awkward.sum(
        jet_pt * sj - jet_pt_orig * sj, axis=1
    )
    if deltas:
        positive, dx, dy = deltas
        x = x + dx if positive else x - dx
        y = y + dy if positive else y - dy
    return awkward.zip({"pt": numpy.hypot(x, y), "phi": numpy.arctan2(y, x)})


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

    Flattens jagged arrays to numpy before calling JEC correctors
    (forces eager evaluation, avoiding awkward.virtual), then unflattens back.

    Returns
    -------
    factor_L1, factor_L1L2L3 : awkward.Array
        Jagged arrays of JEC factors matching jets shape.
    """
    jet_pt = jets[name_map["JetPt"]]
    raw_factor = jets[name_map["JetRawFactor"]]
    jet_pt_raw = jet_pt * (1.0 - raw_factor)

    counts = awkward.num(jets)

    # Build and flatten JEC inputs for L1 — use numpy.asarray for eager eval
    jec_inputs_L1 = {}
    for k in jec_L1.signature:
        if k == "JetPt":
            jec_inputs_L1[k] = numpy.asarray(awkward.flatten(jet_pt_raw))
        else:
            jec_inputs_L1[k] = numpy.asarray(awkward.flatten(jets[name_map[k]]))
    flat_factor_L1 = jec_L1.getCorrection(**jec_inputs_L1)
    factor_L1 = awkward.unflatten(flat_factor_L1, counts)

    # Build and flatten JEC inputs for L1L2L3
    jec_inputs_L1L2L3 = {}
    for k in jec_L1L2L3.signature:
        if k == "JetPt":
            jec_inputs_L1L2L3[k] = numpy.asarray(awkward.flatten(jet_pt_raw))
        else:
            jec_inputs_L1L2L3[k] = numpy.asarray(
                awkward.flatten(jets[name_map[k]])
            )
    flat_factor_L1L2L3 = jec_L1L2L3.getCorrection(**jec_inputs_L1L2L3)
    factor_L1L2L3 = awkward.unflatten(flat_factor_L1L2L3, counts)

    return factor_L1, factor_L1L2L3


def _compute_corrt1_jec_factors(corrt1jets, name_map, jec_L1, jec_L1L2L3):
    """Compute L1 and L1L2L3 JEC factors for the CorrT1METJet collection.

    Flattens jagged arrays to numpy before calling JEC correctors
    (forces eager evaluation), then unflattens back.

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
            jec_inputs_L1[k] = numpy.asarray(awkward.flatten(raw_pt))
        elif k == "JetEta":
            jec_inputs_L1[k] = numpy.asarray(
                awkward.flatten(corrt1jets[name_map["CorrT1JetEta"]])
            )
        elif k == "JetA":
            jec_inputs_L1[k] = numpy.asarray(
                awkward.flatten(corrt1jets[name_map["CorrT1JetArea"]])
            )
        elif k == "Rho":
            jec_inputs_L1[k] = numpy.asarray(
                awkward.flatten(corrt1jets[name_map["Rho"]])
            )
        else:
            jec_inputs_L1[k] = numpy.asarray(
                awkward.flatten(corrt1jets[name_map[k]])
            )
    flat_factor_L1 = jec_L1.getCorrection(**jec_inputs_L1)
    factor_L1 = awkward.unflatten(flat_factor_L1, counts)

    # Build and flatten JEC inputs for L1L2L3
    jec_inputs_L1L2L3 = {}
    for k in jec_L1L2L3.signature:
        if k == "JetPt":
            jec_inputs_L1L2L3[k] = numpy.asarray(awkward.flatten(raw_pt))
        elif k == "JetEta":
            jec_inputs_L1L2L3[k] = numpy.asarray(
                awkward.flatten(corrt1jets[name_map["CorrT1JetEta"]])
            )
        elif k == "JetA":
            jec_inputs_L1L2L3[k] = numpy.asarray(
                awkward.flatten(corrt1jets[name_map["CorrT1JetArea"]])
            )
        elif k == "Rho":
            jec_inputs_L1L2L3[k] = numpy.asarray(
                awkward.flatten(corrt1jets[name_map["Rho"]])
            )
        else:
            jec_inputs_L1L2L3[k] = numpy.asarray(
                awkward.flatten(corrt1jets[name_map[k]])
            )
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


class CorrectedMETFactory(object):
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

    def build(self, MET, corrected_jets, lazy_cache, RawMET=None, CorrT1METJets=None):
        if not isinstance(MET, awkward.highlevel.Array) or not isinstance(
            corrected_jets, awkward.highlevel.Array
        ):
            raise Exception(
                "'MET' and 'corrected_jets' must be an awkward array of some kind!"
            )

        if self.type1_mode:
            if RawMET is None:
                raise ValueError(
                    "RawMET is required when CorrectedMETFactory is in Type-1 mode "
                    "(jec_L1L2L3 and jec_L1 were provided)."
                )
            if CorrT1METJets is not None:
                # Validate CorrT1 name_map keys
                for name in _TYPE1_CORRT1_KEYS:
                    if name not in self.name_map or self.name_map[name] is None:
                        raise ValueError(
                            f"There is no name mapping for {name}, which is needed "
                            f"when CorrT1METJets is provided"
                        )
            return self._build_type1(MET, corrected_jets, RawMET, CorrT1METJets)

        # --- Legacy path ---
        return self._build_legacy(MET, corrected_jets, lazy_cache)

    def _build_legacy(self, MET, corrected_jets, lazy_cache):
        """Legacy MET correction path — identical to the original implementation."""
        if lazy_cache is None:
            raise Exception(
                "CorrectedMETFactory requires a awkward-array cache to function correctly."
            )
        lazy_cache = awkward._util.MappingProxy.maybe_wrap(lazy_cache)

        length = len(MET)
        form = awkward.forms.RecordForm(
            {
                "pt": MET[self.name_map["METpt"]].layout.form,
                "phi": MET[self.name_map["METphi"]].layout.form,
            },
        )

        def make_variant(*args):
            variant = copy(MET)
            corrected_met = awkward.virtual(
                corrected_polar_met,
                args=args,
                length=length,
                form=form,
                cache=lazy_cache,
            )
            variant[self.name_map["METpt"]] = awkward.virtual(
                lambda: awkward.materialized(corrected_met.pt),
                length=length,
                form=form.contents["pt"],
                cache=lazy_cache,
            )
            variant[self.name_map["METphi"]] = awkward.virtual(
                lambda: awkward.materialized(corrected_met.phi),
                length=length,
                form=form.contents["phi"],
                cache=lazy_cache,
            )
            return variant

        def lazy_variant(unc, metpt, metphi, jetpt, jetphi, jetpt_orig):
            return awkward.zip(
                {
                    "up": make_variant(
                        MET[metpt],
                        MET[metphi],
                        corrected_jets[unc].up[jetpt],
                        corrected_jets[unc].up[jetphi],
                        corrected_jets[unc].up[jetpt_orig],
                    ),
                    "down": make_variant(
                        MET[metpt],
                        MET[metphi],
                        corrected_jets[unc].down[jetpt],
                        corrected_jets[unc].down[jetphi],
                        corrected_jets[unc].down[jetpt_orig],
                    ),
                },
                depth_limit=1,
                with_name="METSystematic",
            )

        out = make_variant(
            MET[self.name_map["METpt"]],
            MET[self.name_map["METphi"]],
            corrected_jets[self.name_map["JetPt"]],
            corrected_jets[self.name_map["JetPhi"]],
            corrected_jets[self.name_map["JetPt"] + "_orig"],
        )
        out[self.name_map["METpt"] + "_orig"] = MET[self.name_map["METpt"]]
        out[self.name_map["METphi"] + "_orig"] = MET[self.name_map["METphi"]]

        out_dict = {field: out[field] for field in awkward.fields(out)}

        out_dict["MET_UnclusteredEnergy"] = awkward.zip(
            {
                "up": make_variant(
                    MET[self.name_map["METpt"]],
                    MET[self.name_map["METphi"]],
                    corrected_jets[self.name_map["JetPt"]],
                    corrected_jets[self.name_map["JetPhi"]],
                    corrected_jets[self.name_map["JetPt"] + "_orig"],
                    (
                        True,
                        MET[self.name_map["UnClusteredEnergyDeltaX"]],
                        MET[self.name_map["UnClusteredEnergyDeltaY"]],
                    ),
                ),
                "down": make_variant(
                    MET[self.name_map["METpt"]],
                    MET[self.name_map["METphi"]],
                    corrected_jets[self.name_map["JetPt"]],
                    corrected_jets[self.name_map["JetPhi"]],
                    corrected_jets[self.name_map["JetPt"] + "_orig"],
                    (
                        False,
                        MET[self.name_map["UnClusteredEnergyDeltaX"]],
                        MET[self.name_map["UnClusteredEnergyDeltaY"]],
                    ),
                ),
            },
            depth_limit=1,
            with_name="METSystematic",
        )

        for unc in filter(
            lambda x: x.startswith(("JER", "JES")), awkward.fields(corrected_jets)
        ):
            out_dict[unc] = awkward.virtual(
                lazy_variant,
                args=(
                    unc,
                    self.name_map["METpt"],
                    self.name_map["METphi"],
                    self.name_map["JetPt"],
                    self.name_map["JetPhi"],
                    self.name_map["JetPt"] + "_orig",
                ),
                length=length,
                cache={},
            )

        out_parms = out.layout.parameters
        out = awkward.zip(
            out_dict, depth_limit=1, parameters=out_parms, behavior=out.behavior
        )

        return out

    def _build_type1(self, MET, corrected_jets, raw_met, corrt1jets):
        """Type-1 MET correction path — all computations are eager (no awkward.virtual)."""

        # --- Compute JEC factors once (reused for all variations) ---
        jet_factor_L1, jet_factor_L1L2L3 = _compute_jec_factors(
            corrected_jets, self.name_map, self.jec_L1, self.jec_L1L2L3
        )

        # --- Compute nominal Jet deltas ---
        jet_deltas = _compute_jet_type1_deltas_with_factors(
            corrected_jets, self.name_map, jet_factor_L1, jet_factor_L1L2L3
        )
        total_dpx = jet_deltas.delta_px
        total_dpy = jet_deltas.delta_py

        # --- Compute CorrT1METJet deltas (if provided) ---
        corrt1_dpx = None
        corrt1_dpy = None
        if corrt1jets is not None:
            corrt1_factor_L1, corrt1_factor_L1L2L3 = _compute_corrt1_jec_factors(
                corrt1jets, self.name_map, self.jec_L1, self.jec_L1L2L3
            )
            corrt1_deltas = _compute_corrt1_type1_deltas_with_factors(
                corrt1jets, self.name_map, corrt1_factor_L1, corrt1_factor_L1L2L3
            )
            corrt1_dpx = corrt1_deltas.delta_px
            corrt1_dpy = corrt1_deltas.delta_py
            total_dpx = total_dpx + corrt1_dpx
            total_dpy = total_dpy + corrt1_dpy

        # --- Nominal corrected MET ---
        raw_pt = raw_met[self.name_map["RawMETpt"]]
        raw_phi = raw_met[self.name_map["RawMETphi"]]
        variation = corrected_type1_met(raw_pt, raw_phi, total_dpx, total_dpy)

        out = copy(MET)
        out[self.name_map["METpt"]] = variation.pt
        out[self.name_map["METphi"]] = variation.phi
        out[self.name_map["METpt"] + "_orig"] = raw_pt
        out[self.name_map["METphi"] + "_orig"] = raw_phi

        out_dict = {field: out[field] for field in awkward.fields(out)}

        # --- Unclustered energy systematics ---
        dx = MET[self.name_map["UnClusteredEnergyDeltaX"]]
        dy = MET[self.name_map["UnClusteredEnergyDeltaY"]]

        var_up = corrected_type1_met(
            raw_pt, raw_phi, total_dpx, total_dpy, positive=True, dx=dx, dy=dy
        )
        var_down = corrected_type1_met(
            raw_pt, raw_phi, total_dpx, total_dpy, positive=False, dx=dx, dy=dy
        )

        up_out = copy(MET)
        up_out[self.name_map["METpt"]] = var_up.pt
        up_out[self.name_map["METphi"]] = var_up.phi
        down_out = copy(MET)
        down_out[self.name_map["METpt"]] = var_down.pt
        down_out[self.name_map["METphi"]] = var_down.phi

        out_dict["MET_UnclusteredEnergy"] = awkward.zip(
            {"up": up_out, "down": down_out},
            depth_limit=1,
            with_name="METSystematic",
        )

        # --- JES/JER systematics ---
        for unc in filter(
            lambda x: x.startswith(("JER", "JES")), awkward.fields(corrected_jets)
        ):
            # Compute scale factors: varied_pt / nominal_pt
            nominal_pt = corrected_jets[self.name_map["JetPt"]]
            up_pt = corrected_jets[unc].up[self.name_map["JetPt"]]
            down_pt = corrected_jets[unc].down[self.name_map["JetPt"]]

            # Protect against division by zero
            safe_nominal = awkward.where(nominal_pt > 0, nominal_pt, 1.0)
            scale_up = up_pt / safe_nominal
            scale_down = down_pt / safe_nominal

            # Recompute Jet deltas with varied scale, reusing JEC factors
            up_deltas = _compute_jet_type1_deltas_with_factors(
                corrected_jets,
                self.name_map,
                jet_factor_L1,
                jet_factor_L1L2L3,
                pt_scale_factor=scale_up,
            )
            down_deltas = _compute_jet_type1_deltas_with_factors(
                corrected_jets,
                self.name_map,
                jet_factor_L1,
                jet_factor_L1L2L3,
                pt_scale_factor=scale_down,
            )

            # Add CorrT1 nominal contribution (stays constant)
            if corrt1_dpx is not None:
                up_total_dpx = up_deltas.delta_px + corrt1_dpx
                up_total_dpy = up_deltas.delta_py + corrt1_dpy
                down_total_dpx = down_deltas.delta_px + corrt1_dpx
                down_total_dpy = down_deltas.delta_py + corrt1_dpy
            else:
                up_total_dpx = up_deltas.delta_px
                up_total_dpy = up_deltas.delta_py
                down_total_dpx = down_deltas.delta_px
                down_total_dpy = down_deltas.delta_py

            unc_var_up = corrected_type1_met(
                raw_pt, raw_phi, up_total_dpx, up_total_dpy
            )
            unc_var_down = corrected_type1_met(
                raw_pt, raw_phi, down_total_dpx, down_total_dpy
            )

            unc_up_out = copy(MET)
            unc_up_out[self.name_map["METpt"]] = unc_var_up.pt
            unc_up_out[self.name_map["METphi"]] = unc_var_up.phi
            unc_down_out = copy(MET)
            unc_down_out[self.name_map["METpt"]] = unc_var_down.pt
            unc_down_out[self.name_map["METphi"]] = unc_var_down.phi

            out_dict[unc] = awkward.zip(
                {"up": unc_up_out, "down": unc_down_out},
                depth_limit=1,
                with_name="METSystematic",
            )

        out_parms = out.layout.parameters
        out = awkward.zip(
            out_dict, depth_limit=1, parameters=out_parms, behavior=out.behavior
        )

        return out

    def uncertainties(self):
        return ["MET_UnclusteredEnergy"]
