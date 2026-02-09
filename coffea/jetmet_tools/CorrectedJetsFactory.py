import awkward
import numpy
import warnings
from functools import partial
import operator
from coffea.jetmet_tools import JECStack

_stack_parts = ["jec", "junc", "jer", "jersf"]
_MIN_JET_ENERGY = numpy.array(1e-2, dtype=numpy.float32)
_JERSF_FORM = {
    "class": "NumpyArray",
    "inner_shape": [3],
    "itemsize": 4,
    "format": "f",
    "primitive": "float32",
}


def rewrap_recordarray(layout, depth, data):
    if isinstance(layout, awkward.layout.RecordArray):
        return lambda: data
    return None


def awkward_rewrap(arr, like_what, gfunc):
    behavior = awkward._util.behaviorof(like_what)
    func = partial(gfunc, data=arr.layout)
    layout = awkward.operations.convert.to_layout(like_what)
    newlayout = awkward._util.recursively_apply(layout, func)
    return awkward._util.wrap(newlayout, behavior=behavior)


def rand_gauss(item, randomstate):
    def getfunction(layout, depth):
        if isinstance(layout, awkward.layout.NumpyArray) or not isinstance(
            layout, (awkward.layout.Content, awkward.partition.PartitionedArray)
        ):
            return lambda: awkward.layout.NumpyArray(
                randomstate.normal(size=len(layout)).astype(numpy.float32)
            )
        return None

    out = awkward._util.recursively_apply(
        awkward.operations.convert.to_layout(item), getfunction
    )
    assert out is not None
    return awkward._util.wrap(out, awkward._util.behaviorof(item))


def jer_smear(
    variation,
    forceStochastic,
    pt_gen,
    jetPt,
    etaJet,
    jet_energy_resolution,
    jet_resolution_rand_gauss,
    jet_energy_resolution_scale_factor,
):
    pt_gen = pt_gen if not forceStochastic else None
    if not isinstance(jetPt, awkward.highlevel.Array):
        raise Exception("'jetPt' must be an awkward array of some kind!")
    if forceStochastic:
        pt_gen = awkward.without_parameters(awkward.zeros_like(jetPt))

    jersmear = jet_energy_resolution * jet_resolution_rand_gauss
    jersf = jet_energy_resolution_scale_factor[:, variation]
    deltaPtRel = (jetPt - pt_gen) / jetPt
    doHybrid = (pt_gen > 0) & (numpy.abs(deltaPtRel) < 3 * jet_energy_resolution)
    detSmear = 1 + (jersf - 1) * deltaPtRel
    stochSmear = 1 + numpy.sqrt(numpy.maximum(jersf**2 - 1, 0)) * jersmear

    min_jet_pt = _MIN_JET_ENERGY / numpy.cosh(etaJet)
    min_jet_pt_corr = min_jet_pt / jetPt
    smearfact = awkward.where(doHybrid, detSmear, stochSmear)
    smearfact = awkward.where(
        (smearfact * jetPt) < min_jet_pt, min_jet_pt_corr, smearfact
    )

    def getfunction(layout, depth):
        if isinstance(layout, awkward.layout.NumpyArray) or not isinstance(
            layout, (awkward.layout.Content, awkward.partition.PartitionedArray)
        ):
            return lambda: awkward.layout.NumpyArray(smearfact)
        return None

    smearfact = awkward._util.recursively_apply(
        awkward.operations.convert.to_layout(jetPt), getfunction
    )
    smearfact = awkward._util.wrap(smearfact, awkward._util.behaviorof(jetPt))
    return smearfact


# Wrapper function to apply jec corrections
def rawvar_jec(jecval, rawvar, lazy_cache):
    return awkward.virtual(
        operator.mul,
        args=(jecval, rawvar),
        cache=lazy_cache,
    )


def get_corr_inputs(jets, corr_obj, name_map, cache=None, corrections=None):
    """
    Helper function for getting values of input variables
    given a dictionary and a correction object.
    """

    def _maybe_from_metadata(inp):
        """Derive input values based on optional metadata hints without changing the public API."""

        meta = getattr(inp, "metadata", None) or {}

        use_raw = meta.get("isRawPt", False) or meta.get("useRawPt", False)
        use_corr = meta.get("isCorrectedPt", False) or meta.get("useCorrectedPt", False)

        if use_raw:
            raw_field = name_map.get("ptRaw", name_map.get(inp.name, inp.name))
            return awkward.flatten(jets[raw_field])

        if corrections is not None and use_corr:
            rawvar = awkward.flatten(jets[name_map[inp.name]])
            init_input_value = partial(rawvar_jec, rawvar=rawvar, lazy_cache=cache)
            return init_input_value(jecval=corrections)

        return None

    if corrections is None:
        input_values = []
        for inp in corr_obj.inputs:
            if inp.name == "systematic":
                continue

            from_meta = _maybe_from_metadata(inp)
            if from_meta is not None:
                input_values.append(from_meta)
                continue

            input_values.append(awkward.flatten(jets[name_map[inp.name]]))
    else:
        ## This is needed to propagate the previous level of corrections, before applying the next one
        input_values = []
        for inp in corr_obj.inputs:
            if inp.name == "systematic":
                continue

            from_meta = _maybe_from_metadata(inp)
            if from_meta is not None:
                input_values.append(from_meta)
                continue

            elif inp.name == "JetPt":
                rawvar = awkward.flatten(jets[name_map[inp.name]])
                init_input_value = partial(rawvar_jec, rawvar=rawvar, lazy_cache=cache)
                input_value = init_input_value(jecval=corrections)
            else:
                input_value = awkward.flatten(jets[name_map[inp.name]])
            input_values.append(input_value)
    return input_values


class CorrectedJetsFactory(object):
    def __init__(self, name_map, jec_stack):
        if not isinstance(jec_stack, JECStack):
            raise TypeError("jec_stack must be an instance of JECStack")

        self.tool = "clib" if jec_stack.use_clib else "jecstack"
        self.forceStochastic = False

        # Start from the stack-provided defaults when using correctionlib,
        # allowing user inputs to override or augment the inferred mapping.
        provided_name_map = {} if name_map is None else dict(name_map)
        user_raw_keys = {k for k in ("ptRaw", "massRaw") if k in provided_name_map}
        if self.tool == "clib":
            stack_map = dict(jec_stack.blank_name_map)
            name_map = dict(stack_map)
            name_map.update(provided_name_map)

            # Allow raw pt/mass inference unless the caller explicitly supplied
            # a non-default mapping. Passing through the stack defaults alone
            # should not pre-populate raw keys and block fallback inference.
            for raw_key in ("ptRaw", "massRaw"):
                if raw_key not in user_raw_keys:
                    name_map.pop(raw_key, None)
        else:
            name_map = provided_name_map

        # Handle name map for raw pt and mass
        pt_raw_missing = "ptRaw" not in name_map or name_map["ptRaw"] is None
        if pt_raw_missing:
            warnings.warn(
                "There is no name mapping for ptRaw,"
                " CorrectedJets will fall back to <object>.pt_raw"
                " as the raw pt field."
            )
            name_map["ptRaw"] = name_map["JetPt"] + "_raw"

        mass_raw_missing = "massRaw" not in name_map or name_map["massRaw"] is None
        if mass_raw_missing:
            warnings.warn(
                "There is no name mapping for massRaw,"
                " CorrectedJets will fall back to <object>.mass_raw"
                " as the raw mass field."
            )
            name_map["massRaw"] = name_map["JetMass"] + "_raw"

        # Only treat pt/mass as already-raw when the user explicitly indicated so
        # by mapping ptRaw to the corrected pt field. Missing raw mappings should
        # fall back to inference rather than clobbering existing raw inputs.
        self.treat_pt_as_raw = (
            "ptRaw" in provided_name_map
            and provided_name_map.get("ptRaw") is not None
            and provided_name_map.get("ptRaw") == provided_name_map.get("JetPt")
        )

        self.jec_stack = jec_stack
        self.name_map = name_map

        if self.jec_stack.use_clib:
            # For clib scenario, load corrections from json_path
            self.load_corrections_clib()
        else:
            # For non-clib scenario, use the provided corrections (e.g., JEC/JER)
            self.load_corrections_jecstack()

        if "ptGenJet" not in name_map:
            warnings.warn(
                'Input JaggedCandidateArray must have "ptGenJet" in order to apply hybrid JER smearing method. Stochastic smearing will be applied.'
            )
            self.forceStochastic = True

    def uncertainties(self):
        """Return the available JES uncertainty branch names.

        The list mirrors the public ``CorrectedJetsFactory`` contract used by
        analyses: names are ordered exactly as provided in the underlying JEC
        configuration and prefixed with ``"JES_"`` to match the fields added
        during :meth:`build`.
        """

        sources = []
        if self.tool == "clib":
            sources = [
                name.split("_")[-2] for name in self.jec_stack.jec_uncsources_clib
            ]
        else:
            junc = getattr(self.jec_stack, "junc", None)
            if junc is not None:
                sources = list(junc.levels)

        return [f"JES_{source}" for source in sources]

    def load_corrections_clib(self):
        """Load the corrections from correctionlib using the json_path in JECStack."""
        self.corrections = self.jec_stack.corrections

    def load_corrections_jecstack(self):
        """Use the corrections provided in the JECStack for non-clib scenario."""
        self.corrections = self.jec_stack.corrections

        # Ensure all required inputs have mappings
        total_signature = set()
        for part in _stack_parts:
            attr = getattr(self.jec_stack, part)
            if attr is not None:
                total_signature.update(attr.signature)

        missing = total_signature - set(self.name_map.keys())
        if len(missing) > 0:
            raise Exception(
                f"Missing mapping of {missing} in name_map!"
                + " Cannot evaluate jet corrections!"
                + " Please supply mappings for these variables!"
            )

    def _prepare_jets(self, jets):
        fields = awkward.fields(jets)
        if len(fields) == 0:
            raise Exception(
                "Empty record, please pass a jet object with at least {self.real_sig} defined!"
            )

        try:
            out = awkward.flatten(jets)
        except ValueError:
            flattened_fields = {field: awkward.flatten(jets[field]) for field in fields}
            out = awkward.zip(
                flattened_fields,
                depth_limit=1,
                behavior=getattr(jets, "behavior", None),
                parameters=getattr(getattr(jets, "layout", None), "parameters", None),
            )

        wrap = partial(awkward_rewrap, like_what=jets, gfunc=rewrap_recordarray)
        scalar_form = awkward.without_parameters(
            out[self.name_map["ptRaw"]]
        ).layout.form

        in_dict = {field: out[field] for field in fields}
        in_dict[self.name_map["JetPt"] + "_orig"] = in_dict[self.name_map["JetPt"]]
        in_dict[self.name_map["JetMass"] + "_orig"] = in_dict[self.name_map["JetMass"]]
        out_dict = dict(in_dict)

        if self.treat_pt_as_raw:
            out_dict[self.name_map["ptRaw"]] = out_dict[self.name_map["JetPt"]]
            out_dict[self.name_map["massRaw"]] = out_dict[self.name_map["JetMass"]]

        return out, wrap, scalar_form, in_dict, out_dict

    def _resolve_level_limit(self, target_level):
        if target_level is None:
            return len(getattr(self.jec_stack, "jec_names_clib", []) or [])

        available = {}
        names = list(getattr(self.jec_stack, "jec_names_clib", []) or [])
        short_names = list(getattr(self.jec_stack, "jec_levels", []) or [])
        for idx, name in enumerate(names):
            available[name] = idx + 1
            if idx < len(short_names):
                available[short_names[idx]] = idx + 1

        if target_level not in available:
            raise ValueError(
                f"Unknown target_level '{target_level}'. Available levels are: "
                + ", ".join(available.keys())
            )

        return available[target_level]

    def _evaluate_correctionlib_correction(
        self,
        jets,
        jec_name_map,
        lazy_cache,
        target_level,
        out_dict,
    ):
        level_limit = self._resolve_level_limit(target_level)
        if level_limit == 0:
            return awkward.values_astype(
                awkward.ones_like(out_dict[self.name_map["JetPt"]]), numpy.float32
            )

        cumulative = None
        for idx, lvl in enumerate(self.jec_stack.jec_names_clib):
            sf = self.corrections.get(lvl)
            if sf is None:
                raise ValueError(f"Correction {lvl} not found in self.corrections")

            inputs = get_corr_inputs(
                jets=jets,
                corr_obj=sf,
                name_map=jec_name_map,
                cache=lazy_cache,
                corrections=cumulative,
            )
            correction = sf.evaluate(*inputs).astype(dtype=numpy.float32)
            if cumulative is None:
                cumulative = correction
            else:
                cumulative = correction * cumulative

            if idx + 1 == level_limit:
                break

        if cumulative is None:
            cumulative = awkward.values_astype(
                awkward.ones_like(out_dict[self.name_map["JetPt"]]), numpy.float32
            )

        return cumulative

    def _evaluate_total_correction(
        self,
        jets,
        out_dict,
        jec_name_map,
        lazy_cache,
        scalar_form,
        target_level,
    ):
        if self.tool == "jecstack":
            if target_level is not None:
                raise ValueError(
                    "target_level selection is only supported for correctionlib-based inputs"
                )

            if self.jec_stack.jec is not None:
                jec_args = {
                    k: out_dict[jec_name_map[k]] for k in self.jec_stack.jec.signature
                }
                return self.jec_stack.jec.getCorrection(
                    **jec_args, form=scalar_form, lazy_cache=lazy_cache
                )

            return awkward.ones_like(out_dict[self.name_map["JetPt"]])

        if self.tool == "clib":
            return self._evaluate_correctionlib_correction(
                jets, jec_name_map, lazy_cache, target_level, out_dict
            )

        raise RuntimeError("Unsupported correction tool configuration")

    def correction_factors(self, jets, lazy_cache, target_level=None):
        if lazy_cache is None:
            raise Exception(
                "CorrectedJetsFactory requires an awkward-array cache to function correctly."
            )
        mapping_proxy = getattr(awkward._util, "MappingProxy", None)
        if mapping_proxy is not None:
            lazy_cache = mapping_proxy.maybe_wrap(lazy_cache)
        if not isinstance(jets, awkward.highlevel.Array):
            raise Exception("'jets' must be an awkward > 1.0.0 array of some kind!")

        out, wrap, scalar_form, _, out_dict = self._prepare_jets(jets)

        jec_name_map = dict(self.name_map)
        jec_name_map["JetPt"] = jec_name_map["ptRaw"]
        jec_name_map["JetMass"] = jec_name_map["massRaw"]

        total_correction = self._evaluate_total_correction(
            jets, out_dict, jec_name_map, lazy_cache, scalar_form, target_level
        )

        correction_record = awkward.zip(
            {"correction": awkward.Array(total_correction)},
            depth_limit=1,
            parameters=out.layout.parameters,
            behavior=out.behavior,
        )

        return wrap(correction_record)["correction"]

    def build(self, jets, lazy_cache, target_level=None):
        if lazy_cache is None:
            raise Exception(
                "CorrectedJetsFactory requires an awkward-array cache to function correctly."
            )
        mapping_proxy = getattr(awkward._util, "MappingProxy", None)
        if mapping_proxy is not None:
            lazy_cache = mapping_proxy.maybe_wrap(lazy_cache)
        if not isinstance(jets, awkward.highlevel.Array):
            raise Exception("'jets' must be an awkward > 1.0.0 array of some kind!")

        out, wrap, scalar_form, in_dict, out_dict = self._prepare_jets(jets)

        jec_name_map = dict(self.name_map)
        jec_name_map["JetPt"] = jec_name_map["ptRaw"]
        jec_name_map["JetMass"] = jec_name_map["massRaw"]

        total_correction = self._evaluate_total_correction(
            jets, out_dict, jec_name_map, lazy_cache, scalar_form, target_level
        )
        out_dict["jet_energy_correction"] = total_correction

        # Finally, the lazy binding to the JEC
        init_pt = partial(
            awkward.virtual,
            operator.mul,
            args=(out_dict["jet_energy_correction"], out_dict[self.name_map["ptRaw"]]),
            cache=lazy_cache,
        )
        init_mass = partial(
            awkward.virtual,
            operator.mul,
            args=(
                out_dict["jet_energy_correction"],
                out_dict[self.name_map["massRaw"]],
            ),
            cache=lazy_cache,
        )

        out_dict[self.name_map["JetPt"]] = init_pt(length=len(out), form=scalar_form)
        out_dict[self.name_map["JetMass"]] = init_mass(
            length=len(out), form=scalar_form
        )

        out_dict[self.name_map["JetPt"] + "_jec"] = out_dict[self.name_map["JetPt"]]
        out_dict[self.name_map["JetMass"] + "_jec"] = out_dict[self.name_map["JetMass"]]

        has_jer = False
        if self.tool == "jecstack":
            if self.jec_stack.jer is not None and self.jec_stack.jersf is not None:
                has_jer = True
        elif self.tool == "clib":
            has_jer = len(self.jec_stack.jer_names_clib) > 0

        if has_jer:
            jer_name_map = dict(self.name_map)
            jer_name_map["JetPt"] = jer_name_map["JetPt"] + "_jec"
            jer_name_map["JetMass"] = jer_name_map["JetMass"] + "_jec"

            if self.tool == "jecstack":
                jer_args = {
                    k: out_dict[jer_name_map[k]] for k in self.jec_stack.jer.signature
                }
                out_dict["jet_energy_resolution"] = self.jec_stack.jer.getResolution(
                    **jer_args, form=scalar_form, lazy_cache=lazy_cache
                )

                jersf_args = {
                    k: out_dict[jer_name_map[k]] for k in self.jec_stack.jersf.signature
                }
                out_dict[
                    "jet_energy_resolution_scale_factor"
                ] = self.jec_stack.jersf.getScaleFactor(
                    **jersf_args, form=_JERSF_FORM, lazy_cache=lazy_cache
                )

            elif self.tool == "clib":
                # Prepare for clib-based corrections
                jer_out_parms = out.layout.parameters
                jer_out_parms["corrected"] = True
                jer_out = awkward.zip(
                    out_dict,
                    depth_limit=1,
                    parameters=jer_out_parms,
                    behavior=out.behavior,
                )
                jerjets = wrap(jer_out)

                for jer_entry in self.jec_stack.jer_names_clib:
                    outtag = "jet_energy_resolution"
                    jer_entry = jer_entry.replace("SF", "ScaleFactor")
                    sf = self.corrections[jer_entry]
                    inputs = get_corr_inputs(
                        jets=jerjets, corr_obj=sf, name_map=jer_name_map
                    )
                    if "ScaleFactor" in jer_entry:
                        outtag += "_scale_factor"
                        correction = awkward.Array(
                            [
                                sf.evaluate(*inputs, "nom").astype(dtype=numpy.float32),
                                sf.evaluate(*inputs, "up").astype(dtype=numpy.float32),
                                sf.evaluate(*inputs, "down").astype(
                                    dtype=numpy.float32
                                ),
                            ]
                        )
                        correction = awkward.concatenate(
                            [
                                correction[0][:, numpy.newaxis],
                                correction[1][:, numpy.newaxis],
                                correction[2][:, numpy.newaxis],
                            ],
                            axis=1,
                        )
                    else:
                        correction = awkward.Array(
                            sf.evaluate(*inputs).astype(dtype=numpy.float32),
                        )

                    out_dict[outtag] = correction

                del jerjets

            # Gaussian smearing
            seeds = numpy.array(out_dict[self.name_map["JetPt"] + "_orig"])[
                [0, -1]
            ].view("i4")
            out_dict["jet_resolution_rand_gauss"] = awkward.virtual(
                rand_gauss,
                args=(
                    out_dict[self.name_map["JetPt"] + "_orig"],
                    numpy.random.Generator(numpy.random.PCG64(seeds)),
                ),
                cache=lazy_cache,
                length=len(out),
                form=scalar_form,
            )

            init_jerc = partial(
                awkward.virtual,
                jer_smear,
                args=(
                    0,
                    self.forceStochastic,
                    awkward.values_astype(
                        out_dict[jer_name_map["ptGenJet"]], numpy.float32
                    ),
                    awkward.values_astype(
                        out_dict[jer_name_map["JetPt"]], numpy.float32
                    ),
                    awkward.values_astype(
                        out_dict[jer_name_map["JetEta"]], numpy.float32
                    ),
                    awkward.values_astype(
                        out_dict["jet_energy_resolution"], numpy.float32
                    ),
                    awkward.values_astype(
                        out_dict["jet_resolution_rand_gauss"], numpy.float32
                    ),
                    awkward.values_astype(
                        out_dict["jet_energy_resolution_scale_factor"], numpy.float32
                    ),
                ),
                cache=lazy_cache,
            )
            out_dict["jet_energy_resolution_correction"] = init_jerc(
                length=len(out), form=scalar_form
            )

            init_pt_jer = partial(
                awkward.virtual,
                operator.mul,
                args=(
                    out_dict["jet_energy_resolution_correction"],
                    out_dict[jer_name_map["JetPt"]],
                ),
                cache=lazy_cache,
            )
            init_mass_jer = partial(
                awkward.virtual,
                operator.mul,
                args=(
                    out_dict["jet_energy_resolution_correction"],
                    out_dict[jer_name_map["JetMass"]],
                ),
                cache=lazy_cache,
            )

            out_dict[self.name_map["JetPt"]] = init_pt_jer(
                length=len(out), form=scalar_form
            )
            out_dict[self.name_map["JetMass"]] = init_mass_jer(
                length=len(out), form=scalar_form
            )

            out_dict[self.name_map["JetPt"] + "_jer"] = out_dict[self.name_map["JetPt"]]
            out_dict[self.name_map["JetMass"] + "_jer"] = out_dict[
                self.name_map["JetMass"]
            ]

            # JER systematics
            jerc_up = partial(
                awkward.virtual,
                jer_smear,
                args=(
                    1,
                    self.forceStochastic,
                    awkward.values_astype(
                        out_dict[jer_name_map["ptGenJet"]], numpy.float32
                    ),
                    awkward.values_astype(
                        out_dict[jer_name_map["JetPt"]], numpy.float32
                    ),
                    awkward.values_astype(
                        out_dict[jer_name_map["JetEta"]], numpy.float32
                    ),
                    awkward.values_astype(
                        out_dict["jet_energy_resolution"], numpy.float32
                    ),
                    awkward.values_astype(
                        out_dict["jet_resolution_rand_gauss"], numpy.float32
                    ),
                    awkward.values_astype(
                        out_dict["jet_energy_resolution_scale_factor"], numpy.float32
                    ),
                ),
                cache=lazy_cache,
            )
            up = awkward.flatten(jets)
            # always forward the original (likely corrected) pt/mass
            up[self.name_map["JetPt"] + "_orig"] = up[self.name_map["JetPt"]]
            up[self.name_map["JetMass"] + "_orig"] = up[self.name_map["JetMass"]]
            up["jet_energy_resolution_correction"] = jerc_up(
                length=len(out), form=scalar_form
            )
            init_pt_jer = partial(
                awkward.virtual,
                operator.mul,
                args=(
                    up["jet_energy_resolution_correction"],
                    out_dict[jer_name_map["JetPt"]],
                ),
                cache=lazy_cache,
            )
            init_mass_jer = partial(
                awkward.virtual,
                operator.mul,
                args=(
                    up["jet_energy_resolution_correction"],
                    out_dict[jer_name_map["JetMass"]],
                ),
                cache=lazy_cache,
            )
            up[self.name_map["JetPt"]] = init_pt_jer(length=len(out), form=scalar_form)
            up[self.name_map["JetMass"]] = init_mass_jer(
                length=len(out), form=scalar_form
            )

            jerc_down = partial(
                awkward.virtual,
                jer_smear,
                args=(
                    2,
                    self.forceStochastic,
                    awkward.values_astype(
                        out_dict[jer_name_map["ptGenJet"]], numpy.float32
                    ),
                    awkward.values_astype(
                        out_dict[jer_name_map["JetPt"]], numpy.float32
                    ),
                    awkward.values_astype(
                        out_dict[jer_name_map["JetEta"]], numpy.float32
                    ),
                    awkward.values_astype(
                        out_dict["jet_energy_resolution"], numpy.float32
                    ),
                    awkward.values_astype(
                        out_dict["jet_resolution_rand_gauss"], numpy.float32
                    ),
                    awkward.values_astype(
                        out_dict["jet_energy_resolution_scale_factor"], numpy.float32
                    ),
                ),
                cache=lazy_cache,
            )
            down = awkward.flatten(jets)
            # always forward the original (likely corrected) pt/mass
            down[self.name_map["JetPt"] + "_orig"] = down[self.name_map["JetPt"]]
            down[self.name_map["JetMass"] + "_orig"] = down[self.name_map["JetMass"]]
            down["jet_energy_resolution_correction"] = jerc_down(
                length=len(out), form=scalar_form
            )
            init_pt_jer = partial(
                awkward.virtual,
                operator.mul,
                args=(
                    down["jet_energy_resolution_correction"],
                    out_dict[jer_name_map["JetPt"]],
                ),
                cache=lazy_cache,
            )
            init_mass_jer = partial(
                awkward.virtual,
                operator.mul,
                args=(
                    down["jet_energy_resolution_correction"],
                    out_dict[jer_name_map["JetMass"]],
                ),
                cache=lazy_cache,
            )
            down[self.name_map["JetPt"]] = init_pt_jer(
                length=len(out), form=scalar_form
            )
            down[self.name_map["JetMass"]] = init_mass_jer(
                length=len(out), form=scalar_form
            )
            out_dict["JER"] = awkward.zip(
                {"up": up, "down": down}, depth_limit=1, with_name="JetSystematic"
            )

        # Apply uncertainties (JES)
        has_junc = self.jec_stack.junc is not None
        if self.tool == "clib":
            has_junc = len(self.jec_stack.jec_uncsources_clib) > 0

        if has_junc:
            junc_name_map = dict(self.name_map)
            if has_jer:
                junc_name_map["JetPt"] = junc_name_map["JetPt"] + "_jer"
                junc_name_map["JetMass"] = junc_name_map["JetMass"] + "_jer"
            else:
                junc_name_map["JetPt"] = junc_name_map["JetPt"] + "_jec"
                junc_name_map["JetMass"] = junc_name_map["JetMass"] + "_jec"

            if self.tool == "jecstack":
                junc_args = {
                    k: out_dict[junc_name_map[k]] for k in self.jec_stack.junc.signature
                }
                juncs = self.jec_stack.junc.getUncertainty(**junc_args)

            elif self.tool == "clib":
                junc_out_parms = out.layout.parameters
                junc_out_parms["corrected"] = True
                junc_out = awkward.zip(
                    out_dict,
                    depth_limit=1,
                    parameters=junc_out_parms,
                    behavior=out.behavior,
                )
                juncjets = wrap(junc_out)

                uncnames, uncvalues = [], []
                for junc_name in self.jec_stack.jec_uncsources_clib:
                    sf = self.corrections[junc_name]
                    if sf is None:
                        raise ValueError(
                            f"Correction {junc_name} not found in self.corrections"
                        )

                    inputs = get_corr_inputs(
                        jets=juncjets, corr_obj=sf, name_map=junc_name_map
                    )
                    unc = awkward.values_astype(sf.evaluate(*inputs), numpy.float32)
                    central = awkward.ones_like(out_dict[self.name_map["JetPt"]])
                    unc_up = central + unc
                    unc_down = central - unc
                    uncnames.append(junc_name.split("_")[-2])
                    uncvalues.append([unc_up, unc_down])
                del juncjets

                # Combine the up and down values into pairs
                combined_uncvalues = [
                    awkward.Array([[up, down] for up, down in zip(unc_up, unc_down)])
                    for unc_up, unc_down in uncvalues
                ]

                juncs = zip(uncnames, combined_uncvalues)

            def junc_smeared_val(uncvals, up_down, variable):
                return awkward.materialized(uncvals[:, up_down] * variable)

            def build_variation(unc, jetpt, jetpt_orig, jetmass, jetmass_orig, updown):
                var_dict = dict(in_dict)
                var_dict[jetpt] = awkward.virtual(
                    junc_smeared_val,
                    args=(
                        awkward.to_numpy(awkward.values_astype(unc, numpy.float32)),
                        updown,
                        jetpt_orig,
                    ),
                    length=len(out),
                    form=scalar_form,
                    cache=lazy_cache,
                )
                var_dict[jetmass] = awkward.virtual(
                    junc_smeared_val,
                    args=(
                        awkward.to_numpy(awkward.values_astype(unc, numpy.float32)),
                        updown,
                        jetmass_orig,
                    ),
                    length=len(out),
                    form=scalar_form,
                    cache=lazy_cache,
                )
                return awkward.zip(
                    var_dict,
                    depth_limit=1,
                    parameters=out.layout.parameters,
                    behavior=out.behavior,
                )

            def build_variant(unc, jetpt, jetpt_orig, jetmass, jetmass_orig):
                up = build_variation(unc, jetpt, jetpt_orig, jetmass, jetmass_orig, 0)
                down = build_variation(unc, jetpt, jetpt_orig, jetmass, jetmass_orig, 1)
                return awkward.zip(
                    {"up": up, "down": down}, depth_limit=1, with_name="JetSystematic"
                )

            for name, func in juncs:
                out_dict[f"jet_energy_uncertainty_{name}"] = func
                out_dict[f"JES_{name}"] = build_variant(
                    func,
                    self.name_map["JetPt"],
                    out_dict[junc_name_map["JetPt"]],
                    self.name_map["JetMass"],
                    out_dict[junc_name_map["JetMass"]],
                )

        out_parms = out.layout.parameters
        out_parms["corrected"] = True
        out = awkward.zip(
            out_dict, depth_limit=1, parameters=out_parms, behavior=out.behavior
        )

        return wrap(out)
