import warnings
from functools import partial

import awkward
import dask_awkward
import numpy

_stack_parts = ["jec", "junc", "jer", "jersf"]
_MIN_JET_ENERGY = numpy.float32(1e-2)
_ONE_F32 = numpy.float32(1.0)
_ZERO_F32 = numpy.float32(0.0)
_JERSF_FORM = {
    "class": "NumpyArray",
    "inner_shape": [3],
    "itemsize": 4,
    "format": "f",
    "primitive": "float32",
}


# we're gonna assume that the first record array we encounter is the flattened data
def rewrap_recordarray(layout, depth, data, **kwargs):
    if isinstance(layout, awkward.contents.RecordArray):
        return data
    return None


def awkward_rewrap(arr, like_what, gfunc):
    func = partial(gfunc, data=arr.layout)
    return awkward.transform(func, like_what, behavior=like_what.behavior)


class _AwkwardRewrapFn:
    def __init__(self, gfunc):
        self.gfunc = gfunc

    def __call__(self, array, like_what):
        func = partial(self.gfunc, data=array.layout)
        return awkward.transform(func, like_what, behavior=like_what.behavior)


def rand_gauss(item):
    seeds = (
        awkward.typetracer.length_one_if_typetracer(item).to_numpy()[[0, -1]].view("i4")
    )
    randomstate = numpy.random.Generator(numpy.random.PCG64(seeds))

    def getfunction(layout, depth, **kwargs):
        if isinstance(layout, awkward.contents.NumpyArray) or not isinstance(
            layout, (awkward.contents.Content,)
        ):
            return awkward.contents.NumpyArray(
                randomstate.normal(size=len(layout)).astype(numpy.float32)
            )
        return None

    out = awkward.transform(
        getfunction,
        awkward.typetracer.length_zero_if_typetracer(item),
        behavior=item.behavior,
    )
    if awkward.backend(item) == "typetracer":
        out = awkward.Array(
            out.layout.to_typetracer(forget_length=True), behavior=out.behavior
        )

    assert out is not None
    return out


def jer_smear(
    pt_gen,
    jetPt,
    etaJet,
    jet_energy_resolution,
    jet_resolution_rand_gauss,
    jet_energy_resolution_scale_factor,
    variation,
    forceStochastic,
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

    backend = awkward.backend(smearfact, jetPt)

    smearfact = awkward.typetracer.length_zero_if_typetracer(smearfact)
    jetPt = awkward.typetracer.length_zero_if_typetracer(jetPt)

    def getfunction(layout, depth, **kwargs):
        if isinstance(layout, awkward.contents.NumpyArray) or not isinstance(
            layout, awkward.contents.Content
        ):
            return awkward.contents.NumpyArray(smearfact)
        return None

    smearfact = awkward.transform(getfunction, jetPt, behavior=jetPt.behavior)

    if backend == "typetracer":
        jetPt = awkward.Array(
            jetPt.layout.to_typetracer(forget_length=True), behavior=jetPt.behavior
        )
        smearfact = awkward.Array(
            smearfact.layout.to_typetracer(forget_length=True),
            behavior=smearfact.behavior,
        )

    return smearfact


class CorrectedJetsFactory:
    """
    Factory class for applying corrections to jets, including organizing variations
    (eg: JES up and down). It is constructed from a name map, which translates between
    field names for corrections and field names in inputs, and a JECStack, which
    contains the actual correction functions.

    Once a CorrectedJetsFactory is constructed, the `build` method can produce corrected
    jets from an input array of jets.

    Parameters
    ----------
        name_map: dict[str,str]
            Keys are argument names in the various corrections' signatures (eg: the `signature`
            attribute of a `FactorizedJJetCorrector` object). Values are the names of the
            corresponding fields as they would appear in the jet array passed to the `build`
            method.
        jec_stack: JECStack
            Contains the corrections that will be applied to the input jet array when calling
            `build`.
    """

    def __init__(self, name_map, jec_stack):
        # from PhysicsTools/PatUtils/interface/SmearedJetProducerT.h#L283
        self.forceStochastic = False

        if "ptRaw" not in name_map or name_map["ptRaw"] is None:
            warnings.warn(
                "There is no name mapping for ptRaw,"
                " CorrectedJets will assume that <object>.pt is raw pt!"
            )
            name_map["ptRaw"] = name_map["JetPt"] + "_raw"
        self.treat_pt_as_raw = "ptRaw" not in name_map

        if "massRaw" not in name_map or name_map["massRaw"] is None:
            warnings.warn(
                "There is no name mapping for massRaw,"
                " CorrectedJets will assume that <object>.mass is raw pt!"
            )
            name_map["ptRaw"] = name_map["JetMass"] + "_raw"

        total_signature = set()
        for part in _stack_parts:
            attr = getattr(jec_stack, part)
            if attr is not None:
                total_signature.update(attr.signature)

        missing = total_signature - set(name_map.keys())
        if len(missing) > 0:
            raise Exception(
                f"Missing mapping of {missing} in name_map!"
                + " Cannot evaluate jet corrections!"
                + " Please supply mappings for these variables!"
            )

        if "ptGenJet" not in name_map:
            warnings.warn(
                'Input JaggedCandidateArray must have "ptGenJet" in order to apply hybrid JER smearing method. Stochastic smearing will be applied.'
            )
            self.forceStochastic = True

        self.real_sig = [v for k, v in name_map.items()]
        self.name_map = name_map
        self.jec_stack = jec_stack

    def uncertainties(self):
        """
        Returns a list of the sources of uncertainty included in the stack.

        Returns
        -------
            list[str]
                A list of the sources of uncertainty.
        """
        out = ["JER"] if self.jec_stack.jer is not None else []
        if self.jec_stack.junc is not None:
            out.extend([f"JES_{unc}" for unc in self.jec_stack.junc.levels])
        return out

    def build(self, injets):
        """
        Apply the corrections to the array of jets, returning an array of corrected
        jets.

        Parameters
        ----------
            injets: (Awkward array[jets])
                An array of uncorrected jets, to which we want to apply corrections.

        Returns
        -------
            Awkward array of jets, representing the corrected jets, with shape matching
            `injets`.
        """
        if not isinstance(injets, (awkward.highlevel.Array, dask_awkward.Array)):
            raise Exception("input jets must be an (dask_)awkward array of some kind!")

        jets = (
            injets
            if isinstance(injets, dask_awkward.Array)
            else dask_awkward.from_awkward(injets, 1)
        )

        fields = dask_awkward.fields(jets)
        if len(fields) == 0:
            raise Exception(
                "Empty record, please pass a jet object with at least {self.real_sig} defined!"
            )
        out = dask_awkward.flatten(jets)
        wrap = partial(awkward_rewrap, like_what=jets._meta, gfunc=rewrap_recordarray)

        in_dict = {field: out[field] for field in fields}
        out_dict = dict(in_dict)

        # take care of nominal JEC (no JER if available)
        out_dict[self.name_map["JetPt"] + "_orig"] = out_dict[self.name_map["JetPt"]]
        out_dict[self.name_map["JetMass"] + "_orig"] = out_dict[
            self.name_map["JetMass"]
        ]
        if self.treat_pt_as_raw:
            out_dict[self.name_map["ptRaw"]] = out_dict[self.name_map["JetPt"]]
            out_dict[self.name_map["massRaw"]] = out_dict[self.name_map["JetMass"]]

        jec_name_map = dict(self.name_map)
        jec_name_map["JetPt"] = jec_name_map["ptRaw"]
        jec_name_map["JetMass"] = jec_name_map["massRaw"]
        if self.jec_stack.jec is not None:
            jec_args = {
                k: out_dict[jec_name_map[k]] for k in self.jec_stack.jec.signature
            }
            out_dict["jet_energy_correction"] = self.jec_stack.jec.getCorrection(
                **jec_args
            )
        else:
            out_dict["jet_energy_correction"] = dask_awkward.without_parameters(
                dask_awkward.ones_like(out_dict[self.name_map["JetPt"]])
            )

        # finally the lazy binding to the JEC
        init_pt = out_dict["jet_energy_correction"] * out_dict[self.name_map["ptRaw"]]
        init_mass = (
            out_dict["jet_energy_correction"] * out_dict[self.name_map["massRaw"]]
        )

        out_dict[self.name_map["JetPt"]] = init_pt
        out_dict[self.name_map["JetMass"]] = init_mass

        out_dict[self.name_map["JetPt"] + "_jec"] = out_dict[self.name_map["JetPt"]]
        out_dict[self.name_map["JetMass"] + "_jec"] = out_dict[self.name_map["JetMass"]]

        # in jer we need to have a stash for the intermediate JEC products
        has_jer = False
        if self.jec_stack.jer is not None and self.jec_stack.jersf is not None:
            has_jer = True
            jer_name_map = dict(self.name_map)
            jer_name_map["JetPt"] = jer_name_map["JetPt"] + "_jec"
            jer_name_map["JetMass"] = jer_name_map["JetMass"] + "_jec"

            jerargs = {
                k: out_dict[jer_name_map[k]] for k in self.jec_stack.jer.signature
            }
            out_dict["jet_energy_resolution"] = self.jec_stack.jer.getResolution(
                **jerargs
            )

            jersfargs = {
                k: out_dict[jer_name_map[k]] for k in self.jec_stack.jersf.signature
            }
            out_dict["jet_energy_resolution_scale_factor"] = (
                self.jec_stack.jersf.getScaleFactor(**jersfargs)
            )

            out_dict["jet_resolution_rand_gauss"] = dask_awkward.map_partitions(
                rand_gauss,
                out_dict[self.name_map["JetPt"] + "_orig"],
            )

            init_jerc = dask_awkward.map_partitions(
                jer_smear,
                out_dict[jer_name_map["ptGenJet"]],
                out_dict[jer_name_map["JetPt"]],
                out_dict[jer_name_map["JetEta"]],
                out_dict["jet_energy_resolution"],
                out_dict["jet_resolution_rand_gauss"],
                out_dict["jet_energy_resolution_scale_factor"],
                0,
                self.forceStochastic,
            )
            out_dict["jet_energy_resolution_correction"] = init_jerc

            init_pt_jer = (
                out_dict["jet_energy_resolution_correction"]
                * out_dict[jer_name_map["JetPt"]]
            )
            init_mass_jer = (
                out_dict["jet_energy_resolution_correction"]
                * out_dict[jer_name_map["JetMass"]]
            )

            out_dict[self.name_map["JetPt"]] = init_pt_jer
            out_dict[self.name_map["JetMass"]] = init_mass_jer

            out_dict[self.name_map["JetPt"] + "_jer"] = out_dict[self.name_map["JetPt"]]
            out_dict[self.name_map["JetMass"] + "_jer"] = out_dict[
                self.name_map["JetMass"]
            ]

            # JER systematics
            jerc_up = dask_awkward.map_partitions(
                jer_smear,
                out_dict[jer_name_map["ptGenJet"]],
                out_dict[jer_name_map["JetPt"]],
                out_dict[jer_name_map["JetEta"]],
                out_dict["jet_energy_resolution"],
                out_dict["jet_resolution_rand_gauss"],
                out_dict["jet_energy_resolution_scale_factor"],
                1,
                self.forceStochastic,
            )
            up = dask_awkward.flatten(jets)
            up = dask_awkward.with_field(
                up, jerc_up, where="jet_energy_resolution_correction"
            )

            init_pt_jer = (
                up["jet_energy_resolution_correction"] * out_dict[jer_name_map["JetPt"]]
            )
            init_mass_jer = (
                up["jet_energy_resolution_correction"]
                * out_dict[jer_name_map["JetMass"]]
            )

            up = dask_awkward.with_field(up, init_pt_jer, where=self.name_map["JetPt"])
            up = dask_awkward.with_field(
                up, init_mass_jer, where=self.name_map["JetMass"]
            )

            jerc_down = dask_awkward.map_partitions(
                jer_smear,
                out_dict[jer_name_map["ptGenJet"]],
                out_dict[jer_name_map["JetPt"]],
                out_dict[jer_name_map["JetEta"]],
                out_dict["jet_energy_resolution"],
                out_dict["jet_resolution_rand_gauss"],
                out_dict["jet_energy_resolution_scale_factor"],
                2,
                self.forceStochastic,
            )
            down = dask_awkward.flatten(jets)
            down = dask_awkward.with_field(
                down, jerc_down, where="jet_energy_resolution_correction"
            )

            init_pt_jer = (
                down["jet_energy_resolution_correction"]
                * out_dict[jer_name_map["JetPt"]]
            )
            init_mass_jer = (
                down["jet_energy_resolution_correction"]
                * out_dict[jer_name_map["JetMass"]]
            )

            down = dask_awkward.with_field(
                down, init_pt_jer, where=self.name_map["JetPt"]
            )
            down = dask_awkward.with_field(
                down, init_mass_jer, where=self.name_map["JetMass"]
            )

            out_dict["JER"] = dask_awkward.zip(
                {"up": up, "down": down}, depth_limit=1, with_name="JetSystematic"
            )

        if self.jec_stack.junc is not None:
            juncnames = {}
            juncnames.update(self.name_map)
            if has_jer:
                juncnames["JetPt"] = juncnames["JetPt"] + "_jer"
                juncnames["JetMass"] = juncnames["JetMass"] + "_jer"
            else:
                juncnames["JetPt"] = juncnames["JetPt"] + "_jec"
                juncnames["JetMass"] = juncnames["JetMass"] + "_jec"
            juncargs = {
                k: out_dict[juncnames[k]] for k in self.jec_stack.junc.signature
            }
            juncs = self.jec_stack.junc.getUncertainty(**juncargs)

            def junc_smeared_val(uncvals, up_down, variable):
                return uncvals[:, up_down] * variable

            def build_variation(
                unc, template, jetpt, jetpt_orig, jetmass, jetmass_orig, updown
            ):
                var_dict = {
                    field: template[field] for field in awkward.fields(template)
                }
                var_dict[jetpt] = junc_smeared_val(
                    unc,
                    updown,
                    jetpt_orig,
                )
                var_dict[jetmass] = junc_smeared_val(
                    unc,
                    updown,
                    jetmass_orig,
                )
                return awkward.zip(
                    var_dict,
                    depth_limit=1,
                    parameters=template.layout.parameters,
                    behavior=template.behavior,
                )

            def build_variant(unc, template, jetpt, jetpt_orig, jetmass, jetmass_orig):
                up = build_variation(
                    unc, template, jetpt, jetpt_orig, jetmass, jetmass_orig, 0
                )
                down = build_variation(
                    unc, template, jetpt, jetpt_orig, jetmass, jetmass_orig, 1
                )
                return awkward.zip(
                    {"up": up, "down": down}, depth_limit=1, with_name="JetSystematic"
                )

            for name, func in juncs:
                out_dict[f"jet_energy_uncertainty_{name}"] = func
                out_dict[f"JES_{name}"] = dask_awkward.map_partitions(
                    build_variant,
                    func,
                    out,
                    self.name_map["JetPt"],
                    out_dict[juncnames["JetPt"]],
                    self.name_map["JetMass"],
                    out_dict[juncnames["JetMass"]],
                    label=f"{name}",
                )

        out_parms = out._meta.layout.parameters
        out_parms["corrected"] = True
        out = dask_awkward.zip(
            out_dict, depth_limit=1, parameters=out_parms, behavior=out.behavior
        )

        out_meta = wrap(out._meta)

        return dask_awkward.map_partitions(
            _AwkwardRewrapFn(gfunc=rewrap_recordarray),
            out,
            jets,
            label="corrected_jets",
            meta=out_meta,
        )
