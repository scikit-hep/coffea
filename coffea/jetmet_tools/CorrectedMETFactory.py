import awkward
import numpy
from copy import copy


def corrected_polar_met(met_pt, met_phi, jet_pt, jet_phi, jet_pt_orig, deltas=None):
    sj, cj = numpy.sin(jet_phi), numpy.cos(jet_phi)
    x = met_pt * numpy.cos(met_phi) + awkward.sum(
        jet_pt * cj - jet_pt_orig * cj, axis=1
    )
    y = met_pt * numpy.sin(met_phi) + awkward.sum(
        jet_pt * sj - jet_pt_orig * sj, axis=1
    )
    if deltas:
        positive, dx, dy = deltas
        x = x + dx if positive else x - dx
        y = y + dy if positive else y - dy
    return awkward.zip({"pt": numpy.hypot(x, y), "phi": numpy.arctan2(y, x)})


class CorrectedMETFactory(object):
    def __init__(self, name_map):
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

        self.name_map = name_map

    def build(self, MET, corrected_jets, lazy_cache):
        if lazy_cache is None:
            raise Exception(
                "CorrectedMETFactory requires a awkward-array cache to function correctly."
            )
        lazy_cache = awkward._util.MappingProxy.maybe_wrap(lazy_cache)
        if not isinstance(MET, awkward.highlevel.Array) or not isinstance(
            corrected_jets, awkward.highlevel.Array
        ):
            raise Exception(
                "'MET' and 'corrected_jets' must be an awkward array of some kind!"
            )

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

        def lazy_variant(unc, metpt, metphi, jetpt, jetphi, jetptraw):
            return awkward.zip(
                {
                    "up": make_variant(
                        MET[metpt],
                        MET[metphi],
                        corrected_jets[unc].up[jetpt],
                        corrected_jets[unc].up[jetphi],
                        corrected_jets[unc].up[jetptraw],
                    ),
                    "down": make_variant(
                        MET[metpt],
                        MET[metphi],
                        corrected_jets[unc].down[jetpt],
                        corrected_jets[unc].down[jetphi],
                        corrected_jets[unc].down[jetptraw],
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
            corrected_jets[self.name_map["ptRaw"]],
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
                    corrected_jets[self.name_map["ptRaw"]],
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
                    corrected_jets[self.name_map["ptRaw"]],
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
                    self.name_map["ptRaw"],
                ),
                length=length,
                cache={},
            )

        out_parms = out.layout.parameters
        out = awkward.zip(
            out_dict, depth_limit=1, parameters=out_parms, behavior=out.behavior
        )

        return out

    def uncertainties(self):
        return ["MET_UnclusteredEnergy"]
