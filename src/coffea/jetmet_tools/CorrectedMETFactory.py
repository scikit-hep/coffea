import awkward
import dask_awkward
import numpy


def corrected_polar_met(
    met_pt, met_phi, jet_pt, jet_phi, jet_pt_orig, positive=None, dx=None, dy=None
):
    sj, cj = numpy.sin(jet_phi), numpy.cos(jet_phi)
    x = met_pt * numpy.cos(met_phi) + awkward.sum((jet_pt - jet_pt_orig) * cj, axis=1)
    y = met_pt * numpy.sin(met_phi) + awkward.sum((jet_pt - jet_pt_orig) * sj, axis=1)
    if positive is not None and dx is not None and dy is not None:
        x = x + dx if positive else x - dx
        y = y + dy if positive else y - dy

    return awkward.zip(
        {"pt": numpy.hypot(x, y), "phi": numpy.arctan2(y, x)}, depth_limit=1
    )


class CorrectedMETFactory:
    """
    Factory class for propagating corrections made to jets into a corrected value
    of MET. This includes organizing different variations associated with uncertainties
    in MET from unclustered energy.

    Once the `CorrectedMETFactory` is constructed, an array of corrected MET values and
    variations can be produced with the `build` method, which requires an array of
    uncorrected MET and an array of corrected jets.

    Parameters
    ----------
        name_map: dict[str,str]
            Keys must include at least the following:
                - METpt
                - METphi
                - JetPt
                - JetPhi
                - ptRaw
                - UnClusteredEnergyDeltaX
                - UnClusteredEnergyDeltaY
            and each of those must be mapped to the corresponding field name of the input
            arrays `in_MET` and `in_corrected_jets` for the `build` method.
    """

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

    def build(self, in_MET, in_corrected_jets):
        """
        Produce an array of corrected MET values from an array of uncorrected MET
        values and an array of corrected jets.

        Parameters
        ----------
            in_MET: (Awkward array[float])
                An array of raw (uncorrected) MET values.
            in_corrected_jets: (Awkward array[jets])
                An array of corrected jets, as produced by `CorrectedJetsFactory`.

        Returns
        -------
            Awkward array of corrected MET values, with shape matching `in_MET`.
        """
        if not isinstance(
            in_MET, (awkward.highlevel.Array, dask_awkward.Array)
        ) or not isinstance(
            in_corrected_jets, (awkward.highlevel.Array, dask_awkward.Array)
        ):
            raise Exception(
                "'MET' and 'corrected_jets' must be an (dask_)awkward array of some kind!"
            )

        MET = in_MET
        if isinstance(in_MET, awkward.highlevel.Array):
            MET = dask_awkward.from_awkward(in_MET, 1)

        corrected_jets = in_corrected_jets
        if isinstance(in_corrected_jets, awkward.highlevel.Array):
            corrected_jets = dask_awkward.from_awkward(in_corrected_jets, 1)

        def switch_properties(raw_met, corrected_jets, dx, dy, positive, save_orig):
            variation = corrected_polar_met(
                raw_met[self.name_map["METpt"]],
                raw_met[self.name_map["METphi"]],
                corrected_jets[self.name_map["JetPt"]],
                corrected_jets[self.name_map["JetPhi"]],
                corrected_jets[self.name_map["ptRaw"]],
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

        out = dask_awkward.map_partitions(
            switch_properties,
            MET,
            corrected_jets,
            None,
            None,
            None,
            True,
            label="nominal_corrected_met",
        )

        out_dict = {field: out[field] for field in dask_awkward.fields(out)}

        out_dict["MET_UnclusteredEnergy"] = dask_awkward.map_partitions(
            create_variants,
            MET,
            corrected_jets,
            MET[self.name_map["UnClusteredEnergyDeltaX"]],
            MET[self.name_map["UnClusteredEnergyDeltaY"]],
            label="UnclusteredEnergy_met",
        )

        for unc in filter(
            lambda x: x.startswith(("JER", "JES")), dask_awkward.fields(corrected_jets)
        ):
            out_dict[unc] = dask_awkward.map_partitions(
                create_variants,
                MET,
                corrected_jets[unc],
                None,
                None,
                label=f"{unc}_met",
            )

        out_parms = out._meta.layout.parameters
        out = dask_awkward.zip(
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
