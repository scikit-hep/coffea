import awkward
import dask_awkward as dak
import hist
import numpy

from coffea import processor
from coffea.analysis_tools import (
    PackedSelection,
    Weights,
)


def convert_GenModel_to_int(GenModel, selected_models, dtype=numpy.int16):
    """Converts a GenModel RecordArray to an integer array, where each model is assigned a unique integer

    This function critically relies upon a consistent order for the GenModel's fields array (selected_models are indexed into it)
    Not suitable for processing concrete arrays with files with different GenModel subsets, use save_form when preprocessing
    inputs for coffea NanoEventsFactory
    """
    models = sorted(
        GenModel.fields,
        key=lambda name: 1000 * int(name.split("_")[-2]) + int(name.split("_")[-1]),
    )
    labels = {mp: models.index(mp) for mp in selected_models}
    labels = {"uncategorized": -1, **labels}
    if "TChiZH" in models[0]:
        assert len(models) == 296
    arr = (-1) * awkward.ones_like(GenModel[selected_models[0]], dtype=dtype)
    for field in selected_models:
        # offset by 1 to counter the default -1 value
        i = models.index(field) + 1
        arr = arr + (GenModel[field] * dtype(i))
    return arr, labels


class NanoEventsGenModelProcessor(processor.ProcessorABC):
    def __init__(self):
        pass

    def process(self, events):
        # Configure everything for the GenModel categorical axis
        # selected_models=["TChiZH_700_1", "TChiZH_1100_200", "TChiZH_950_400"]
        selected_models = ["TChiZH_1100_200", "TChiZH_950_400"]
        for model in selected_models:
            events["GenModel", model] = awkward.fill_none(
                getattr(events.GenModel, model), False, axis=None
            )
        model_categories, labels = convert_GenModel_to_int(
            events.GenModel,
            selected_models=selected_models,
        )
        categorical_args = {
            "axis": hist.axis.IntCategory(
                list(labels.values()),
                name="model_point",
                label=str(labels) + "::Gen Model",
                flow=False,
            ),
            "values": model_categories,
            "labels": list(labels.keys()),
        }

        # Prepare the PackedSelection
        selection = PackedSelection()
        btag = awkward.sum(events.Jet.btagDeepFlavB >= 0.4, axis=1) >= 1
        threejets = awkward.sum(events.Jet.pt >= 20, axis=1) >= 2
        selection.add_multiple(
            {
                "included_models": (model_categories > -1),
                "btag": btag,
                "threejets": threejets,
            }
        )
        cuts = ["included_models", "btag", "threejets"]

        # Prepare the Weights
        weight = Weights(None)
        weight.add(
            "test",
            dak.ones_like(events.genWeight),
            weightUp=1.25 * awkward.ones_like(events.genWeight),
            weightDown=0.5 * awkward.ones_like(events.genWeight),
        )

        # Instantiate the Cutflow and generate the histograms
        cutflow = selection.cutflow(
            *cuts,
            commonmask=None,
            weights=weight,
            weightsmodifier=None,
        )

        honecut, hcutflow, hlabels, catlabel = cutflow.yieldhist(
            weighted=True,
            categorical=categorical_args,
        )
        array_dict = {"jpt": events.Jet.pt, "jbtag": events.Jet.btagDeepFlavB}
        axes_dict = {
            "jpt": hist.axis.Regular(20, 0, 20, name="jpt", flow=True),
            "jbtag": hist.axis.Regular(20, 0, 1, name="jbtag", flow=True),
        }
        varsonecuts, varscutflows, varslabels, varscatlabels = cutflow.plot_vars(
            array_dict,
            axes=axes_dict.values(),
            weighted=True,
            categorical=categorical_args,
        )

        return {
            "honecut": honecut,
            "hcutflow": hcutflow,
            "hlabels": hlabels,
            "catlabel": catlabel,
            "varsonecuts": varsonecuts,
            "varscutflows": varscutflows,
            "varslabels": varslabels,
            "varscatlabels": varscatlabels,
        }

    def postprocess(self, accumulator):
        return accumulator
