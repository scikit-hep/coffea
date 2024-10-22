import re

import uproot

from ..util import deprecate
from ..util import numpy as np

cycle = re.compile(r";\d+")


def killcycle(s, cycle):
    return cycle.sub("", s)


histTypes = ["TH1D", "TH1F", "TH2D", "TH2F", "TH3D", "TH3F"]
graphTypes = ["TGraphAsymmErrors", "TGraph2D"]


def convert_histo_root_file(file):
    converted_file = {}
    fin = uproot.open({file.strip(): None})
    for path, item in fin.iteritems(recursive=True):
        if isinstance(item, uproot.ReadOnlyDirectory):
            continue
        nicepath = killcycle(path, cycle)
        rootclass = item.classname
        if rootclass in histTypes:
            edges = tuple(ax.edges() for ax in item.axes)
            converted_file[(nicepath, "dense_lookup")] = item.values(), edges
            if hasattr(item, "variances"):
                converted_file[(nicepath + "_error", "dense_lookup")] = (
                    np.sqrt(item.variances()),
                    edges,
                )
        elif rootclass in graphTypes:
            deprecate(
                RuntimeError(
                    "The support for TGraph-types will be removed in a future coffea release. In case you need support for TGraph-type, please submit an issue to `https://github.com/CoffeaTeam/coffea/issues`."  # noqa: E501
                ),
                "<unknown>",
            )
            continue
        elif hasattr(item, "_fEX") and hasattr(item, "_fEY"):
            # TODO what is this?
            tempArrX = item._fEX
            tempArrY = item._fEY
            converted_file[(nicepath, "dense_lookup")] = [tempArrX, tempArrY]

    return converted_file
