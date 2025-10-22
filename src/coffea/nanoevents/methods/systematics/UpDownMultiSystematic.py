from copy import copy

import awkward

from coffea.nanoevents.methods.base import behavior
from coffea.nanoevents.methods.systematics.UpDownSystematic import UpDownSystematic


@awkward.behaviors.mixins.mixin_class(behavior)
class UpDownMultiSystematic(UpDownSystematic):
    def _build_variations(self, name, what, varying_function, *args, **kwargs):
        if what == "weight":
            whatarray = self["__systematics__", "__ones__"]
        elif isinstance(what, str):
            whatarray = self[what]
        elif (isinstance(what, list) or isinstance(what, tuple)) and len(what) == 1:
            whatarray = self[what[0]]
        else:
            whatarray = awkward.zip({w: self[w] for w in what}, depth_limit=1)

        self["__systematics__", f"__{name}__"] = varying_function(
            *(whatarray, *args), **kwargs
        )

    def get_variation(self, name, what, astype, updown):
        fields = awkward.fields(self)
        fields.remove("__systematics__")

        varied = self["__systematics__", f"__{name}__", :, self._udmap[updown]]

        params = copy(self.layout.parameters)
        params["variation"] = f"{name}-{what}-{updown}"

        out = {field: self[field] for field in fields}
        if what == "weight":
            out[f"weight_{name}"] = varied
        elif isinstance(what, str):
            out[what] = varied
        elif (isinstance(what, list) or isinstance(what, tuple)) and len(what) == 1:
            out[what[0]] = varied
        else:
            for w in what:
                out[w] = varied[w]

        return awkward.zip(
            out,
            depth_limit=1,
            parameters=params,
            behavior=self.behavior,
            with_name=astype,
        )


behavior[("__typestr__", "UpDownMultiSystematic")] = "UpDownMultiSystematic"
