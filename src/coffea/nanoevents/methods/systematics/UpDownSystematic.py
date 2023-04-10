from copy import copy

import awkward

from coffea.nanoevents.methods.base import Systematic, behavior


@awkward.behaviors.mixins.mixin_class(behavior)
class UpDownSystematic(Systematic):
    """An example instance of a simple systematic with only up/down variations."""

    _udmap = {"up": 0, "down": 1}

    def _build_variations(self, name, what, varying_function, *args, **kwargs):
        whatarray = (
            self[what] if what != "weight" else self["__systematics__", "__ones__"]
        )

        self["__systematics__", f"__{name}__"] = varying_function(
            *(whatarray, *args), **kwargs

    def describe_variations(self):
        """Show the map of variation names to indices."""
        return list(self._udmap.keys())

    def get_variation(self, name, what, astype, updown):
        """Calculate and up or down variation."""
        fields = awkward.fields(self)
        fields.remove("__systematics__")

        varied = self["__systematics__", f"__{name}__", :, self._udmap[updown]]

        params = copy(self.layout.parameters)
        params["variation"] = f"{name}-{what}-{updown}"

        out = {field: self[field] for field in fields}
        if what == "weight":
            out[f"weight_{name}"] = varied
        else:
            out[what] = varied

        return awkward.zip(
            out,
            depth_limit=1,
            parameters=params,
            behavior=self.behavior,
            with_name=astype,
        )

    def up(self, name, what, astype):
        """Return the "up" variation of this observable."""
        return self.get_variation(name, what, astype, "up")

    def down(self, name, what, astype):
        """Return the "down" variation of this observable."""
        return self.get_variation(name, what, astype, "down")


behavior[("__typestr__", "UpDownSystematic")] = "UpDownSystematic"
