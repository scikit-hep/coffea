from coffea.lookup_tools.lookup_base import lookup_base


class correctionlib_wrapper(lookup_base):
    def __init__(self, payload):
        super(correctionlib_wrapper, self).__init__()
        self._corr = payload
        self._is_jec_stack = self._detect_jec_stack(payload)

    @staticmethod
    def _detect_jec_stack(payload):
        """Detect if the payload originates from a correctionlib JEC stack."""

        meta = getattr(payload, "metadata", {}) or {}
        name = getattr(payload, "name", "") or ""

        meta_markers = {
            str(meta.get(k, "")).lower() for k in ("type", "origin", "source")
        }

        return (
            "jec" in meta_markers
            or meta.get("jec_stack", False)
            or "jecstack" in name.lower()
        )

    def _evaluate(self, *args, **kwargs):
        if self._is_jec_stack:
            return self._evaluate_jec_stack(*args, **kwargs)
        return self._corr.evaluate(*args, **kwargs)

    def _evaluate_jec_stack(self, *args, **kwargs):
        """Use a specialized evaluation path for JEC stack payloads.

        The stack expects keyword-style inputs keyed by the correction input names,
        with an optional "systematic" defaulting to "nom".
        """

        inputs = dict(kwargs)

        if len(args) == 1 and isinstance(args[0], dict):
            inputs.update(args[0])
        elif args:
            ordered_inputs = getattr(self._corr, "inputs", [])
            inputs.update({inp.name: arg for inp, arg in zip(ordered_inputs, args)})

        if "systematic" not in inputs and any(
            getattr(inp, "name", "") == "systematic"
            for inp in getattr(self._corr, "inputs", [])
        ):
            inputs["systematic"] = "nom"

        return self._corr.evaluate(**inputs)

    def __repr__(self):
        signature = ",".join(
            inp.name if len(inp.name) > 0 else f"input{i}"
            for i, inp in enumerate(self._corr._base.inputs)
        )
        return f"correctionlib Correction: {self._corr.name}({signature})"
