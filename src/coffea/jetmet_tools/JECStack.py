from coffea.jetmet_tools.FactorizedJetCorrector import FactorizedJetCorrector, _levelre
from coffea.jetmet_tools.JetCorrectionUncertainty import JetCorrectionUncertainty
from coffea.jetmet_tools.JetResolution import JetResolution
from coffea.jetmet_tools.JetResolutionScaleFactor import JetResolutionScaleFactor

_singletons = ["jer", "jersf"]
_nicenames = ["Jet Resolution Calculator", "Jet Resolution Scale Factor Calculator"]


class JECStack:
    """
    Mostly used as an input to `CorrectedJetsFactory`. Hosts and organizes multiple
    corrections under one object.

    jec, junc, etc. can be explicitly set by passing in the appropriate corrector class
    (eg: FactorizedJetCorrector). If they are not set, correctors will be created, using
    the info in `corrections` as input.

    Parameters
    ---------
        corrections: dict[str,lookup_base]
            A dict-like of function names and functions. The function depends on the type
            of correction (eg: for JEC, should be jme_standard_function). We expect JEC
            names to be formatted as their filenames.
        jec: FactorizedJetCorrector, optional
            If provided, overrides the jec that would be created from `corrections` in
            the stack.
        junc: JetCorrectionUncertainty, optional
            If provided, overrides the junc that would be created from `corrections` in
            the stack.
        jer: JetResolution, optional
            If provided, overrides the jer that would be created from `corrections` in
            the stack.
        jersf: JetResolutionScaleFactor, optional
            If provided, overrides the jersf that would be created from `corrections` in
            the stack.
    """

    def __init__(self, corrections, jec=None, junc=None, jer=None, jersf=None):
        self._jec = None
        self._junc = None
        self._jer = None
        self._jersf = None

        assembled = {"jec": {}, "junc": {}, "jer": {}, "jersf": {}}
        for key in corrections.keys():
            if "Uncertainty" in key:
                assembled["junc"][key] = corrections[key]
            elif "SF" in key:
                assembled["jersf"][key] = corrections[key]
            elif "Resolution" in key and "SF" not in key:
                assembled["jer"][key] = corrections[key]
            elif len(_levelre.findall(key)) > 0:
                assembled["jec"][key] = corrections[key]

        for corrtype, nname in zip(_singletons, _nicenames):
            Noftype = len(assembled[corrtype])
            if Noftype > 1:
                raise Exception(
                    f"JEC Stack has at most one {nname}, {Noftype} are present"
                )

        if jec is None:
            if len(assembled["jec"]) == 0:
                self._jec = None  # allow for no JEC
            else:
                self._jec = FactorizedJetCorrector(
                    **{name: corrections[name] for name in assembled["jec"]}
                )
        else:
            if isinstance(jec, FactorizedJetCorrector):
                self._jec = jec
            else:
                raise Exception(
                    'JECStack needs a FactorizedJetCorrector passed as "jec"'
                    + f" got object of type {type(jec)}"
                )

        if junc is None:
            if len(assembled["junc"]) > 0:
                self._junc = JetCorrectionUncertainty(
                    **{name: corrections[name] for name in assembled["junc"]}
                )
        else:
            if isinstance(junc, JetCorrectionUncertainty):
                self._junc = junc
            else:
                raise Exception(
                    'JECStack needs a JetCorrectionUncertainty passed as "junc"'
                    + f" got object of type {type(junc)}"
                )

        if jer is None:
            if len(assembled["jer"]) > 0:
                self._jer = JetResolution(
                    **{name: corrections[name] for name in assembled["jer"]}
                )
        else:
            if isinstance(jer, JetResolution):
                self._jer = jer
            else:
                raise Exception(
                    '"jer" must be of type "JetResolution"' + f" got {type(jer)}"
                )

        if jersf is None:
            if len(assembled["jersf"]) > 0:
                self._jersf = JetResolutionScaleFactor(
                    **{name: corrections[name] for name in assembled["jersf"]}
                )
        else:
            if isinstance(jer, JetResolutionScaleFactor):
                self._jersf = jersf
            else:
                raise Exception(
                    '"jer" must be of type "JetResolutionScaleFactor"'
                    + f" got {type(jer)}"
                )

        if (self.jer is None) != (self.jersf is None):
            raise Exception("Cannot apply JER-SF without an input JER, and vice-versa!")

    @property
    def blank_name_map(self):
        """
        A dictionary in the form of the `name_map` input parameter for
        `CorrectedJetsFactory`, with all keys mapped to None.
        """
        out = {
            "massRaw",
            "ptRaw",
            "JetMass",
            "JetPt",
            "METpt",
            "METphi",
            "JetPhi",
            "UnClusteredEnergyDeltaX",
            "UnClusteredEnergyDeltaY",
        }
        if self._jec is not None:
            for name in self._jec.signature:
                out.add(name)
        if self._junc is not None:
            for name in self._junc.signature:
                out.add(name)
        if self._jer is not None:
            for name in self._jer.signature:
                out.add(name)
        if self._jersf is not None:
            for name in self._jersf.signature:
                out.add(name)
        return {name: None for name in out}

    @property
    def jec(self):
        """
        The stack's FactorizedJetCorrector object.
        """
        return self._jec

    @property
    def junc(self):
        """
        The stack's JetCorrectionUncertainty object.
        """
        return self._junc

    @property
    def jer(self):
        """
        The stack's JetResolution object.
        """
        return self._jer

    @property
    def jersf(self):
        """
        The stack's JetResolutionScaleFactor object.
        """
        return self._jersf
