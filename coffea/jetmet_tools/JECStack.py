from typing import Optional, TypedDict
from typing_extensions import NotRequired
from coffea.jetmet_tools.FactorizedJetCorrector import FactorizedJetCorrector, _levelre
from coffea.jetmet_tools.JetResolution import JetResolution
from coffea.jetmet_tools.JetResolutionScaleFactor import JetResolutionScaleFactor
from coffea.jetmet_tools.JetCorrectionUncertainty import JetCorrectionUncertainty

_singletons = ["jer", "jersf"]
_nicenames = ["Jet Resolution Calculator", "Jet Resolution Scale Factor Calculator"]


class JECNameMap(TypedDict):
    JetPt: Optional[str]
    """Jet transverse momentum (typically already corrected)"""
    ptRaw: Optional[str]
    """Jet raw (uncorrected) transverse momentum"""
    JetMass: Optional[str]
    """Jet mass"""
    massRaw: Optional[str]
    """Jet raw (uncorrected) mass"""
    JetPhi: Optional[str]
    """Jet polar angle (required for MET corrections)"""
    METpt: Optional[str]
    """Missing transverse energy (corrected or not, but consistent with JetPt correction)"""
    METphi: Optional[str]
    """Angle of missing transverse energy"""
    UnClusteredEnergyDeltaX: Optional[str]
    """Sum of unclustered energy systematic shift (x component)"""
    UnClusteredEnergyDeltaY: Optional[str]
    """Sum of unclustered energy systematic shift (y component)"""
    JetEta: NotRequired[Optional[str]]
    """Jet pseudorapidity (optional in principle but required for most JEC)"""
    JetA: NotRequired[Optional[str]]
    """Jet area (optional in principle but required for most JEC)"""
    Rho: NotRequired[Optional[str]]
    """Event average energy deposition (optional in principle but required for most JEC)"""
    ptGenJet: NotRequired[Optional[str]]
    """Transverse momentum of the gen-level jet if this jet is matched to a gen jet, 0 if unmatched.
    (optional in principle but required for most JEC)
    """


def blank_name_map() -> JECNameMap:
    """
    Create a new JECNameMap with all values set to None.
    This is useful for creating a blank map to fill in later.
    """
    return JECNameMap(
        JetPt=None,
        ptRaw=None,
        JetMass=None,
        massRaw=None,
        JetPhi=None,
        METpt=None,
        METphi=None,
        UnClusteredEnergyDeltaX=None,
        UnClusteredEnergyDeltaY=None,
    )


class JECStack(object):
    def __init__(self, corrections, jec=None, junc=None, jer=None, jersf=None):
        """
        corrections is a dict-like of function names and functions
        we expect JEC names to be formatted as their filenames
        jecs, etc. can be overridden by passing in the appropriate corrector class.
        """
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
                    + " got object of type {}".format(type(jec))
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
                    + " got object of type {}".format(type(junc))
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
                    '"jer" must be of type "JetResolution"'
                    + " got {}".format(type(jer))
                )

        if jersf is None:
            if len(assembled["jersf"]) > 0:
                self._jersf = JetResolutionScaleFactor(
                    **{name: corrections[name] for name in assembled["jersf"]}
                )
        else:
            if isinstance(jersf, JetResolutionScaleFactor):
                self._jersf = jersf
            else:
                raise Exception(
                    '"jer" must be of type "JetResolutionScaleFactor"'
                    + " got {}".format(type(jer))
                )

        if (self.jer is None) != (self.jersf is None):
            raise Exception("Cannot apply JER-SF without an input JER, and vice-versa!")

    @property
    def blank_name_map(self) -> JECNameMap:
        out = blank_name_map()
        if self._jec is not None:
            for name in self._jec.signature:
                out[name] = None
        if self._junc is not None:
            for name in self._junc.signature:
                out[name] = None
        if self._jer is not None:
            for name in self._jer.signature:
                out[name] = None
        if self._jersf is not None:
            for name in self._jersf.signature:
                out[name] = None
        return out

    @property
    def jec(self):
        return self._jec

    @property
    def junc(self):
        return self._junc

    @property
    def jer(self):
        return self._jer

    @property
    def jersf(self):
        return self._jersf
