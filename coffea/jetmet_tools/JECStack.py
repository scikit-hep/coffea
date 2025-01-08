from dataclasses import dataclass, field
from typing import List, Dict, Optional
from coffea.jetmet_tools.FactorizedJetCorrector import FactorizedJetCorrector, _levelre
from coffea.jetmet_tools.JetResolution import JetResolution
from coffea.jetmet_tools.JetResolutionScaleFactor import JetResolutionScaleFactor
from coffea.jetmet_tools.JetCorrectionUncertainty import JetCorrectionUncertainty
import correctionlib as clib

@dataclass
class JECStack:
    """Handles both JEC and clib cases with conditional attributes."""
    # Common fields for both scenarios
    corrections: Dict[str, any] = field(default_factory=dict)
    use_clib: bool = False  # Set to True if useclib is needed

    # Fields for the clib scenario (useclib=True)
    jec_tag: Optional[str] = None
    jec_levels: Optional[List[str]] = field(default_factory=list)
    jer_tag: Optional[str] = None
    jet_algo: Optional[str] = None
    junc_types: Optional[List[str]] = field(default_factory=list)
    json_path: Optional[str] = None
    savecorr: bool = False

    # Fields for the usejecstack scenario (useclib=False)
    jec: Optional[FactorizedJetCorrector] = None
    junc: Optional[JetCorrectionUncertainty] = None
    jer: Optional[JetResolution] = None
    jersf: Optional[JetResolutionScaleFactor] = None


    def __post_init__(self):
        """Handle initialization based on use_clib flag."""
        if self.use_clib:
            self._initialize_clib()
        else:
            self._initialize_jecstack()

    def _initialize_clib(self):
        """Initialize the clib-based correction tools."""
        if not self.json_path:
            raise ValueError("json_path is required for clib initialization.")

        # Load corrections directly from the JSON path
        self.cset = clib.CorrectionSet.from_file(self.json_path)

        # Construct lists for jec, jer, and uncertainties
        self.jec_names_clib = [f"{self.jec_tag}_{level}_{self.jet_algo}" for level in self.jec_levels]
        self.jer_names_clib = []
        self.jec_uncsources_clib = []

        if self.jer_tag is not None:
            self.jer_names_clib = [
                f"{self.jer_tag}_ScaleFactor_{self.jet_algo}",
                f"{self.jer_tag}_PtResolution_{self.jet_algo}"
            ]

        if self.junc_types:
            self.jec_uncsources_clib = [f"{self.jec_tag}_{junc_type}_{self.jet_algo}" for junc_type in self.junc_types]

        # Combine requested corrections
        requested_corrections = self.jec_names_clib + self.jer_names_clib + self.jec_uncsources_clib
        available_corrections = list(self.cset.keys())
        missing_corrections = [name for name in requested_corrections if name not in available_corrections]

        if missing_corrections:
            raise ValueError(
                f"\nMissing corrections in the CorrectionSet: {missing_corrections}. "
                f"\n\nAvailable corrections are: {available_corrections}. "
                f"\n\nRequested corrections are: {requested_corrections}"
            )

        # Store corrections directly in the JECStack for easy access
        self.corrections = {name: self.cset[name] for name in requested_corrections}

    def _initialize_jecstack(self):
        """Initialize the JECStack tools for the non-clib scenario."""
        assembled = self.assemble_corrections()

        if len(assembled["jec"]) > 0:
            self.jec = FactorizedJetCorrector(**assembled["jec"])
        if len(assembled["junc"]) > 0:
            self.junc = JetCorrectionUncertainty(**assembled["junc"])
        if len(assembled["jer"]) > 0:
            self.jer = JetResolution(**assembled["jer"])
        if len(assembled["jersf"]) > 0:
            self.jersf = JetResolutionScaleFactor(**assembled["jersf"])

        if (self.jer is None) != (self.jersf is None):
            raise ValueError("Cannot apply JER-SF without an input JER, and vice-versa!")

    def to_list(self):
        """Convert to list for clib case."""
        return self.jec_names_clib + self.jer_names_clib + self.jec_uncsources_clib + [self.json_path, self.savecorr]

    def assemble_corrections(self):
        """Assemble corrections for both scenarios."""
        assembled = {"jec": {}, "junc": {}, "jer": {}, "jersf": {}}

        for key in self.corrections.keys():
            if "Uncertainty" in key:
                assembled["junc"][key] = self.corrections[key]
            elif ("ScaleFactor" in key or "SF" in key):
                assembled["jersf"][key] = self.corrections[key]
            elif "Resolution" in key and not ("ScaleFactor" in key or "SF" in key):
                assembled["jer"][key] = self.corrections[key]
            elif len(_levelre.findall(key)) > 0:
                assembled["jec"][key] = self.corrections[key]
            else:
                print(f"Unknown correction type for key: {key}")

        return assembled

    @property
    def blank_name_map(self):
        """Returns a blank name map for corrections."""
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
        if self.jec is not None:
            for name in self.jec.signature:
                out.add(name)
        if self.junc is not None:
            for name in self.junc.signature:
                out.add(name)
        if self.jer is not None:
            for name in self.jer.signature:
                out.add(name)
        if self.jersf is not None:
            for name in self.jersf.signature:
                out.add(name)
        return {name: None for name in out}
