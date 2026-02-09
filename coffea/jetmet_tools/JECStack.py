import os
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Union
from coffea.jetmet_tools.FactorizedJetCorrector import FactorizedJetCorrector, _levelre
from coffea.jetmet_tools.JetResolution import JetResolution
from coffea.jetmet_tools.JetResolutionScaleFactor import JetResolutionScaleFactor
from coffea.jetmet_tools.JetCorrectionUncertainty import JetCorrectionUncertainty
import correctionlib as clib
import correctionlib.schemav2  # Ensure schemav2 is registered on the package


_GLOBAL_JECSTACK_CACHE: Dict[str, Dict[str, Any]] = {}


@dataclass
class JECStack:
    """Handles both JEC and clib cases with conditional attributes.

    The clib pathway can be configured either with a concrete ``json_path``, a
    fully materialized :class:`correctionlib.schemav2.CorrectionSet`, or a
    resolver that returns one of those.  Resolvers may be callables or string
    templates (``"/path/{jec_tag}_{jet_algo}.json"``) evaluated against the
    dataclass fields.  This makes it easy to point to local or site-specific
    locations instead of relying on CVMFS defaults.
    """

    # Common fields for both scenarios
    corrections: Dict[str, any] = field(default_factory=dict)
    use_clib: bool = False  # Set to True if useclib is needed

    # Fields for the clib scenario (useclib=True)
    jec_tag: Optional[str] = None
    jec_levels: Optional[List[str]] = field(default_factory=list)
    jer_tag: Optional[str] = None
    jet_algo: Optional[str] = None
    junc_types: Optional[List[str]] = field(default_factory=list)
    json_path: Optional[Union[str, PathLike]] = None
    year: Optional[Union[int, str]] = None
    json_search_dirs: Optional[List[Union[str, PathLike]]] = None
    correction_set: Optional[Union[clib.CorrectionSet, clib.schemav2.CorrectionSet]] = None
    resolver: Optional[
        Union[
            str,
            Callable[
                ["JECStack"],
                Union[str, PathLike, clib.CorrectionSet, clib.schemav2.CorrectionSet],
            ],
        ]
    ] = None
    resolved_json_path: Optional[str] = None
    cache: Optional[MutableMapping[str, Dict[str, Any]]] = None
    enable_cache: bool = False
    cache_identifier: Optional[str] = None

    # Fields for the usejecstack scenario (useclib=False)
    jec: Optional[FactorizedJetCorrector] = None
    junc: Optional[JetCorrectionUncertainty] = None
    jer: Optional[JetResolution] = None
    jersf: Optional[JetResolutionScaleFactor] = None

    def __post_init__(self):
        """Handle initialization based on use_clib flag."""
        self._cache_key: Optional[str] = self.cache_identifier
        if self.use_clib:
            if (
                self.json_path is None
                and self.resolver is None
                and self.correction_set is None
            ):
                inferred = self._infer_json_path()
                if inferred is not None:
                    self.json_path = inferred
            self._initialize_clib()
        else:
            self._initialize_jecstack()

    def _initialize_clib(self):
        """Initialize the clib-based correction tools."""
        self.cset, self.resolved_json_path = self._resolve_cset()

        # Construct lists for jec, jer, and uncertainties
        self.jec_names_clib = [
            f"{self.jec_tag}_{level}_{self.jet_algo}" for level in self.jec_levels
        ]
        self.jer_names_clib = []
        self.jec_uncsources_clib = []

        if self.jer_tag is not None:
            self.jer_names_clib = [
                f"{self.jer_tag}_ScaleFactor_{self.jet_algo}",
                f"{self.jer_tag}_PtResolution_{self.jet_algo}",
            ]

        if self.junc_types:
            self.jec_uncsources_clib = [
                f"{self.jec_tag}_{junc_type}_{self.jet_algo}"
                for junc_type in self.junc_types
            ]

        # Combine requested corrections
        requested_corrections = (
            self.jec_names_clib + self.jer_names_clib + self.jec_uncsources_clib
        )
        available_corrections = list(self.cset.keys())
        missing_corrections = [
            name for name in requested_corrections if name not in available_corrections
        ]

        if missing_corrections:
            raise ValueError(
                f"\nMissing corrections in the CorrectionSet: {missing_corrections}. "
                f"\n\nAvailable corrections are: {available_corrections}. "
                f"\n\nRequested corrections are: {requested_corrections}"
            )

        # Store corrections directly in the JECStack for easy access, reusing any
        # cached handles to avoid repeated construction work.
        cache_entry = self._get_cache_entry()
        cached_corrections = None
        if cache_entry is not None:
            cached_corrections = cache_entry.setdefault("corrections", {})

        self.corrections = {}
        for name in requested_corrections:
            if cached_corrections is not None and name in cached_corrections:
                self.corrections[name] = cached_corrections[name]
                continue

            corr = self.cset[name]
            self.corrections[name] = corr
            if cached_corrections is not None:
                cached_corrections[name] = corr

        # Collect the full set of input variables used by any correction
        self.correction_inputs = set()
        for corr in self.corrections.values():
            self.correction_inputs.update(
                inp.name for inp in corr.inputs if inp.name != "systematic"
            )

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
            raise ValueError(
                "Cannot apply JER-SF without an input JER, and vice-versa!"
            )

    def to_list(self):
        """Convert to list for clib case."""
        return (
            self.jec_names_clib
            + self.jer_names_clib
            + self.jec_uncsources_clib
            + [self.resolved_json_path or self.json_path]
        )

    def _resolve_cset(self):
        """Resolve the correction set or path for clib initialization."""

        if self.correction_set is not None:
            cset = self._ensure_highlevel_cset(self.correction_set)
            self._maybe_store_cset_in_cache(cset)
            return cset, None

        source = None
        if self.json_path is not None:
            source = self.json_path
        elif self.resolver is not None:
            if callable(self.resolver):
                source = self.resolver(self)
            elif isinstance(self.resolver, str):
                format_map = {
                    "jec_tag": self.jec_tag,
                    "jer_tag": self.jer_tag,
                    "jet_algo": self.jet_algo,
                }
                source = self.resolver.format(**{k: v for k, v in format_map.items() if v is not None})
            else:
                raise TypeError(
                    "resolver must be a callable or a string path template"
                )

        if isinstance(source, (clib.CorrectionSet, clib.schemav2.CorrectionSet)):
            cset = self._ensure_highlevel_cset(source)
            self._maybe_store_cset_in_cache(cset)
            return cset, None

        if isinstance(source, (str, PathLike)):
            resolved_path = os.path.abspath(os.fspath(source))
            cache_key = self.cache_identifier or resolved_path
            cset = self._load_cset_from_path(resolved_path, cache_key)
            self._cache_key = cache_key
            return cset, resolved_path

        raise ValueError(
            "A json_path, correction_set, or resolver is required for clib initialization."
        )

    def _infer_json_path(self) -> Optional[str]:
        """Infer the correctionlib JSON path from standard directory layouts."""

        if self.jec_tag is None or self.jet_algo is None:
            return None

        search_dirs = self._collect_json_search_dirs()
        if not search_dirs:
            return None

        year_segment = None
        if self.year is not None:
            year_segment = str(self.year)

        filename_root = f"{self.jec_tag}_{self.jet_algo}"
        candidate_suffixes = [".json", ".corr.json", ".json.gz", ".corr.json.gz"]
        filenames = [f"{filename_root}{suffix}" for suffix in candidate_suffixes]

        for base in search_dirs:
            base = Path(base)
            if year_segment is not None:
                for fname in filenames:
                    candidate = base / year_segment / fname
                    if candidate.exists():
                        return os.fspath(candidate)
            for fname in filenames:
                candidate = base / fname
                if candidate.exists():
                    return os.fspath(candidate)

        return None

    def _collect_json_search_dirs(self) -> List[Path]:
        """Gather candidate directories for correctionlib JSON discovery."""

        search_dirs: List[Path] = []

        def _append(entry: Union[str, PathLike, Path]):
            path = Path(entry).expanduser()
            if path not in search_dirs:
                search_dirs.append(path)

        if self.json_search_dirs:
            for entry in self.json_search_dirs:
                _append(entry)

        env_value = os.environ.get("COFFEA_JME_JSONS")
        if env_value:
            for raw_entry in env_value.split(os.pathsep):
                raw_entry = raw_entry.strip()
                if raw_entry:
                    _append(raw_entry)

        default_dir = Path("/cvmfs/cms.cern.ch/rsync/cms-jet/JME-JSONs")
        if default_dir.exists():
            _append(default_dir)

        return search_dirs

    def _maybe_store_cset_in_cache(self, cset: clib.CorrectionSet) -> None:
        cache_entry = self._get_cache_entry(force_key=self.cache_identifier)
        if cache_entry is not None and "cset" not in cache_entry:
            cache_entry["cset"] = cset

    def _load_cset_from_path(self, resolved_path: str, cache_key: Optional[str]):
        cache_entry = self._get_cache_entry(force_key=cache_key)
        if cache_entry is not None and "cset" in cache_entry:
            return cache_entry["cset"]

        cset = clib.CorrectionSet.from_file(resolved_path)
        cache_entry = self._get_cache_entry(force_key=cache_key)
        if cache_entry is not None:
            cache_entry["cset"] = cset
        return cset

    def _ensure_highlevel_cset(
        self, cset: Union[clib.CorrectionSet, clib.schemav2.CorrectionSet]
    ) -> clib.CorrectionSet:
        """Convert schema objects into high-level correction sets."""

        if isinstance(cset, clib.CorrectionSet):
            return cset

        if isinstance(cset, clib.schemav2.CorrectionSet):
            return clib.CorrectionSet.from_string(cset.json())

        raise TypeError("Unsupported correction set type for conversion")

    def _get_cache_mapping(self) -> Optional[MutableMapping[str, Dict[str, Any]]]:
        if self.cache is not None:
            return self.cache
        if self.enable_cache:
            return _GLOBAL_JECSTACK_CACHE
        return None

    def _get_cache_entry(
        self, force_key: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        cache_map = self._get_cache_mapping()
        key = force_key or self._cache_key
        if cache_map is None or key is None:
            return None
        return cache_map.setdefault(key, {})

    def assemble_corrections(self):
        """Assemble corrections for both scenarios."""
        assembled = {"jec": {}, "junc": {}, "jer": {}, "jersf": {}}

        for key in self.corrections.keys():
            if "Uncertainty" in key:
                assembled["junc"][key] = self.corrections[key]
            elif "ScaleFactor" in key or "SF" in key:
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
        if self.use_clib:
            out.update(getattr(self, "correction_inputs", set()))
            return {name: name for name in out}

        if self.jec is not None:
            out.update(self.jec.signature)
        if self.junc is not None:
            out.update(self.junc.signature)
        if self.jer is not None:
            out.update(self.jer.signature)
        if self.jersf is not None:
            out.update(self.jersf.signature)
        return {name: None for name in out}
