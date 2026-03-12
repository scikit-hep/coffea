"""Tests for the correctionlib (JSON-POG) adapter classes.

Two JSON-POG test files are used:
- ``jet_jerc.json.gz``: hand-crafted dummy corrections with simple constant outputs
  (e.g. JEC returns ``1 + 0.01*JetA``, JER returns flat ``0.1``).  Used for unit
  tests where exact expected values can be asserted.
- ``jet_jerc_Summer22_V3.json.gz``: real Summer22_V3 corrections.  Used in the
  cross-validation tests (``test_crossval_*``) to verify that correctionlib adapters
  match the txt-based correctors.
"""

import os

import awkward as ak
import dask_awkward as dak
import numpy as np
import pytest
from dummy_distributions import dummy_jagged_eta_pt

SAMPLES = os.path.join(os.path.dirname(__file__), "samples")
JERC_FILE = os.path.join(SAMPLES, "jet_jerc.json.gz")

JEC_TAG = "Summer24Prompt24_V2"
JER_TAG = "Summer23BPixPrompt23_RunD_JRV1"
DATA_TYPE = "MC"
JET_TYPE = "AK4PFPuppi"
UNC_SOURCES = ["Regrouped_FlavorQCD", "Regrouped_AbsoluteMPFBias"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clib_stack():
    from coffea.jetmet_tools import CorrectionLibJECStack

    return CorrectionLibJECStack.from_file(
        JERC_FILE,
        jec_tag=JEC_TAG,
        data_type=DATA_TYPE,
        jet_type=JET_TYPE,
        jec_level="L1L2L3Res",
        unc_sources=UNC_SOURCES,
        jer_tag=JER_TAG,
    )


@pytest.fixture
def flat_jet_arrays():
    """Return flat arrays (JetA, JetEta, JetPt, Rho) for testing."""
    counts, test_eta, test_pt = dummy_jagged_eta_pt()
    test_Rho = np.full_like(test_eta, 30.0)
    test_A = np.full_like(test_eta, 0.5)
    return counts, test_A, test_eta, test_pt, test_Rho


# ---------------------------------------------------------------------------
# Individual adapter tests
# ---------------------------------------------------------------------------


def test_correctionlib_jec(clib_stack, flat_jet_arrays):
    counts, test_A, test_eta, test_pt, test_Rho = flat_jet_arrays

    jec = clib_stack.jec
    assert "JetA" in jec.signature
    assert "JetPt" in jec.signature

    # Eager (flat numpy)
    corr = jec.getCorrection(
        JetA=ak.Array(test_A),
        JetEta=ak.Array(test_eta),
        JetPt=ak.Array(test_pt),
        Rho=ak.Array(test_Rho),
    )
    assert len(corr) == len(test_pt)
    # Each level returns 1+0.01*JetA = 1+0.01*0.5 = 1.005, compound multiplies 4 levels
    expected = np.float32(1.005) ** 4
    assert np.allclose(np.asarray(corr), expected, atol=1e-6)


def test_correctionlib_jec_jagged(clib_stack, flat_jet_arrays):
    counts, test_A, test_eta, test_pt, test_Rho = flat_jet_arrays

    jec = clib_stack.jec

    # Jagged awkward — adapters flatten internally then return 1D result
    A_jag = ak.unflatten(test_A, counts)
    eta_jag = ak.unflatten(test_eta, counts)
    pt_jag = ak.unflatten(test_pt, counts)
    Rho_jag = ak.unflatten(test_Rho, counts)

    corr_jag = jec.getCorrection(JetA=A_jag, JetEta=eta_jag, JetPt=pt_jag, Rho=Rho_jag)
    # Result preserves jagged structure
    assert len(corr_jag) == len(counts)
    assert len(ak.flatten(corr_jag)) == len(test_pt)


def test_correctionlib_jec_dask(clib_stack, flat_jet_arrays):
    counts, test_A, test_eta, test_pt, test_Rho = flat_jet_arrays

    jec = clib_stack.jec

    A_jag = ak.unflatten(test_A, counts)
    eta_jag = ak.unflatten(test_eta, counts)
    pt_jag = ak.unflatten(test_pt, counts)
    Rho_jag = ak.unflatten(test_Rho, counts)

    corr_dak = jec.getCorrection(
        JetA=dak.from_awkward(A_jag, 1),
        JetEta=dak.from_awkward(eta_jag, 1),
        JetPt=dak.from_awkward(pt_jag, 1),
        Rho=dak.from_awkward(Rho_jag, 1),
    )
    assert isinstance(corr_dak, dak.Array)
    result = corr_dak.compute()
    expected = np.float32(1.005) ** 4
    assert np.allclose(
        np.asarray(ak.flatten(result, axis=None)), expected, atol=1e-6
    )


def test_correctionlib_jer(clib_stack, flat_jet_arrays):
    counts, _, test_eta, test_pt, test_Rho = flat_jet_arrays

    jer = clib_stack.jer
    assert "JetEta" in jer.signature
    assert "JetPt" in jer.signature
    assert "Rho" in jer.signature

    reso = jer.getResolution(
        JetEta=ak.Array(test_eta),
        JetPt=ak.Array(test_pt),
        Rho=ak.Array(test_Rho),
    )
    assert len(reso) == len(test_pt)
    # Our test correction returns flat 0.1
    assert np.allclose(np.asarray(reso), 0.1, atol=1e-6)


def test_correctionlib_jersf(clib_stack, flat_jet_arrays):
    counts, _, test_eta, test_pt, test_Rho = flat_jet_arrays

    jersf = clib_stack.jersf
    # "systematic" should NOT be in the signature
    assert "systematic" not in jersf.signature
    assert "JetEta" in jersf.signature

    sf = jersf.getScaleFactor(JetEta=ak.Array(test_eta))
    assert isinstance(sf, ak.Array)
    # Shape should be (N, 3)
    arr = np.asarray(sf)
    assert arr.shape == (len(test_eta), 3)
    # nom=1.1, up=1.2, down=1.0
    assert np.allclose(arr[:, 0], 1.1, atol=1e-6)
    assert np.allclose(arr[:, 1], 1.2, atol=1e-6)
    assert np.allclose(arr[:, 2], 1.0, atol=1e-6)


def test_correctionlib_junc(clib_stack, flat_jet_arrays):
    counts, _, test_eta, test_pt, test_Rho = flat_jet_arrays

    junc = clib_stack.junc
    assert "JetEta" in junc.signature
    assert "JetPt" in junc.signature
    assert junc.levels == UNC_SOURCES

    uncs = junc.getUncertainty(
        JetEta=ak.Array(test_eta),
        JetPt=ak.Array(test_pt),
    )
    assert len(uncs) == 2
    for name, vals in uncs:
        assert name in UNC_SOURCES
        arr = np.asarray(vals)
        assert arr.shape == (len(test_eta), 2)
        # delta=0.02 => up=1.02, down=0.98
        assert np.allclose(arr[:, 0], 1.02, atol=1e-6)
        assert np.allclose(arr[:, 1], 0.98, atol=1e-6)


# ---------------------------------------------------------------------------
# Stack-level tests
# ---------------------------------------------------------------------------


def test_correctionlib_jec_stack_properties(clib_stack):
    assert clib_stack.jec is not None
    assert clib_stack.junc is not None
    assert clib_stack.jer is not None
    assert clib_stack.jersf is not None

    bm = clib_stack.blank_name_map
    assert "JetPt" in bm
    assert "JetMass" in bm
    assert "ptRaw" in bm
    assert "massRaw" in bm


def test_correctionlib_jec_stack_jer_validation():
    # Should raise when jer is set but jersf is not
    import correctionlib

    from coffea.jetmet_tools import CorrectionLibJECStack

    cset = correctionlib.CorrectionSet.from_file(JERC_FILE)
    jer_name = f"{JER_TAG}_{DATA_TYPE}_PtResolution_{JET_TYPE}"

    from coffea.jetmet_tools import CorrectionLibJER

    jer = CorrectionLibJER(cset[jer_name])
    with pytest.raises(ValueError, match="Cannot apply JER-SF"):
        CorrectionLibJECStack(jer=jer)


def test_correctionlib_jec_stack_no_jer():
    """Stack with JEC + JUNC only (no JER/JERSF)."""
    from coffea.jetmet_tools import CorrectionLibJECStack

    stack = CorrectionLibJECStack.from_file(
        JERC_FILE,
        jec_tag=JEC_TAG,
        data_type=DATA_TYPE,
        jet_type=JET_TYPE,
        unc_sources=UNC_SOURCES,
        # jer_tag not provided => no JER/JERSF
    )
    assert stack.jer is None
    assert stack.jersf is None
    assert stack.jec is not None
    assert stack.junc is not None


# ---------------------------------------------------------------------------
# End-to-end: CorrectedJetsFactory with correctionlib stack
# ---------------------------------------------------------------------------


def test_corrected_jets_factory_with_correctionlib(clib_stack):
    """Full integration test: build corrected jets using the correctionlib stack."""
    from coffea.jetmet_tools import CorrectedJetsFactory

    name_map = clib_stack.blank_name_map
    name_map["JetPt"] = "pt"
    name_map["JetMass"] = "mass"
    name_map["JetEta"] = "eta"
    name_map["JetA"] = "area"
    name_map["ptRaw"] = "pt_raw"
    name_map["massRaw"] = "mass_raw"
    name_map["Rho"] = "rho"
    name_map["ptGenJet"] = "pt_gen"
    name_map["METpt"] = "met_pt"
    name_map["METphi"] = "met_phi"
    name_map["JetPhi"] = "phi"
    name_map["UnClusteredEnergyDeltaX"] = "ue_dx"
    name_map["UnClusteredEnergyDeltaY"] = "ue_dy"

    factory = CorrectedJetsFactory(name_map, clib_stack)

    # Build synthetic jets
    counts, test_eta, test_pt = dummy_jagged_eta_pt()

    n_flat = len(test_eta)
    jets_dict = {
        "pt": test_pt.astype(np.float32),
        "mass": (test_pt * 0.1).astype(np.float32),
        "eta": test_eta.astype(np.float32),
        "phi": np.zeros(n_flat, dtype=np.float32),
        "area": np.full(n_flat, 0.5, dtype=np.float32),
        "pt_raw": test_pt.astype(np.float32),
        "mass_raw": (test_pt * 0.1).astype(np.float32),
        "rho": np.full(n_flat, 30.0, dtype=np.float32),
        "pt_gen": (test_pt * 0.95).astype(np.float32),
    }
    jets_jag = ak.unflatten(ak.zip(jets_dict), counts)

    corrected = factory.build(jets_jag)
    assert corrected is not None

    # Check corrected pt exists and has the right shape
    flat_corrected = ak.flatten(corrected)
    assert "pt" in ak.fields(flat_corrected)
    assert len(flat_corrected) == n_flat

    # Check uncertainties are present
    uncs = factory.uncertainties()
    assert "JER" in uncs
    assert f"JES_{UNC_SOURCES[0]}" in uncs
    assert f"JES_{UNC_SOURCES[1]}" in uncs

    # Check that JES variations are accessible
    for unc in uncs:
        assert unc in ak.fields(flat_corrected)
        assert "up" in ak.fields(flat_corrected[unc])
        assert "down" in ak.fields(flat_corrected[unc])


def test_corrected_jets_factory_with_correctionlib_dask(clib_stack):
    """Full integration test with dask arrays."""
    from coffea.jetmet_tools import CorrectedJetsFactory

    name_map = clib_stack.blank_name_map
    name_map["JetPt"] = "pt"
    name_map["JetMass"] = "mass"
    name_map["JetEta"] = "eta"
    name_map["JetA"] = "area"
    name_map["ptRaw"] = "pt_raw"
    name_map["massRaw"] = "mass_raw"
    name_map["Rho"] = "rho"
    name_map["ptGenJet"] = "pt_gen"
    name_map["METpt"] = "met_pt"
    name_map["METphi"] = "met_phi"
    name_map["JetPhi"] = "phi"
    name_map["UnClusteredEnergyDeltaX"] = "ue_dx"
    name_map["UnClusteredEnergyDeltaY"] = "ue_dy"

    factory = CorrectedJetsFactory(name_map, clib_stack)

    counts, test_eta, test_pt = dummy_jagged_eta_pt()
    n_flat = len(test_eta)

    jets_dict = {
        "pt": test_pt.astype(np.float32),
        "mass": (test_pt * 0.1).astype(np.float32),
        "eta": test_eta.astype(np.float32),
        "phi": np.zeros(n_flat, dtype=np.float32),
        "area": np.full(n_flat, 0.5, dtype=np.float32),
        "pt_raw": test_pt.astype(np.float32),
        "mass_raw": (test_pt * 0.1).astype(np.float32),
        "rho": np.full(n_flat, 30.0, dtype=np.float32),
        "pt_gen": (test_pt * 0.95).astype(np.float32),
    }
    jets_jag = ak.unflatten(ak.zip(jets_dict), counts)
    jets_dak = dak.from_awkward(jets_jag, 1)

    corrected = factory.build(jets_dak)
    assert isinstance(corrected, dak.Array)

    # Compute and check
    result = corrected.compute()
    flat_result = ak.flatten(result)
    assert "pt" in ak.fields(flat_result)
    assert len(flat_result) == n_flat


def test_corrected_jets_factory_jec_only():
    """Test with JEC only (no JER, no JUNC)."""
    from coffea.jetmet_tools import CorrectedJetsFactory, CorrectionLibJECStack

    stack = CorrectionLibJECStack.from_file(
        JERC_FILE,
        jec_tag=JEC_TAG,
        data_type=DATA_TYPE,
        jet_type=JET_TYPE,
        # no unc_sources, no jer_tag
    )

    name_map = stack.blank_name_map
    name_map["JetPt"] = "pt"
    name_map["JetMass"] = "mass"
    name_map["JetEta"] = "eta"
    name_map["JetA"] = "area"
    name_map["ptRaw"] = "pt_raw"
    name_map["massRaw"] = "mass_raw"
    name_map["Rho"] = "rho"
    name_map["METpt"] = "met_pt"
    name_map["METphi"] = "met_phi"
    name_map["JetPhi"] = "phi"
    name_map["UnClusteredEnergyDeltaX"] = "ue_dx"
    name_map["UnClusteredEnergyDeltaY"] = "ue_dy"

    factory = CorrectedJetsFactory(name_map, stack)
    assert factory.uncertainties() == []

    counts, test_eta, test_pt = dummy_jagged_eta_pt()
    n_flat = len(test_eta)
    jets_dict = {
        "pt": test_pt.astype(np.float32),
        "mass": (test_pt * 0.1).astype(np.float32),
        "eta": test_eta.astype(np.float32),
        "phi": np.zeros(n_flat, dtype=np.float32),
        "area": np.full(n_flat, 0.5, dtype=np.float32),
        "pt_raw": test_pt.astype(np.float32),
        "mass_raw": (test_pt * 0.1).astype(np.float32),
        "rho": np.full(n_flat, 30.0, dtype=np.float32),
    }
    jets_jag = ak.unflatten(ak.zip(jets_dict), counts)
    corrected = factory.build(jets_jag)

    # JEC-only: corrected pt should be pt_raw * correction_factor
    flat_corrected = ak.flatten(corrected)
    expected_factor = np.float32(1.005) ** 4
    expected_pt = test_pt.astype(np.float32) * expected_factor
    assert np.allclose(np.asarray(flat_corrected["pt"]), expected_pt, rtol=1e-5)


# ---------------------------------------------------------------------------
# Cross-validation: correctionlib adapters vs txt-based correctors
# ---------------------------------------------------------------------------

REAL_JERC_FILE = os.path.join(SAMPLES, "jet_jerc_Summer22_V3.json.gz")
REAL_JEC_TAG = "Summer22_22Sep2023_V3"
REAL_JER_TAG = "Summer22_22Sep2023_JRV1"
REAL_DATA_TYPE = "MC"
REAL_JET_TYPE = "AK4PFPuppi"
# Pick a few sources that exist in both txt and JSON-POG
XVAL_UNC_SOURCES = ["FlavorQCD", "AbsoluteMPFBias"]


def _make_txt_evaluator():
    """Build an evaluator from the real Summer22_V3 txt files (JEC + JUNC + JER + JERSF)."""
    from coffea.lookup_tools import extractor

    extract = extractor()
    extract.add_weight_sets(
        [
            "* * tests/samples/Summer22_22Sep2023_V3_MC_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "* * tests/samples/Summer22_22Sep2023_V3_MC_L2Relative_AK4PFPuppi.jec.txt.gz",
            "* * tests/samples/Summer22_22Sep2023_V3_MC_L3Absolute_AK4PFPuppi.jec.txt.gz",
            "* * tests/samples/Summer22_22Sep2023_V3_MC_L2L3Residual_AK4PFPuppi.jec.txt.gz",
            "* * tests/samples/Summer22_22Sep2023_V3_MC_UncertaintySources_AK4PFPuppi.junc.txt.gz",
            "* * tests/samples/Summer22_22Sep2023_JRV1_MC_PtResolution_AK4PFPuppi.jr.txt.gz",
            # JERSF: rename to a 5-part name so JetResolutionScaleFactor's parser accepts it
            # (the real name has 6 underscore-separated parts due to "Summer22_22Sep2023")
            "Summer2222Sep2023_JRV1_MC_SF_AK4PFPuppi Summer22_22Sep2023_JRV1_MC_SF_AK4PFPuppi tests/samples/Summer22_22Sep2023_JRV1_MC_SF_AK4PFPuppi.jersf.txt.gz",
        ]
    )
    extract.finalize()
    return extract.make_evaluator()


def test_crossval_jec():
    """JEC from correctionlib compound correction must match txt FactorizedJetCorrector."""
    from coffea.jetmet_tools import CorrectionLibJECStack, FactorizedJetCorrector

    # -- txt-based JEC --
    ev = _make_txt_evaluator()
    jec_names = [
        "Summer22_22Sep2023_V3_MC_L1FastJet_AK4PFPuppi",
        "Summer22_22Sep2023_V3_MC_L2Relative_AK4PFPuppi",
        "Summer22_22Sep2023_V3_MC_L3Absolute_AK4PFPuppi",
        "Summer22_22Sep2023_V3_MC_L2L3Residual_AK4PFPuppi",
    ]
    txt_jec = FactorizedJetCorrector(**{name: ev[name] for name in jec_names})

    # -- correctionlib JEC --
    clib_stack = CorrectionLibJECStack.from_file(
        REAL_JERC_FILE,
        jec_tag=REAL_JEC_TAG,
        data_type=REAL_DATA_TYPE,
        jet_type=REAL_JET_TYPE,
    )
    clib_jec = clib_stack.jec

    # -- Build test inputs --
    _, test_eta, test_pt = dummy_jagged_eta_pt()
    test_Rho = np.full_like(test_eta, 30.0)
    test_A = np.full_like(test_eta, 0.5)

    # txt corrector works with numpy arrays
    txt_corr = txt_jec.getCorrection(
        JetEta=test_eta, Rho=test_Rho, JetPt=test_pt, JetA=test_A
    )
    txt_corr = np.asarray(txt_corr, dtype=np.float32)

    # correctionlib adapter works with awkward arrays
    clib_corr = clib_jec.getCorrection(
        JetA=ak.Array(test_A),
        JetEta=ak.Array(test_eta),
        JetPt=ak.Array(test_pt),
        Rho=ak.Array(test_Rho),
    )
    clib_corr = np.asarray(clib_corr, dtype=np.float32)

    assert txt_corr.shape == clib_corr.shape
    assert np.allclose(
        txt_corr, clib_corr, rtol=1e-5
    ), f"Max relative diff: {np.max(np.abs(txt_corr - clib_corr) / np.abs(txt_corr))}"


def test_crossval_junc():
    """JUNC from correctionlib must match txt JetCorrectionUncertainty."""
    from coffea.jetmet_tools import (
        CorrectionLibJECStack,
        JetCorrectionUncertainty,
    )

    # -- txt-based JUNC --
    ev = _make_txt_evaluator()
    junc_names = [k for k in dir(ev) if "UncertaintySources" in k and "AK4PFPuppi" in k]
    txt_junc = JetCorrectionUncertainty(**{name: ev[name] for name in junc_names})

    # -- correctionlib JUNC --
    clib_stack = CorrectionLibJECStack.from_file(
        REAL_JERC_FILE,
        jec_tag=REAL_JEC_TAG,
        data_type=REAL_DATA_TYPE,
        jet_type=REAL_JET_TYPE,
        unc_sources=XVAL_UNC_SOURCES,
    )
    clib_junc = clib_stack.junc

    # -- Build test inputs --
    _, test_eta, test_pt = dummy_jagged_eta_pt()

    # txt JUNC
    txt_results = dict(txt_junc.getUncertainty(JetEta=test_eta, JetPt=test_pt))

    # correctionlib JUNC
    clib_results = dict(
        clib_junc.getUncertainty(JetEta=ak.Array(test_eta), JetPt=ak.Array(test_pt))
    )

    for source in XVAL_UNC_SOURCES:
        assert source in clib_results, f"Source {source} missing from correctionlib"
        assert source in txt_results, f"Source {source} missing from txt results"

        txt_arr = np.asarray(txt_results[source], dtype=np.float32)
        clib_arr = np.asarray(clib_results[source], dtype=np.float32)

        assert (
            txt_arr.shape == clib_arr.shape
        ), f"Shape mismatch for {source}: txt={txt_arr.shape}, clib={clib_arr.shape}"
        assert np.allclose(txt_arr, clib_arr, rtol=1e-5), (
            f"Source {source} max relative diff: "
            f"{np.max(np.abs(txt_arr - clib_arr) / np.clip(np.abs(txt_arr), 1e-10, None))}"
        )


def test_crossval_jer():
    """JER PtResolution from correctionlib must match txt JetResolution."""
    from coffea.jetmet_tools import CorrectionLibJECStack, JetResolution

    # -- txt-based JER --
    ev = _make_txt_evaluator()
    jer_name = "Summer22_22Sep2023_JRV1_MC_PtResolution_AK4PFPuppi"
    txt_jer = JetResolution(**{jer_name: ev[jer_name]})

    # -- correctionlib JER --
    clib_stack = CorrectionLibJECStack.from_file(
        REAL_JERC_FILE,
        jec_tag=REAL_JEC_TAG,
        data_type=REAL_DATA_TYPE,
        jet_type=REAL_JET_TYPE,
        jer_tag=REAL_JER_TAG,
    )
    clib_jer = clib_stack.jer

    # -- Build test inputs --
    _, test_eta, test_pt = dummy_jagged_eta_pt()
    test_Rho = np.full_like(test_eta, 30.0)

    # txt JER
    txt_reso = txt_jer.getResolution(JetEta=test_eta, Rho=test_Rho, JetPt=test_pt)
    txt_reso = np.asarray(txt_reso, dtype=np.float32)

    # correctionlib JER
    clib_reso = clib_jer.getResolution(
        JetEta=ak.Array(test_eta),
        JetPt=ak.Array(test_pt),
        Rho=ak.Array(test_Rho),
    )
    clib_reso = np.asarray(clib_reso, dtype=np.float32)

    assert txt_reso.shape == clib_reso.shape
    assert np.allclose(
        txt_reso, clib_reso, rtol=1e-5
    ), f"Max relative diff: {np.max(np.abs(txt_reso - clib_reso) / np.clip(np.abs(txt_reso), 1e-10, None))}"


def test_crossval_jersf():
    """JERSF from correctionlib must match txt JetResolutionScaleFactor.

    Note: the JSON-POG JERSF is pt-dependent and returns SF=1.0 for jets below
    a pt threshold (~10 GeV), while the txt format only bins by eta and always
    returns the SF.  We compare only jets above pt=10 GeV where both agree.
    """
    from coffea.jetmet_tools import CorrectionLibJECStack, JetResolutionScaleFactor

    # -- txt-based JERSF --
    ev = _make_txt_evaluator()
    # Use the renamed 5-part key (see _make_txt_evaluator)
    jersf_name = "Summer2222Sep2023_JRV1_MC_SF_AK4PFPuppi"
    txt_jersf = JetResolutionScaleFactor(**{jersf_name: ev[jersf_name]})

    # -- correctionlib JERSF --
    clib_stack = CorrectionLibJECStack.from_file(
        REAL_JERC_FILE,
        jec_tag=REAL_JEC_TAG,
        data_type=REAL_DATA_TYPE,
        jet_type=REAL_JET_TYPE,
        jer_tag=REAL_JER_TAG,
    )
    clib_jersf = clib_stack.jersf

    # -- Build test inputs, filter to pt >= 10 where both formats agree --
    _, test_eta, test_pt = dummy_jagged_eta_pt()
    valid = test_pt >= 10.0
    test_eta = test_eta[valid]
    test_pt = test_pt[valid]
    assert len(test_eta) > 0

    # txt JERSF returns (N, 3) with [nom, up, down]
    txt_sf = txt_jersf.getScaleFactor(JetEta=test_eta, JetPt=test_pt)
    txt_sf = np.asarray(txt_sf, dtype=np.float32)

    # correctionlib JERSF also returns (N, 3) with [nom, up, down]
    clib_sf = clib_jersf.getScaleFactor(
        JetEta=ak.Array(test_eta),
        JetPt=ak.Array(test_pt),
    )
    clib_sf = np.asarray(clib_sf, dtype=np.float32)

    assert (
        txt_sf.shape == clib_sf.shape
    ), f"Shape mismatch: txt={txt_sf.shape}, clib={clib_sf.shape}"
    assert np.allclose(
        txt_sf, clib_sf, rtol=1e-5
    ), f"Max relative diff: {np.max(np.abs(txt_sf - clib_sf) / np.clip(np.abs(txt_sf), 1e-10, None))}"
