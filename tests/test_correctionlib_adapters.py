"""Tests for the correctionlib (JSON-POG) adapter classes."""

import os

import awkward as ak
import dask
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
    assert isinstance(corr, ak.Array)
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
    # Result is flat (1D) since adapters flatten internally
    assert len(corr_jag) == len(test_pt)


@pytest.mark.parametrize("optimization_enabled", [True, False])
def test_correctionlib_jec_dask(clib_stack, flat_jet_arrays, optimization_enabled):
    with dask.config.set({"awkward.optimization.enabled": optimization_enabled}):
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
        # Result is flat (1D) since adapters flatten internally
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
    assert isinstance(reso, ak.Array)
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


@pytest.mark.parametrize("optimization_enabled", [True, False])
def test_corrected_jets_factory_with_correctionlib_dask(
    clib_stack, optimization_enabled
):
    """Full integration test with dask arrays."""
    with dask.config.set({"awkward.optimization.enabled": optimization_enabled}):
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
