from __future__ import print_function, division

import os
from types import SimpleNamespace

from coffea import lookup_tools
import awkward as ak
import pytest
from coffea.util import numpy as np
from coffea.nanoevents import NanoEventsFactory

from dummy_distributions import dummy_jagged_eta_pt


class _DummyCorrection:
    def __init__(self, name, inputs, metadata=None, evaluate_fn=None):
        self.name = name
        self.inputs = [SimpleNamespace(name=inp) for inp in inputs]
        self._base = SimpleNamespace(inputs=self.inputs)
        self.metadata = metadata or {}
        self._evaluate_fn = evaluate_fn
        self.calls = []

    def evaluate(self, *args, **kwargs):
        if self._evaluate_fn is not None:
            result = self._evaluate_fn(*args, **kwargs)
        else:
            result = {"args": args, "kwargs": kwargs}
        self.calls.append({"args": args, "kwargs": kwargs, "result": result})
        return result


@pytest.fixture
def legacy_jec_correction():
    def _evaluate(*args, **kwargs):
        return args, kwargs

    return _DummyCorrection(
        name="LegacyJEC",
        inputs=("JetPt", "JetEta"),
        metadata={"type": "scale_factor"},
        evaluate_fn=_evaluate,
    )


@pytest.fixture
def correctionlib_jec_stack():
    def _evaluate(**kwargs):
        ordered_keys = ("JetPt", "JetEta", "Rho", "systematic")
        return tuple(kwargs.get(key) for key in ordered_keys)

    return _DummyCorrection(
        name="MyCorrectionJECStack",
        inputs=("JetPt", "JetEta", "Rho", "systematic"),
        metadata={"type": "jec", "jec_stack": True},
        evaluate_fn=_evaluate,
    )

# From make_expected_lookup.py
_testSF2d_expected_output = np.array(
    [
        0.90780139,
        0.82748538,
        0.86332178,
        0.86332178,
        0.97981155,
        0.79701495,
        0.88245934,
        0.82857144,
        0.91884059,
        0.97466666,
        0.94072163,
        1.00775194,
        0.82748538,
        1.00775194,
        0.97203946,
        0.98199672,
        0.80655736,
        0.90893763,
        0.88245934,
        0.79701495,
        0.82748538,
        0.82857144,
        0.91884059,
        0.90893763,
        0.97520661,
        0.97520661,
        0.82748538,
        0.91884059,
        0.97203946,
        0.88245934,
        0.79701495,
        0.9458763,
        1.00775194,
        0.80655736,
        1.00775194,
        1.00775194,
        0.98976982,
        0.98976982,
        0.86332178,
        0.94072163,
        0.80655736,
        0.98976982,
        0.96638656,
        0.9458763,
        0.90893763,
        0.9529984,
        0.9458763,
        0.9529984,
        0.80655736,
        0.80655736,
        0.80655736,
        0.98976982,
        0.97466666,
        0.98199672,
        0.86332178,
        1.03286386,
        0.94072163,
        1.03398061,
        0.82857144,
        0.80655736,
        1.00775194,
        0.80655736,
    ]
)


def test_extractor_exceptions():
    extractor = lookup_tools.extractor()

    # test comments
    extractor.add_weight_sets(["#testSF2d asdfgh tests/samples/testSF2d.histo.root"])

    # test malformed statement
    try:
        extractor.add_weight_sets(
            ["testSF2d testSF2d asdfgh tests/samples/testSF2d.histo.root"]
        )
    except Exception as e:
        assert (
            e.args[0]
            == '"testSF2d testSF2d asdfgh tests/samples/testSF2d.histo.root" not formatted as "<local name> <name> <weights file>"'
        )

    # test not existant file entry
    try:
        extractor.add_weight_sets(["testSF2d asdfgh tests/samples/testSF2d.histo.root"])
    except Exception as e:
        assert (
            e.args[0]
            == 'Weights named "asdfgh" not in tests/samples/testSF2d.histo.root!'
        )

    # test unfinalized evaluator
    try:
        extractor.add_weight_sets(
            ["testSF2d scalefactors_Tight_Electron tests/samples/testSF2d.histo.root"]
        )
        extractor.make_evaluator()
    except Exception as e:
        assert e.args[0] == "Cannot make an evaluator from unfinalized extractor!"


def test_evaluator_exceptions():
    extractor = lookup_tools.extractor()
    extractor.add_weight_sets(
        ["testSF2d scalefactors_Tight_Electron tests/samples/testSF2d.histo.root"]
    )

    counts, test_eta, test_pt = dummy_jagged_eta_pt()
    # test_eta_jagged = ak.unflatten(test_eta, counts)
    test_pt_jagged = ak.unflatten(test_pt, counts)

    extractor.finalize()
    evaluator = extractor.make_evaluator()

    try:
        evaluator["testSF2d"](test_pt_jagged, test_eta)
    except Exception as e:
        assert isinstance(e, TypeError)


def test_evaluate_noimpl():
    from coffea.lookup_tools.lookup_base import lookup_base

    try:
        lookup_base()._evaluate()
    except NotImplementedError:
        pass


def test_correctionlib():
    extractor = lookup_tools.extractor()
    extractor.add_weight_sets(["* * tests/samples/testSF2d.corr.json.gz"])

    extractor.finalize()

    evaluator = extractor.make_evaluator()

    counts, test_eta, test_pt = dummy_jagged_eta_pt()

    test_out = evaluator["scalefactors_Tight_Electron"](test_eta, test_pt)

    # print it
    print(evaluator["scalefactors_Tight_Electron"])

    # test structured eval
    test_eta_jagged = ak.unflatten(test_eta, counts)
    test_pt_jagged = ak.unflatten(test_pt, counts)
    test_out_jagged = evaluator["scalefactors_Tight_Electron"](
        test_eta_jagged, test_pt_jagged
    )

    assert ak.all(ak.num(test_out_jagged) == counts)
    assert ak.all(ak.flatten(test_out_jagged) == test_out)

    print(test_out)

    diff = np.abs(test_out - _testSF2d_expected_output)
    print("Max diff: %.16f" % diff.max())
    print("Median diff: %.16f" % np.median(diff))
    print(
        "Diff over threshold rate: %.1f %%" % (100 * (diff >= 1.0e-8).sum() / diff.size)
    )
    assert (diff < 1.0e-8).all()


def test_correctionlib_wrapper_jec_stack_detection():
    class DummyCorrection:
        def __init__(self):
            self.metadata = {"type": "jec", "jec_stack": True}
            self.inputs = [SimpleNamespace(name="JetPt"), SimpleNamespace(name="systematic")]
            self.name = "MyJECStack"
            self.called_with = None

        def evaluate(self, **kwargs):
            self.called_with = kwargs
            return kwargs.get("JetPt", None), kwargs.get("systematic", None)

    corr = DummyCorrection()
    wrapper = lookup_tools.correctionlib_wrapper(corr)

    jet_pt = 50.0
    out_pt, out_sys = wrapper(jet_pt)

    assert corr.called_with == {"JetPt": jet_pt, "systematic": "nom"}
    assert out_pt == jet_pt
    assert out_sys == "nom"


def test_correctionlib_wrapper_legacy_signature(legacy_jec_correction):
    wrapper = lookup_tools.correctionlib_wrapper(legacy_jec_correction)

    jet_pt = 75.0
    jet_eta = 1.25

    result_args, result_kwargs = wrapper(jet_pt, jet_eta, flavor="b")

    assert not wrapper._is_jec_stack
    assert legacy_jec_correction.calls[-1]["args"] == (jet_pt, jet_eta)
    assert legacy_jec_correction.calls[-1]["kwargs"] == {"flavor": "b"}
    assert result_args == (jet_pt, jet_eta)
    assert result_kwargs == {"flavor": "b"}
    assert "LegacyJEC(JetPt,JetEta)" in repr(wrapper)


def test_correctionlib_wrapper_jec_stack_evaluation_paths(correctionlib_jec_stack):
    wrapper = lookup_tools.correctionlib_wrapper(correctionlib_jec_stack)

    jet_pt = 120.0
    jet_eta = -0.4
    rho = 18.5

    positional_out = wrapper(jet_pt, jet_eta, rho)
    dict_out = wrapper({"JetPt": jet_pt, "JetEta": jet_eta, "Rho": rho})
    override_out = wrapper(jet_pt, jet_eta, rho, systematic="up")

    assert positional_out == (jet_pt, jet_eta, rho, "nom")
    assert dict_out == (jet_pt, jet_eta, rho, "nom")
    assert override_out == (jet_pt, jet_eta, rho, "up")

    assert correctionlib_jec_stack.calls[0]["kwargs"] == {
        "JetPt": jet_pt,
        "JetEta": jet_eta,
        "Rho": rho,
        "systematic": "nom",
    }
    assert correctionlib_jec_stack.calls[1]["kwargs"] == {
        "JetPt": jet_pt,
        "JetEta": jet_eta,
        "Rho": rho,
        "systematic": "nom",
    }
    assert correctionlib_jec_stack.calls[2]["kwargs"] == {
        "JetPt": jet_pt,
        "JetEta": jet_eta,
        "Rho": rho,
        "systematic": "up",
    }


def test_root_scalefactors():
    extractor = lookup_tools.extractor()
    extractor.add_weight_sets(
        ["testSF2d scalefactors_Tight_Electron tests/samples/testSF2d.histo.root"]
    )

    extractor.finalize(reduce_list=["testSF2d"])

    evaluator = extractor.make_evaluator()

    counts, test_eta, test_pt = dummy_jagged_eta_pt()

    # test flat eval
    test_out = evaluator["testSF2d"](test_eta, test_pt)

    # print it
    print(evaluator["testSF2d"])

    # test structured eval
    test_eta_jagged = ak.unflatten(test_eta, counts)
    test_pt_jagged = ak.unflatten(test_pt, counts)
    test_out_jagged = evaluator["testSF2d"](test_eta_jagged, test_pt_jagged)

    assert ak.all(ak.num(test_out_jagged) == counts)
    assert ak.all(ak.flatten(test_out_jagged) == test_out)

    print(test_out)

    diff = np.abs(test_out - _testSF2d_expected_output)
    print("Max diff: %.16f" % diff.max())
    print("Median diff: %.16f" % np.median(diff))
    print(
        "Diff over threshold rate: %.1f %%" % (100 * (diff >= 1.0e-8).sum() / diff.size)
    )
    assert (diff < 1.0e-8).all()


def test_btag_csv_scalefactors():
    extractor = lookup_tools.extractor()
    extractor.add_weight_sets(
        [
            "testBTag * tests/samples/testBTagSF.btag.csv",
            "* * tests/samples/DeepCSV_102XSF_V1.btag.csv.gz",
            "* * tests/samples/DeepJet_102XSF_WP_V1.btag.csv.gz",
        ]
    )
    extractor.finalize()

    evaluator = extractor.make_evaluator()

    counts, test_eta, test_pt = dummy_jagged_eta_pt()
    # discriminant used for reshaping, zero otherwise
    test_discr = np.zeros_like(test_eta)

    print(evaluator["testBTagCSVv2_1_comb_up_0"])

    print(evaluator["DeepCSV_1_comb_up_0"])

    print(evaluator["btagsf_1_comb_up_0"])

    sf_out = evaluator["testBTagCSVv2_1_comb_up_0"](test_eta, test_pt, test_discr)
    print(sf_out)


def test_histo_json_scalefactors():
    extractor = lookup_tools.extractor()
    extractor.add_weight_sets(["testJson * tests/samples/EIDISO_WH_out.histo.json"])
    extractor.finalize()

    evaluator = extractor.make_evaluator()

    counts, test_eta, test_pt = dummy_jagged_eta_pt()

    sf_out = evaluator["testJsonEIDISO_WH/eta_pt_ratio_value"](test_eta, test_pt)
    sf_err_out = evaluator["testJsonEIDISO_WH/eta_pt_ratio_error"](test_eta, test_pt)
    print(sf_out)
    print(sf_err_out)


def test_jec_txt_scalefactors():
    extractor = lookup_tools.extractor()
    extractor.add_weight_sets(
        [
            "testJEC * tests/samples/Fall17_17Nov2017_V32_MC_L2Relative_AK4PFPuppi.jec.txt",
            "* * tests/samples/Summer16_07Aug2017_V11_L1fix_MC_L2Relative_AK4PFchs.jec.txt.gz",
            "* * tests/samples/Fall17_17Nov2017_V32_MC_Uncertainty_AK4PFPuppi.junc.txt",
            "* * tests/samples/Autumn18_V8_MC_UncertaintySources_AK4PFchs.junc.txt",
            "* * tests/samples/Spring16_25nsV10_MC_SF_AK4PFPuppi.jersf.txt",
            "* * tests/samples/Autumn18_V7b_MC_SF_AK8PFchs.jersf.txt.gz",
            "* * tests/samples/Fall17_17Nov2017_V32_MC_L2Relative_AK4Calo.jec.txt.gz",
            "* * tests/samples/Fall17_17Nov2017_V32_MC_L1JPTOffset_AK4JPT.jec.txt.gz",
            "* * tests/samples/Fall17_17Nov2017B_V32_DATA_L2Relative_AK4Calo.txt.gz",
            "* * tests/samples/Autumn18_V7b_DATA_SF_AK4PF.jersf.txt",
            "* * tests/samples/Autumn18_RunC_V19_DATA_L2Relative_AK8PFchs.jec.txt.gz",
            "* * tests/samples/Autumn18_RunA_V19_DATA_L2Relative_AK4Calo.jec.txt",
            "* * tests/samples/Winter14_V8_MC_L5Flavor_AK5Calo.txt",
        ]
    )
    extractor.finalize()

    evaluator = extractor.make_evaluator()

    counts, test_eta, test_pt = dummy_jagged_eta_pt()

    # test structured eval
    test_eta_jagged = ak.unflatten(test_eta, counts)
    test_pt_jagged = ak.unflatten(test_pt, counts)

    jec_out = evaluator["testJECFall17_17Nov2017_V32_MC_L2Relative_AK4PFPuppi"](
        test_eta, test_pt
    )
    jec_out_jagged = evaluator["testJECFall17_17Nov2017_V32_MC_L2Relative_AK4PFPuppi"](
        test_eta_jagged, test_pt_jagged
    )

    print(evaluator["testJECFall17_17Nov2017_V32_MC_L2Relative_AK4PFPuppi"])

    jec_out = evaluator["Summer16_07Aug2017_V11_L1fix_MC_L2Relative_AK4PFchs"](
        test_eta, test_pt
    )
    jec_out_jagged = evaluator["Summer16_07Aug2017_V11_L1fix_MC_L2Relative_AK4PFchs"](
        test_eta_jagged, test_pt_jagged
    )
    print(jec_out)
    print(jec_out_jagged)
    print(evaluator["Summer16_07Aug2017_V11_L1fix_MC_L2Relative_AK4PFchs"])

    jersf_out = evaluator["Spring16_25nsV10_MC_SF_AK4PFPuppi"](test_eta, test_pt)
    jersf_out_jagged = evaluator["Spring16_25nsV10_MC_SF_AK4PFPuppi"](
        test_eta_jagged, test_pt_jagged
    )
    print(jersf_out)
    print(jersf_out_jagged)

    # single jet jersf lookup test:
    single_jersf_out_1d = evaluator["Spring16_25nsV10_MC_SF_AK4PFPuppi"](
        np.array([1.4]), np.array([44.0])
    )
    single_jersf_out_0d = evaluator["Spring16_25nsV10_MC_SF_AK4PFPuppi"](
        np.array(1.4), np.array(44.0)
    )
    truth_out = np.array([[1.084, 1.095, 1.073]], dtype=np.float32)
    assert np.all(single_jersf_out_1d == truth_out)
    assert np.all(single_jersf_out_0d == truth_out)

    print(evaluator["Spring16_25nsV10_MC_SF_AK4PFPuppi"])

    junc_out = evaluator["Fall17_17Nov2017_V32_MC_Uncertainty_AK4PFPuppi"](
        test_eta, test_pt
    )
    junc_out_jagged = evaluator["Fall17_17Nov2017_V32_MC_Uncertainty_AK4PFPuppi"](
        test_eta_jagged, test_pt_jagged
    )
    print(junc_out)
    print(junc_out_jagged)
    print(evaluator["Fall17_17Nov2017_V32_MC_Uncertainty_AK4PFPuppi"])

    assert (
        "Autumn18_V8_MC_UncertaintySources_AK4PFchs_AbsoluteScale" in evaluator.keys()
    )
    junc_out = evaluator["Autumn18_V8_MC_UncertaintySources_AK4PFchs_AbsoluteScale"](
        test_eta, test_pt
    )
    junc_out_jagged = evaluator[
        "Autumn18_V8_MC_UncertaintySources_AK4PFchs_AbsoluteScale"
    ](test_eta_jagged, test_pt_jagged)
    print(junc_out)
    print(junc_out_jagged)
    print(evaluator["Autumn18_V8_MC_UncertaintySources_AK4PFchs_AbsoluteScale"])


def test_jec_txt_effareas():
    extractor = lookup_tools.extractor()
    extractor.add_weight_sets(["* * tests/samples/photon_id.ea.txt"])
    extractor.finalize()

    evaluator = extractor.make_evaluator()

    counts, test_eta, test_pt = dummy_jagged_eta_pt()

    ch_out = evaluator["photon_id_EA_CHad"](test_eta)
    print(ch_out)
    print(evaluator["photon_id_EA_CHad"])

    nh_out = evaluator["photon_id_EA_NHad"](test_eta)
    print(nh_out)
    print(evaluator["photon_id_EA_NHad"])

    ph_out = evaluator["photon_id_EA_Pho"](test_eta)
    print(ph_out)
    print(evaluator["photon_id_EA_Pho"])


def test_rochester():
    rochester_data = lookup_tools.txt_converters.convert_rochester_file(
        "tests/samples/RoccoR2018.txt.gz", loaduncs=True
    )
    rochester = lookup_tools.rochester_lookup.rochester_lookup(rochester_data)

    # to test 1-to-1 agreement with official Rochester requires loading C++ files
    # instead, preload the correct scales in the sample directory
    # the script tests/samples/rochester/build_rochester.py produces these
    official_data_k = np.load("tests/samples/nano_dimuon_rochester.npy")
    official_data_err = np.load("tests/samples/nano_dimuon_rochester_err.npy")
    official_mc_k = np.load("tests/samples/nano_dy_rochester.npy")
    official_mc_err = np.load("tests/samples/nano_dy_rochester_err.npy")
    mc_rand = np.load("tests/samples/nano_dy_rochester_rand.npy")

    # test against nanoaod
    events = NanoEventsFactory.from_root(
        os.path.abspath("tests/samples/nano_dimuon.root")
    ).events()

    data_k = rochester.kScaleDT(
        events.Muon.charge, events.Muon.pt, events.Muon.eta, events.Muon.phi
    )
    data_k = np.array(ak.flatten(data_k))
    assert all(np.isclose(data_k, official_data_k))
    data_err = rochester.kScaleDTerror(
        events.Muon.charge, events.Muon.pt, events.Muon.eta, events.Muon.phi
    )
    data_err = np.array(ak.flatten(data_err), dtype=float)
    assert all(np.isclose(data_err, official_data_err, atol=1e-8))

    # test against mc
    events = NanoEventsFactory.from_root(
        os.path.abspath("tests/samples/nano_dy.root")
    ).events()

    hasgen = ~np.isnan(ak.fill_none(events.Muon.matched_gen.pt, np.nan))
    mc_rand = ak.unflatten(mc_rand, ak.num(hasgen))
    mc_kspread = rochester.kSpreadMC(
        events.Muon.charge[hasgen],
        events.Muon.pt[hasgen],
        events.Muon.eta[hasgen],
        events.Muon.phi[hasgen],
        events.Muon.matched_gen.pt[hasgen],
    )
    mc_ksmear = rochester.kSmearMC(
        events.Muon.charge[~hasgen],
        events.Muon.pt[~hasgen],
        events.Muon.eta[~hasgen],
        events.Muon.phi[~hasgen],
        events.Muon.nTrackerLayers[~hasgen],
        mc_rand[~hasgen],
    )
    mc_k = np.array(ak.flatten(ak.ones_like(events.Muon.pt)))
    hasgen_flat = np.array(ak.flatten(hasgen))
    mc_k[hasgen_flat] = np.array(ak.flatten(mc_kspread))
    mc_k[~hasgen_flat] = np.array(ak.flatten(mc_ksmear))
    assert all(np.isclose(mc_k, official_mc_k))

    mc_errspread = rochester.kSpreadMCerror(
        events.Muon.charge[hasgen],
        events.Muon.pt[hasgen],
        events.Muon.eta[hasgen],
        events.Muon.phi[hasgen],
        events.Muon.matched_gen.pt[hasgen],
    )
    mc_errsmear = rochester.kSmearMCerror(
        events.Muon.charge[~hasgen],
        events.Muon.pt[~hasgen],
        events.Muon.eta[~hasgen],
        events.Muon.phi[~hasgen],
        events.Muon.nTrackerLayers[~hasgen],
        mc_rand[~hasgen],
    )
    mc_err = np.array(ak.flatten(ak.ones_like(events.Muon.pt)))
    mc_err[hasgen_flat] = np.array(ak.flatten(mc_errspread))
    mc_err[~hasgen_flat] = np.array(ak.flatten(mc_errsmear))
    assert all(np.isclose(mc_err, official_mc_err, atol=1e-8))


def test_dense_lookup():
    from coffea.lookup_tools.dense_lookup import dense_lookup
    import numpy

    a = ak.Array([[0.1, 0.2], [0.3]])
    lookup = dense_lookup(
        numpy.ones(shape=(3, 4)), (numpy.linspace(0, 1, 4), numpy.linspace(0, 1, 5))
    )

    with pytest.raises(ValueError):
        lookup(ak.Array([]))

    assert ak.to_list(lookup(ak.Array([]), ak.Array([]))) == []
    assert lookup(0.1, 0.3) == 1.0
    assert numpy.all(lookup(0.1, numpy.array([0.3, 0.5])) == numpy.array([1.0, 1.0]))
    assert ak.to_list(lookup(a, a)) == [[1.0, 1.0], [1.0]]


def test_549():
    import awkward as ak
    from coffea.lookup_tools import extractor

    ext = extractor()
    f_in = "tests/samples/SFttbar_2016_ele_pt.root"
    ext.add_weight_sets(["ele_pt histo_eff_data %s" % f_in])
    ext.finalize()
    evaluator = ext.make_evaluator()

    evaluator["ele_pt"](
        ak.Array([[45]]),
    )


def test_554():
    import uproot
    from coffea.lookup_tools.root_converters import convert_histo_root_file

    f_in = "tests/samples/PR554_SkipReadOnlyDirectory.root"
    rf = uproot.open(f_in)

    # check that input file contains uproot.ReadOnlyDirectory
    assert any(isinstance(v, uproot.ReadOnlyDirectory) for v in rf.values())
    # check that we can do the conversion now and get histograms out of uproot.ReadOnlyDirectories
    out = convert_histo_root_file(f_in)
    assert out
    # check that output does not contain any Directory-like keys
    rfkeys = set(k.rsplit(";")[0] for k in rf.keys())
    assert all(
        not isinstance(rf[k], uproot.ReadOnlyDirectory)
        for k, _ in out.keys()
        if k in rfkeys
    )
