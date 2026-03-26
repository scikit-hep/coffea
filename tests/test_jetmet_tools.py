import time

import awkward as ak
import dask
import dask_awkward as dak
import pyinstrument
import pytest
from dummy_distributions import dummy_jagged_eta_pt

from coffea.util import numpy as np


def jetmet_evaluator():
    from coffea.lookup_tools import extractor

    extract = extractor()

    extract.add_weight_sets(
        [
            "* * tests/samples/Summer16_23Sep2016V3_MC_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "* * tests/samples/Summer16_23Sep2016V3_MC_L2L3Residual_AK4PFPuppi.jec.txt.gz",
            "* * tests/samples/Summer16_23Sep2016V3_MC_L2Relative_AK4PFPuppi.jec.txt.gz",
            "* * tests/samples/Summer16_23Sep2016V3_MC_L3Absolute_AK4PFPuppi.jec.txt.gz",
            "* * tests/samples/Summer16_23Sep2016V3_MC_UncertaintySources_AK4PFPuppi.junc.txt.gz",
            "* * tests/samples/Summer16_23Sep2016V3_MC_Uncertainty_AK4PFPuppi.junc.txt.gz",
            "* * tests/samples/Fall17_17Nov2017_V6_MC_UncertaintySources_AK4PFchs.junc.txt.gz",
            "* * tests/samples/RegroupedV2_Fall17_17Nov2017_V32_MC_UncertaintySources_AK4PFchs.junc.txt.gz",
            "* * tests/samples/Regrouped_Fall17_17Nov2017_V32_MC_UncertaintySources_AK4PFchs.junc.txt",
            "* * tests/samples/Spring16_25nsV10_MC_PtResolution_AK4PFPuppi.jr.txt.gz",
            "* * tests/samples/Summer23Prompt23_RunCv1234_JRV1_MC_PtResolution_AK4PFPuppi.jr.txt.gz",
            "* * tests/samples/Spring16_25nsV10_MC_SF_AK4PFPuppi.jersf.txt.gz",
            "* * tests/samples/Autumn18_V7_MC_SF_AK4PFchs.jersf.txt.gz",
        ]
    )

    extract.finalize()

    return extract.make_evaluator()


evaluator = jetmet_evaluator()


@pytest.mark.parametrize("optimization_enabled", [True, False])
def test_factorized_jet_corrector(optimization_enabled):
    from coffea.jetmet_tools import FactorizedJetCorrector

    with dask.config.set({"awkward.optimization.enabled": optimization_enabled}):
        counts, test_eta, test_pt = dummy_jagged_eta_pt()
        test_Rho = np.full_like(test_eta, 100.0)
        test_A = np.full_like(test_eta, 5.0)

        # Check that the FactorizedJetCorrector is functional
        jec_names = [
            "Summer16_23Sep2016V3_MC_L1FastJet_AK4PFPuppi",
            "Summer16_23Sep2016V3_MC_L2Relative_AK4PFPuppi",
            "Summer16_23Sep2016V3_MC_L2L3Residual_AK4PFPuppi",
            "Summer16_23Sep2016V3_MC_L3Absolute_AK4PFPuppi",
        ]
        corrector = FactorizedJetCorrector(
            **{name: evaluator[name] for name in jec_names}
        )

        print(corrector)

        pt_copy = np.copy(test_pt)

        # Check that the corrector can be evaluated for flattened arrays
        corrs = corrector.getCorrection(
            JetEta=test_eta, Rho=test_Rho, JetPt=test_pt, JetA=test_A
        )

        assert (np.abs(pt_copy - test_pt) < 1e-6).all()

        test_pt_jag = ak.unflatten(test_pt, counts)
        test_eta_jag = ak.unflatten(test_eta, counts)
        test_Rho_jag = ak.unflatten(test_Rho, counts)
        test_A_jag = ak.unflatten(test_A, counts)

        test_pt_dak = dak.from_awkward(test_pt_jag, 1)
        test_eta_dak = dak.from_awkward(test_eta_jag, 1)
        test_Rho_dak = dak.from_awkward(test_Rho_jag, 1)
        test_A_dak = dak.from_awkward(test_A_jag, 1)

        # Check that the corrector can be evaluated for jagged arrays
        corrs_jag = corrector.getCorrection(
            JetEta=test_eta_jag,
            Rho=test_Rho_jag,
            JetPt=test_pt_jag,
            JetA=test_A_jag,
        )

        print(corrs_jag)

        corrs_dak = corrector.getCorrection(
            JetEta=test_eta_dak,
            Rho=test_Rho_dak,
            JetPt=test_pt_dak,
            JetA=test_A_dak,
        )

        print(corrs_dak)
        print(corrs_dak.dask)

        assert ak.all(np.abs(pt_copy - ak.flatten(test_pt_jag)) < 1e-6)
        assert ak.all(np.abs(corrs - ak.flatten(corrs_jag)) < 1e-6)
        assert ak.all(np.abs(corrs - ak.flatten(corrs_dak.compute())) < 1e-6)

        # Check that the corrector returns the correct answers for each level of correction
        # Use a subset of the values so that we can check the corrections by hand
        test_pt_jag = test_pt_jag[0:3]
        test_eta_jag = test_eta_jag[0:3]
        test_Rho_jag = test_Rho_jag[0:3]
        test_A_jag = test_A_jag[0:3]
        counts = counts[0:3]
        test_pt_dak = test_pt_dak[0:3]
        test_eta_dak = test_eta_dak[0:3]
        test_Rho_dak = test_Rho_dak[0:3]
        test_A_dak = test_A_dak[0:3]
        print("Raw jet values:")
        print("pT:", test_pt_jag)
        print("eta:", test_eta_jag)
        print("rho:", test_Rho_jag)
        print("area:", test_A_jag, "\n")

        # Start by checking the L1 corrections
        corrs_L1_jag_ref = ak.full_like(test_pt_jag, 1.0)
        corrector = FactorizedJetCorrector(
            **{name: evaluator[name] for name in jec_names[0:1]}
        )
        corrs_L1_jag = corrector.getCorrection(
            JetEta=test_eta_jag, Rho=test_Rho_jag, JetPt=test_pt_jag, JetA=test_A_jag
        )

        corrs_L1_dak = corrector.getCorrection(
            JetEta=test_eta_dak, Rho=test_Rho_dak, JetPt=test_pt_dak, JetA=test_A_dak
        )

        print(corrs_L1_dak)
        print(corrs_L1_dak.dask)

        print("Reference L1 corrections:", corrs_L1_jag_ref)
        print("Calculated L1 corrections:", corrs_L1_jag)
        assert ak.all(
            np.abs(ak.flatten(corrs_L1_jag_ref) - ak.flatten(corrs_L1_jag)) < 1e-6
        )
        assert ak.all(
            np.abs(ak.flatten(corrs_L1_jag_ref) - ak.flatten(corrs_L1_dak.compute()))
            < 1e-6
        )

        # Apply the L1 corrections and save the result
        test_ptL1_jag = test_pt_jag * corrs_L1_jag
        print("L1 corrected pT values:", test_ptL1_jag, "\n")
        assert ak.all(
            np.abs(ak.flatten(test_pt_jag) - ak.flatten(test_ptL1_jag)) < 1e-6
        )

        test_ptL1_dak = test_pt_dak * corrs_L1_dak
        print(test_ptL1_dak)
        assert ak.all(
            np.abs(ak.flatten(test_pt_jag) - ak.flatten(test_ptL1_dak.compute())) < 1e-6
        )

        # Check the L2 corrections on a subset of jets
        # Look up the parameters for the L2 corrections by hand and calculate the corrections
        # [(1.37906,35.8534,-0.00829227,7.96644e-05,5.18988e-06),
        #  (1.38034,17.9841,-0.00729638,-0.000127141,5.70889e-05),
        #  (1.74466,18.6372,-0.0367036,0.00310864,-0.000277062),
        #  (1.4759,24.8882,-0.0155333,0.0020836,-0.000198039),
        #  (1.14606,36.4215,-0.00174801,-1.76393e-05,1.91863e-06),
        #  (0.999657,4.02981,1.06597,-0.619679,-0.0494)],
        # [(1.54524,23.9023,-0.0162807,0.000665243,-4.66608e-06),
        #  (1.48431,8.68725,0.00642424,0.0252104,-0.0335696)]])
        corrs_L2_jag_ref = ak.unflatten(
            np.array(
                [
                    1.37038741364,
                    1.37710384514,
                    1.65148641108,
                    1.46840446827,
                    1.1328319784,
                    1.0,
                    1.50762056349,
                    1.48719866989,
                ]
            ),
            counts,
        )
        corrector = FactorizedJetCorrector(
            **{name: evaluator[name] for name in jec_names[1:2]}
        )
        corrs_L2_jag = corrector.getCorrection(
            JetEta=test_eta_jag, JetPt=corrs_L1_jag * test_pt_jag
        )
        corrs_L2_dak = corrector.getCorrection(
            JetEta=test_eta_dak, JetPt=corrs_L1_dak * test_pt_dak
        )
        print("Reference L2 corrections:", corrs_L2_jag_ref.tolist())
        print("Calculated L2 corrections:", corrs_L2_jag.tolist())
        assert ak.all(
            np.abs(ak.flatten(corrs_L2_jag_ref) - ak.flatten(corrs_L2_jag)) < 1e-6
        )
        assert ak.all(
            np.abs(ak.flatten(corrs_L2_jag_ref) - ak.flatten(corrs_L2_dak.compute()))
            < 1e-6
        )

        # Apply the L2 corrections and save the result
        test_ptL1L2_jag = test_ptL1_jag * corrs_L2_jag
        print("L1L2 corrected pT values:", test_ptL1L2_jag, "\n")
        test_ptL1L2_dak = test_ptL1_dak * corrs_L2_dak
        print("L1L2 corrected pT values:", test_ptL1L2_dak.compute(), "\n")
        print(test_ptL1L2_dak)
        print(test_ptL1L2_dak.dask)

        # Apply the L3 corrections and save the result
        corrs_L3_jag = ak.full_like(test_pt_jag, 1.0)
        test_ptL1L2L3_jag = test_ptL1L2_jag * corrs_L3_jag
        print("L1L2L3 corrected pT values:", test_ptL1L2L3_jag, "\n")

        corrs_L3_dak = dak.ones_like(test_pt_dak)
        test_ptL1L2L3_dak = test_ptL1L2_dak * corrs_L3_dak
        print("L1L2L3 corrected pT values:", test_ptL1L2L3_dak.compute(), "\n")
        print(test_ptL1L2L3_dak)
        print(test_ptL1L2L3_dak.dask)

        # Check that the corrections can be chained together
        corrs_L1L2L3_jag_ref = ak.unflatten(
            np.array(
                [
                    1.37038741364,
                    1.37710384514,
                    1.65148641108,
                    1.46840446827,
                    1.1328319784,
                    1.0,
                    1.50762056349,
                    1.48719866989,
                ]
            ),
            counts,
        )
        corrector = FactorizedJetCorrector(
            **{name: evaluator[name] for name in (jec_names[0:2] + jec_names[3:])}
        )
        corrs_L1L2L3_jag = corrector.getCorrection(
            JetEta=test_eta_jag, Rho=test_Rho_jag, JetPt=test_pt_jag, JetA=test_A_jag
        )

        corrs_L1L2L3_dak = corrector.getCorrection(
            JetEta=test_eta_dak, Rho=test_Rho_dak, JetPt=test_pt_dak, JetA=test_A_dak
        )

        print("Reference L1L2L3 corrections:", corrs_L1L2L3_jag_ref)
        print("Calculated L1L2L3 corrections:", corrs_L1L2L3_jag)
        print("Calculated L1L2L3 corrections:", corrs_L1L2L3_dak.compute())
        assert ak.all(
            np.abs(ak.flatten(corrs_L1L2L3_jag_ref) - ak.flatten(corrs_L1L2L3_jag))
            < 1e-6
        )
        assert ak.all(
            np.abs(
                ak.flatten(corrs_L1L2L3_jag_ref)
                - ak.flatten(corrs_L1L2L3_dak.compute())
            )
            < 1e-6
        )
        print(corrs_L1L2L3_dak)
        print(corrs_L1L2L3_dak.dask)

        # Apply the L1L2L3 corrections and save the result
        test_ptL1L2L3chain_jag = test_pt_jag * corrs_L1L2L3_jag
        print("Chained L1L2L3 corrected pT values:", test_ptL1L2L3chain_jag, "\n")
        assert ak.all(
            np.abs(ak.flatten(test_ptL1L2L3_jag) - ak.flatten(test_ptL1L2L3chain_jag))
            < 1e-6
        )

        test_ptL1L2L3chain_dak = test_pt_dak * corrs_L1L2L3_dak
        print(
            "Chained L1L2L3 corrected pT values:",
            test_ptL1L2L3chain_dak.compute(),
            "\n",
        )
        assert ak.all(
            np.abs(
                ak.flatten(test_ptL1L2L3_jag)
                - ak.flatten(test_ptL1L2L3chain_dak.compute())
            )
            < 1e-6
        )
        print(test_ptL1L2L3chain_dak)
        print(test_ptL1L2L3chain_dak.dask)


@pytest.mark.parametrize("optimization_enabled", [True, False])
def test_jet_resolution(optimization_enabled):
    from coffea.jetmet_tools import JetResolution

    with dask.config.set({"awkward.optimization.enabled": optimization_enabled}):
        counts, test_eta, test_pt = dummy_jagged_eta_pt()
        test_Rho = np.full_like(test_eta, 10.0)

        test_pt_jag = ak.unflatten(test_pt, counts)
        test_eta_jag = ak.unflatten(test_eta, counts)
        test_Rho_jag = ak.unflatten(test_Rho, counts)

        test_pt_dak = dak.from_awkward(test_pt_jag, 1)
        test_eta_dak = dak.from_awkward(test_eta_jag, 1)
        test_Rho_dak = dak.from_awkward(test_Rho_jag, 1)

        jer_names = ["Spring16_25nsV10_MC_PtResolution_AK4PFPuppi"]
        reso = JetResolution(**{name: evaluator[name] for name in jer_names})

        check2023_names = ["Summer23Prompt23_RunCv1234_JRV1_MC_PtResolution_AK4PFPuppi"]
        _ = JetResolution(**{name: evaluator[name] for name in check2023_names})

        print(reso)

        resos = reso.getResolution(JetEta=test_eta, Rho=test_Rho, JetPt=test_pt)
        resos_jag = reso.getResolution(
            JetEta=test_eta_jag, Rho=test_Rho_jag, JetPt=test_pt_jag
        )
        resos_dak = reso.getResolution(
            JetEta=test_eta_dak, Rho=test_Rho_dak, JetPt=test_pt_dak
        )
        assert ak.all(np.abs(resos - ak.flatten(resos_jag)) < 1e-6)
        assert ak.all(np.abs(resos - ak.flatten(resos_dak.compute())) < 1e-6)
        print(resos_dak)
        print(resos_dak.dask)

        test_pt_jag = test_pt_jag[0:3]
        test_eta_jag = test_eta_jag[0:3]
        test_Rho_jag = test_Rho_jag[0:3]
        test_Rho_jag = ak.concatenate(
            [test_Rho_jag[:-1], [ak.concatenate([test_Rho_jag[-1, :-1], 100.0])]]
        )
        counts = counts[0:3]
        print("Raw jet values:")
        print("pT:", test_pt_jag)
        print("eta:", test_eta_jag)
        print("rho:", test_Rho_jag, "\n")

        resos_jag_ref = ak.unflatten(
            np.array(
                [
                    0.21974642,
                    0.32421591,
                    0.33702479,
                    0.27420327,
                    0.13940689,
                    0.48134521,
                    0.26564994,
                    1.0,
                ]
            ),
            counts,
        )
        resos_jag = reso.getResolution(
            JetEta=test_eta_jag, Rho=test_Rho_jag, JetPt=test_pt_jag
        )
        print("Reference Resolution (jagged):", resos_jag_ref)
        print("Resolution (jagged):", resos_jag)
        # NB: 5e-4 tolerance was agreed upon by lgray and aperloff, if the differences get bigger over time
        #     we need to agree upon how these numbers are evaluated (double/float conversion is kinda random)
        assert ak.all(np.abs(ak.flatten(resos_jag_ref) - ak.flatten(resos_jag)) < 5e-4)


@pytest.mark.parametrize("optimization_enabled", [True, False])
def test_jet_correction_uncertainty(optimization_enabled):
    from coffea.jetmet_tools import JetCorrectionUncertainty

    with dask.config.set({"awkward.optimization.enabled": optimization_enabled}):
        counts, test_eta, test_pt = dummy_jagged_eta_pt()

        test_pt_jag = ak.unflatten(test_pt, counts)
        test_eta_jag = ak.unflatten(test_eta, counts)

        test_pt_dak = dak.from_awkward(test_pt_jag, 1)
        test_eta_dak = dak.from_awkward(test_eta_jag, 1)

        junc_names = ["Summer16_23Sep2016V3_MC_Uncertainty_AK4PFPuppi"]
        junc = JetCorrectionUncertainty(
            **{name: evaluator[name] for name in junc_names}
        )

        print(junc)

        juncs = junc.getUncertainty(JetEta=test_eta, JetPt=test_pt)

        juncs_jag = list(junc.getUncertainty(JetEta=test_eta_jag, JetPt=test_pt_jag))

        juncs_dak = list(junc.getUncertainty(JetEta=test_eta_dak, JetPt=test_pt_dak))

        for i, (level, corrs) in enumerate(juncs):
            assert corrs.shape[0] == test_eta.shape[0]
            assert ak.all(corrs == ak.flatten(juncs_jag[i][1]))
            assert ak.all(corrs == ak.flatten(juncs_dak[i][1].compute()))

        zipped_dak = dak.zip({k: v for k, v in juncs_dak})
        print(zipped_dak)
        print(zipped_dak.dask)

        test_pt_jag = test_pt_jag[0:3]
        test_eta_jag = test_eta_jag[0:3]
        counts = counts[0:3]
        print("Raw jet values:")
        print("pT:", test_pt_jag.tolist())
        print("eta:", test_eta_jag.tolist(), "\n")

        juncs_jag_ref = ak.unflatten(
            np.array(
                [
                    [1.053504214, 0.946495786],
                    [1.033343349, 0.966656651],
                    [1.065159157, 0.934840843],
                    [1.033140127, 0.966859873],
                    [1.016858652, 0.983141348],
                    [1.130199999, 0.869800001],
                    [1.039968468, 0.960031532],
                    [1.033100002, 0.966899998],
                ]
            ),
            counts,
        )
        juncs_jag = list(junc.getUncertainty(JetEta=test_eta_jag, JetPt=test_pt_jag))

        for i, (level, corrs) in enumerate(juncs_jag):
            print("Index:", i)
            print("Correction level:", level)
            print("Reference Uncertainties (jagged):", juncs_jag_ref)
            print("Uncertainties (jagged):", corrs)
            assert ak.all(np.abs(ak.flatten(juncs_jag_ref) - ak.flatten(corrs)) < 1e-6)


@pytest.mark.parametrize("optimization_enabled", [True, False])
def test_jet_correction_uncertainty_sources(optimization_enabled):
    from coffea.jetmet_tools import JetCorrectionUncertainty

    with dask.config.set({"awkward.optimization.enabled": optimization_enabled}):
        counts, test_eta, test_pt = dummy_jagged_eta_pt()

        test_pt_jag = ak.unflatten(test_pt, counts)
        test_eta_jag = ak.unflatten(test_eta, counts)

        test_pt_dak = dak.from_awkward(test_pt_jag, 1)
        test_eta_dak = dak.from_awkward(test_eta_jag, 1)

        junc_names = []
        levels = []
        for name in dir(evaluator):
            if "Summer16_23Sep2016V3_MC_UncertaintySources_AK4PFPuppi" in name:
                junc_names.append(name)
                levels.append(name.split("_")[-1])
            # test for underscore in dataera
            if (
                "Fall17_17Nov2017_V6_MC_UncertaintySources_AK4PFchs_AbsoluteFlavMap"
                in name
            ):
                junc_names.append(name)
                levels.append(name.split("_")[-1])
        junc = JetCorrectionUncertainty(
            **{name: evaluator[name] for name in junc_names}
        )

        print(junc)

        juncs = junc.getUncertainty(JetEta=test_eta, JetPt=test_pt)

        juncs_jag = list(junc.getUncertainty(JetEta=test_eta_jag, JetPt=test_pt_jag))

        juncs_dak = list(junc.getUncertainty(JetEta=test_eta_dak, JetPt=test_pt_dak))

        for i, (level, corrs) in enumerate(juncs):
            assert level in levels
            assert corrs.shape[0] == test_eta.shape[0]
            assert ak.all(corrs == ak.flatten(juncs_jag[i][1]))
            assert ak.all(corrs == ak.flatten(juncs_dak[i][1].compute()))

        zipped_dak = dak.zip({k: v for k, v in juncs_dak})
        print(zipped_dak)
        print(zipped_dak.dask)

        test_pt_jag = test_pt_jag[0:3]
        test_eta_jag = test_eta_jag[0:3]
        counts = counts[0:3]
        print("Raw jet values:")
        print("pT:", test_pt_jag.tolist())
        print("eta:", test_eta_jag.tolist(), "\n")

        juncs_jag_ref = ak.unflatten(
            np.array(
                [
                    [1.053504214, 0.946495786],
                    [1.033343349, 0.966656651],
                    [1.065159157, 0.934840843],
                    [1.033140127, 0.966859873],
                    [1.016858652, 0.983141348],
                    [1.130199999, 0.869800001],
                    [1.039968468, 0.960031532],
                    [1.033100002, 0.966899998],
                ]
            ),
            counts,
        )
        juncs_jag = list(junc.getUncertainty(JetEta=test_eta_jag, JetPt=test_pt_jag))
        for i, (level, corrs) in enumerate(juncs_jag):
            if level != "Total":
                continue
            print("Index:", i)
            print("Correction level:", level)
            print("Reference Uncertainties (jagged):", juncs_jag_ref)
            print("Uncertainties (jagged):", corrs, "\n")
            assert ak.all(np.abs(ak.flatten(juncs_jag_ref) - ak.flatten(corrs)) < 1e-6)


@pytest.mark.parametrize("optimization_enabled", [True, False])
def test_jet_correction_regrouped_uncertainty_sources(optimization_enabled):
    from coffea.jetmet_tools import JetCorrectionUncertainty

    with dask.config.set({"awkward.optimization.enabled": optimization_enabled}):
        counts, test_eta, test_pt = dummy_jagged_eta_pt()

        test_pt_jag = ak.unflatten(test_pt, counts)
        test_eta_jag = ak.unflatten(test_eta, counts)

        test_pt_dak = dak.from_awkward(test_pt_jag, 1)
        test_eta_dak = dak.from_awkward(test_eta_jag, 1)

        junc_names = []
        levels = []
        for name in dir(evaluator):
            if "Regrouped_Fall17_17Nov2017_V32_MC_UncertaintySources_AK4PFchs" in name:
                junc_names.append(name)
                if len(name.split("_")) == 9:
                    levels.append("_".join(name.split("_")[-2:]))
                else:
                    levels.append(name.split("_")[-1])
        junc = JetCorrectionUncertainty(
            **{name: evaluator[name] for name in junc_names}
        )

        print(junc)

        juncs_jag = list(junc.getUncertainty(JetEta=test_eta_jag, JetPt=test_pt_jag))

        juncs_dak = list(junc.getUncertainty(JetEta=test_eta_dak, JetPt=test_pt_dak))

        for i, tpl in enumerate(
            list(junc.getUncertainty(JetEta=test_eta, JetPt=test_pt))
        ):
            assert tpl[0] in levels
            assert tpl[1].shape[0] == test_eta.shape[0]
            assert ak.all(tpl[1] == ak.flatten(juncs_jag[i][1]))
            assert ak.all(tpl[1] == ak.flatten(juncs_dak[i][1].compute()))

        zipped_dak = dak.zip({k: v for k, v in juncs_dak})
        print(zipped_dak)
        print(zipped_dak.dask)

        test_pt_jag = test_pt_jag[0:3]
        test_eta_jag = test_eta_jag[0:3]
        counts = counts[0:3]
        print("Raw jet values:")
        print("pT:", test_pt_jag.tolist())
        print("eta:", test_eta_jag.tolist(), "\n")

        juncs_jag_ref = ak.unflatten(
            np.array(
                [
                    [1.119159088, 0.880840912],
                    [1.027003404, 0.972996596],
                    [1.135201275, 0.864798725],
                    [1.039665259, 0.960334741],
                    [1.015064503, 0.984935497],
                    [1.149900004, 0.850099996],
                    [1.079960600, 0.920039400],
                    [1.041200001, 0.958799999],
                ]
            ),
            counts,
        )
        juncs_jag = list(junc.getUncertainty(JetEta=test_eta_jag, JetPt=test_pt_jag))
        for i, (level, corrs) in enumerate(juncs_jag):
            if level != "Total":
                continue
            print("Index:", i)
            print("Correction level:", level)
            print("Reference Uncertainties (jagged):", juncs_jag_ref)
            print("Uncertainties (jagged):", corrs, "\n")
            assert ak.all(np.abs(ak.flatten(juncs_jag_ref) - ak.flatten(corrs)) < 1e-6)


@pytest.mark.parametrize("optimization_enabled", [True, False])
def test_jet_resolution_sf(optimization_enabled):
    from coffea.jetmet_tools import JetResolutionScaleFactor

    with dask.config.set({"awkward.optimization.enabled": optimization_enabled}):
        counts, test_eta, test_pt = dummy_jagged_eta_pt()

        test_pt_jag = ak.unflatten(test_pt, counts)
        test_eta_jag = ak.unflatten(test_eta, counts)

        test_eta_dak = dak.from_awkward(test_eta_jag, 1)

        jersf_names = ["Spring16_25nsV10_MC_SF_AK4PFPuppi"]
        resosf = JetResolutionScaleFactor(
            **{name: evaluator[name] for name in jersf_names}
        )

        print(resosf)

        # 0-jet compatibility
        assert resosf.getScaleFactor(JetEta=test_eta[:0]).shape == (0, 3)

        resosfs = resosf.getScaleFactor(JetEta=test_eta)
        resosfs_jag = resosf.getScaleFactor(JetEta=test_eta_jag)
        resosfs_dak = resosf.getScaleFactor(JetEta=test_eta_dak)
        assert ak.all(resosfs == ak.flatten(resosfs_jag))
        assert ak.all(resosfs == ak.flatten(resosfs_dak.compute()))
        print(resosfs_dak)
        print(resosfs_dak.dask)

        test_pt_jag = test_pt_jag[0:3]
        test_eta_jag = test_eta_jag[0:3]
        counts = counts[0:3]
        print("Raw jet values:")
        print("pT:", test_pt_jag)
        print("eta:", test_eta_jag, "\n")

        resosfs_jag_ref = ak.unflatten(
            np.array(
                [
                    [1.857, 1.928, 1.786],
                    [1.084, 1.095, 1.073],
                    [1.364, 1.403, 1.325],
                    [1.177, 1.218, 1.136],
                    [1.138, 1.151, 1.125],
                    [1.364, 1.403, 1.325],
                    [1.177, 1.218, 1.136],
                    [1.082, 1.117, 1.047],
                ]
            ),
            counts,
        )
        resosfs_jag = resosf.getScaleFactor(JetEta=test_eta_jag)
        print("Reference Resolution SF (jagged):", resosfs_jag_ref)
        print("Resolution SF (jagged):", resosfs_jag)
        assert ak.all(
            np.abs(ak.flatten(resosfs_jag_ref) - ak.flatten(resosfs_jag)) < 1e-6
        )


@pytest.mark.parametrize("optimization_enabled", [True, False])
def test_jet_resolution_sf_2d(optimization_enabled):
    from coffea.jetmet_tools import JetResolutionScaleFactor

    with dask.config.set({"awkward.optimization.enabled": optimization_enabled}):
        counts, test_eta, test_pt = dummy_jagged_eta_pt()

        test_pt_jag = ak.unflatten(test_pt, counts)
        test_eta_jag = ak.unflatten(test_eta, counts)

        test_pt_dak = dak.from_awkward(test_pt_jag, 1)
        test_eta_dak = dak.from_awkward(test_eta_jag, 1)

        resosf = JetResolutionScaleFactor(
            **{name: evaluator[name] for name in ["Autumn18_V7_MC_SF_AK4PFchs"]}
        )

        print(resosf)

        # 0-jet compatibility
        assert resosf.getScaleFactor(JetPt=test_pt[:0], JetEta=test_eta[:0]).shape == (
            0,
            3,
        )

        resosfs = resosf.getScaleFactor(JetPt=test_pt, JetEta=test_eta)
        resosfs_jag = resosf.getScaleFactor(JetPt=test_pt_jag, JetEta=test_eta_jag)
        resosfs_dak = resosf.getScaleFactor(JetPt=test_pt_dak, JetEta=test_eta_dak)
        assert ak.all(resosfs == ak.flatten(resosfs_jag))
        assert ak.all(resosfs == ak.flatten(resosfs_dak.compute()))
        print(resosfs_dak)
        print(resosfs_dak.dask)

        test_pt_jag = test_pt_jag[0:3]
        test_eta_jag = test_eta_jag[0:3]
        counts = counts[0:3]
        print("Raw jet values:")
        print("pT:", test_pt_jag)
        print("eta:", test_eta_jag, "\n")

        resosfs_jag_ref = ak.unflatten(
            np.array(
                [
                    [1.11904, 1.31904, 1.0],
                    [1.1432, 1.2093, 1.0771],
                    [1.16633, 1.36633, 1.0],
                    [1.17642, 1.37642, 1.0],
                    [1.1808, 1.1977, 1.1640],
                    [1.15965, 1.35965, 1.0],
                    [1.17661, 1.37661, 1.0],
                    [1.1175, 1.1571, 1.0778],
                ]
            ),
            counts,
        )
        resosfs_jag = resosf.getScaleFactor(JetPt=test_pt_jag, JetEta=test_eta_jag)
        print("Reference Resolution SF (jagged):", resosfs_jag_ref)
        print("Resolution SF (jagged):", resosfs_jag)
        assert ak.all(
            np.abs(ak.flatten(resosfs_jag_ref) - ak.flatten(resosfs_jag)) < 1e-6
        )


def test_corrected_jets_factory():
    import os

    from coffea.jetmet_tools import CorrectedJetsFactory, CorrectedMETFactory, JECStack
    from coffea.nanoevents import NanoEventsFactory

    events = NanoEventsFactory.from_root(
        {os.path.abspath("tests/samples/nano_dy.root"): "Events"},
        metadata={},
        mode="virtual",
    ).events()

    jec_stack_names = [
        "Summer16_23Sep2016V3_MC_L1FastJet_AK4PFPuppi",
        "Summer16_23Sep2016V3_MC_L2Relative_AK4PFPuppi",
        "Summer16_23Sep2016V3_MC_L2L3Residual_AK4PFPuppi",
        "Summer16_23Sep2016V3_MC_L3Absolute_AK4PFPuppi",
        "Spring16_25nsV10_MC_PtResolution_AK4PFPuppi",
        "Spring16_25nsV10_MC_SF_AK4PFPuppi",
    ]
    for key in evaluator.keys():
        if "Summer16_23Sep2016V3_MC_UncertaintySources_AK4PFPuppi" in key:
            jec_stack_names.append(key)

    jec_inputs = {name: evaluator[name] for name in jec_stack_names}
    jec_stack = JECStack(jec_inputs)

    name_map = jec_stack.blank_name_map
    name_map["JetPt"] = "pt"
    name_map["JetMass"] = "mass"
    name_map["JetEta"] = "eta"
    name_map["JetA"] = "area"

    jets = events.Jet

    jets["pt_raw"] = (1 - jets["rawFactor"]) * jets.pt
    jets["mass_raw"] = (1 - jets["rawFactor"]) * jets.mass
    jets["pt_gen"] = ak.fill_none(jets.matched_gen.pt, 0)
    jets["Rho"] = events.fixedGridRhoFastjetAll
    name_map["ptGenJet"] = "pt_gen"
    name_map["ptRaw"] = "pt_raw"
    name_map["massRaw"] = "mass_raw"
    name_map["Rho"] = "Rho"

    print(name_map)

    tic = time.time()
    jet_factory = CorrectedJetsFactory(name_map, jec_stack)
    toc = time.time()

    print("setup corrected jets time =", toc - tic)

    tic = time.time()
    prof = pyinstrument.Profiler()
    prof.start()
    corrected_jets = jet_factory.build(jets)
    prof.stop()
    toc = time.time()

    print("corrected_jets build time =", toc - tic)

    print(prof.output_text(unicode=True, color=True, show_all=True))

    print("Generated jet pt:", corrected_jets.pt_gen)
    print("Original jet pt:", corrected_jets.pt_orig)
    print("Raw jet pt:", jets.pt_raw)
    print("Corrected jet pt:", corrected_jets.pt)
    print("Original jet mass:", corrected_jets.mass_orig)
    print("Raw jet mass:", jets["mass_raw"])
    print("Corrected jet mass:", corrected_jets.mass)
    print("jet eta:", jets.eta)

    tic = time.time()
    prof = pyinstrument.Profiler()
    prof.start()

    computed_uncs = {
        unc: {"up": corrected_jets[unc].up.pt, "down": corrected_jets[unc].down.pt}
        for unc in jet_factory.uncertainties()
    }

    for unc in jet_factory.uncertainties():
        print(unc)
        print(computed_uncs[unc]["up"])
        print(computed_uncs[unc]["down"])
    prof.stop()
    toc = time.time()

    print(prof.output_text(unicode=True, color=True, show_all=True))

    print("build all jet variations =", toc - tic)

    # Test that the corrections were applied correctly
    from coffea.jetmet_tools import (
        FactorizedJetCorrector,
        JetResolution,
        JetResolutionScaleFactor,
    )

    corrector = FactorizedJetCorrector(
        **{name: evaluator[name] for name in jec_stack_names[0:4]}
    )

    check_corrs = corrector.getCorrection(
        JetEta=jets.eta,
        Rho=jets.Rho,
        JetPt=jets.pt_raw,
        JetA=jets.area,
    )
    reso = JetResolution(**{name: evaluator[name] for name in jec_stack_names[4:5]})
    check_resos = reso.getResolution(
        JetEta=jets.eta,
        Rho=jets.Rho,
        JetPt=jets.pt_raw,
    )
    resosf = JetResolutionScaleFactor(
        **{name: evaluator[name] for name in jec_stack_names[5:6]}
    )

    check_resosfs = resosf.getScaleFactor(
        JetEta=events.Jet.eta,
    )

    # Filter out the non-deterministic (no gen pt) jets
    def smear_factor(jetPt, pt_gen, jersf):
        return (
            ak.full_like(jetPt, 1.0)
            + (jersf[:, 0] - ak.full_like(jetPt, 1.0)) * (jetPt - pt_gen) / jetPt
        )

    test_gen_pt = ak.concatenate(
        [
            ak.fill_none(events.Jet.matched_gen.pt, 0)[0, :-2],
            ak.fill_none(events.Jet.matched_gen.pt, 0)[-1, :-1],
        ]
    )
    test_raw_pt = ak.concatenate(
        [
            ((1 - events.Jet.rawFactor) * events.Jet.pt)[0, :-2],
            ((1 - events.Jet.rawFactor) * events.Jet.pt)[-1, :-1],
        ]
    )
    test_pt = ak.concatenate([corrected_jets.pt[0, :-2], corrected_jets.pt[-1, :-1]])
    test_eta = ak.concatenate([events.Jet.eta[0, :-2], events.Jet.eta[-1, :-1]])
    test_jer = ak.concatenate([check_resos[0, :-2], check_resos[-1, :-1]])
    test_jer_sf = ak.concatenate(
        [
            check_resosfs[0, :-2],
            check_resosfs[-1, :-1],
        ]
    )
    test_jec = ak.concatenate([check_corrs[0, :-2], check_corrs[-1, :-1]])
    test_corrected_pt = ak.concatenate(
        [corrected_jets.pt[0, :-2], corrected_jets.pt[-1, :-1]]
    )
    test_corr_pt = test_raw_pt * test_jec
    test_pt_smear_corr = test_corr_pt * smear_factor(
        test_corr_pt, test_gen_pt, test_jer_sf
    )

    # Print the results of the "by-hand" calculations and confirm that the values match the expected values
    print("\nConfirm the CorrectedJetsFactory values:")
    print("Jet pt (gen)", test_gen_pt.tolist())
    print("Jet pt (raw)", test_raw_pt.tolist())
    print("Jet pt (nano):", test_pt.tolist())
    print("Jet eta:", test_eta.tolist())
    print("Jet energy resolution:", test_jer.tolist())
    print("Jet energy resolution sf:", test_jer_sf.tolist())
    print("Jet energy correction:", test_jec.tolist())
    print("Corrected jet pt (ref)", test_corr_pt.tolist())
    print("Corrected & smeared jet pt (ref):", test_pt_smear_corr.tolist())
    print("Corrected & smeared jet pt:", test_corrected_pt.tolist(), "\n")
    assert ak.all(np.abs(test_pt_smear_corr - test_corrected_pt) < 1e-6)

    name_map["METpt"] = "pt"
    name_map["METphi"] = "phi"
    name_map["JetPhi"] = "phi"
    name_map["UnClusteredEnergyDeltaX"] = "MetUnclustEnUpDeltaX"
    name_map["UnClusteredEnergyDeltaY"] = "MetUnclustEnUpDeltaY"
    assert name_map["JetPt"] + "_orig" in corrected_jets.fields
    assert name_map["JetPt"] + "_orig" in corrected_jets.JES_AbsoluteStat.up.fields
    assert name_map["JetPt"] + "_orig" in corrected_jets.JER.up.fields

    tic = time.time()
    met_factory = CorrectedMETFactory(name_map)
    toc = time.time()

    print("setup corrected MET time =", toc - tic)

    met = events.MET
    tic = time.time()
    # prof = pyinstrument.Profiler()
    # prof.start()
    corrected_met = met_factory.build(met, corrected_jets)
    # prof.stop()
    toc = time.time()

    # print(prof.output_text(unicode=True, color=True, show_all=True))

    print("corrected_met build time =", toc - tic)

    print(corrected_met.pt_orig)
    print(corrected_met.pt)
    tic = time.time()
    prof = pyinstrument.Profiler()
    prof.start()

    computed_uncs = {
        unc: {"up": corrected_met[unc].up.pt, "down": corrected_met[unc].down.pt}
        for unc in (jet_factory.uncertainties() + met_factory.uncertainties())
    }

    for unc in jet_factory.uncertainties() + met_factory.uncertainties():
        print(unc)
        print(computed_uncs[unc]["up"])
        print(computed_uncs[unc]["down"])
    prof.stop()
    toc = time.time()

    print("build all met variations =", toc - tic)

    print(prof.output_text(unicode=True, color=True, show_all=True))


@pytest.mark.dask_client
@pytest.mark.parametrize("optimization_enabled", [True, False])
def test_corrected_jets_factory_dak(optimization_enabled):
    import os

    from distributed import Client

    from coffea.jetmet_tools import CorrectedJetsFactory, CorrectedMETFactory, JECStack

    events = None
    from coffea.nanoevents import NanoEventsFactory

    with (
        Client(),
        dask.config.set({"awkward.optimization.enabled": optimization_enabled}),
    ):
        events = NanoEventsFactory.from_root(
            {os.path.abspath("tests/samples/nano_dy.root"): "Events"},
            metadata={},
            mode="dask",
        ).events()

        jec_stack_names = [
            "Summer16_23Sep2016V3_MC_L1FastJet_AK4PFPuppi",
            "Summer16_23Sep2016V3_MC_L2Relative_AK4PFPuppi",
            "Summer16_23Sep2016V3_MC_L2L3Residual_AK4PFPuppi",
            "Summer16_23Sep2016V3_MC_L3Absolute_AK4PFPuppi",
            "Spring16_25nsV10_MC_PtResolution_AK4PFPuppi",
            "Spring16_25nsV10_MC_SF_AK4PFPuppi",
        ]
        for key in evaluator.keys():
            if "Summer16_23Sep2016V3_MC_UncertaintySources_AK4PFPuppi" in key:
                jec_stack_names.append(key)

        jec_inputs = {name: evaluator[name] for name in jec_stack_names}
        jec_stack = JECStack(jec_inputs)

        name_map = jec_stack.blank_name_map
        name_map["JetPt"] = "pt"
        name_map["JetMass"] = "mass"
        name_map["JetEta"] = "eta"
        name_map["JetA"] = "area"

        jets = events.Jet

        jets["pt_raw"] = (1 - jets["rawFactor"]) * jets.pt
        jets["mass_raw"] = (1 - jets["rawFactor"]) * jets.mass
        jets["pt_gen"] = dak.fill_none(jets.matched_gen.pt, 0)
        jets["Rho"] = events.fixedGridRhoFastjetAll
        name_map["ptGenJet"] = "pt_gen"
        name_map["ptRaw"] = "pt_raw"
        name_map["massRaw"] = "mass_raw"
        name_map["Rho"] = "Rho"

        print(name_map)

        tic = time.time()
        jet_factory = CorrectedJetsFactory(name_map, jec_stack)
        toc = time.time()

        print("setup corrected jets time =", toc - tic)

        tic = time.time()
        prof = pyinstrument.Profiler()
        prof.start()
        corrected_jets = jet_factory.build(jets)
        prof.stop()
        toc = time.time()

        print("corrected_jets build time =", toc - tic)

        print(prof.output_text(unicode=True, color=True, show_all=True))

        print(corrected_jets.dask)

        print("Generated jet pt:", corrected_jets.pt_gen.compute())
        print("Original jet pt:", corrected_jets.pt_orig.compute())
        print("Raw jet pt:", jets.pt_raw)
        print("Corrected jet pt:", corrected_jets.pt.compute())
        print("Original jet mass:", corrected_jets.mass_orig.compute())
        print("Raw jet mass:", jets["mass_raw"])
        print("Corrected jet mass:", corrected_jets.mass.compute())
        print("jet eta:", jets.eta)

        tic = time.time()
        prof = pyinstrument.Profiler()
        prof.start()

        tocompute = {
            unc: {"up": corrected_jets[unc].up.pt, "down": corrected_jets[unc].down.pt}
            for unc in jet_factory.uncertainties()
        }
        computed_uncs = dask.compute(tocompute)[0]

        for unc in jet_factory.uncertainties():
            print(unc)
            print(computed_uncs[unc]["up"])
            print(computed_uncs[unc]["down"])
        prof.stop()
        toc = time.time()

        print(prof.output_text(unicode=True, color=True, show_all=True))

        print("build all jet variations =", toc - tic)

        # Test that the corrections were applied correctly
        from coffea.jetmet_tools import (
            FactorizedJetCorrector,
            JetResolution,
            JetResolutionScaleFactor,
        )

        corrector = FactorizedJetCorrector(
            **{name: evaluator[name] for name in jec_stack_names[0:4]}
        )

        check_corrs = corrector.getCorrection(
            JetEta=jets.eta,
            Rho=jets.Rho,
            JetPt=jets.pt_raw,
            JetA=jets.area,
        ).compute()
        reso = JetResolution(**{name: evaluator[name] for name in jec_stack_names[4:5]})
        check_resos = reso.getResolution(
            JetEta=jets.eta,
            Rho=jets.Rho,
            JetPt=jets.pt_raw,
        ).compute()
        resosf = JetResolutionScaleFactor(
            **{name: evaluator[name] for name in jec_stack_names[5:6]}
        )

        print(dak.report_necessary_columns(jets.eta))
        print(
            dak.report_necessary_columns(
                resosf.getScaleFactor(
                    JetEta=jets.eta,
                )
            )
        )

        check_resosfs = resosf.getScaleFactor(
            JetEta=events.Jet.eta,
        ).compute()

        # Filter out the non-deterministic (no gen pt) jets
        def smear_factor(jetPt, pt_gen, jersf):
            return (
                ak.full_like(jetPt, 1.0)
                + (jersf[:, 0] - ak.full_like(jetPt, 1.0)) * (jetPt - pt_gen) / jetPt
            )

        test_gen_pt = ak.concatenate(
            [
                dak.fill_none(events.Jet.matched_gen.pt, 0).compute()[0, :-2],
                dak.fill_none(events.Jet.matched_gen.pt, 0).compute()[-1, :-1],
            ]
        )
        test_raw_pt = ak.concatenate(
            [
                ((1 - events.Jet.rawFactor) * events.Jet.pt).compute()[0, :-2],
                ((1 - events.Jet.rawFactor) * events.Jet.pt).compute()[-1, :-1],
            ]
        )
        test_pt = ak.concatenate(
            [corrected_jets.pt.compute()[0, :-2], corrected_jets.pt.compute()[-1, :-1]]
        )
        test_eta = ak.concatenate(
            [events.Jet.eta.compute()[0, :-2], events.Jet.eta.compute()[-1, :-1]]
        )
        test_jer = ak.concatenate([check_resos[0, :-2], check_resos[-1, :-1]])
        test_jer_sf = ak.concatenate(
            [
                check_resosfs[0, :-2],
                check_resosfs[-1, :-1],
            ]
        )
        test_jec = ak.concatenate([check_corrs[0, :-2], check_corrs[-1, :-1]])
        test_corrected_pt = ak.concatenate(
            [corrected_jets.pt.compute()[0, :-2], corrected_jets.pt.compute()[-1, :-1]]
        )
        test_corr_pt = test_raw_pt * test_jec
        test_pt_smear_corr = test_corr_pt * smear_factor(
            test_corr_pt, test_gen_pt, test_jer_sf
        )

        # Print the results of the "by-hand" calculations and confirm that the values match the expected values
        print("\nConfirm the CorrectedJetsFactory values:")
        print("Jet pt (gen)", test_gen_pt.tolist())
        print("Jet pt (raw)", test_raw_pt.tolist())
        print("Jet pt (nano):", test_pt.tolist())
        print("Jet eta:", test_eta.tolist())
        print("Jet energy resolution:", test_jer.tolist())
        print("Jet energy resolution sf:", test_jer_sf.tolist())
        print("Jet energy correction:", test_jec.tolist())
        print("Corrected jet pt (ref)", test_corr_pt.tolist())
        print("Corrected & smeared jet pt (ref):", test_pt_smear_corr.tolist())
        print("Corrected & smeared jet pt:", test_corrected_pt.tolist(), "\n")
        assert ak.all(np.abs(test_pt_smear_corr - test_corrected_pt) < 1e-6)

        name_map["METpt"] = "pt"
        name_map["METphi"] = "phi"
        name_map["JetPhi"] = "phi"
        name_map["UnClusteredEnergyDeltaX"] = "MetUnclustEnUpDeltaX"
        name_map["UnClusteredEnergyDeltaY"] = "MetUnclustEnUpDeltaY"
        assert name_map["JetPt"] + "_orig" in corrected_jets.fields
        assert name_map["JetPt"] + "_orig" in corrected_jets.JES_AbsoluteStat.up.fields
        assert name_map["JetPt"] + "_orig" in corrected_jets.JER.up.fields

        tic = time.time()
        met_factory = CorrectedMETFactory(name_map)
        toc = time.time()

        print("setup corrected MET time =", toc - tic)

        met = events.MET
        tic = time.time()
        # prof = pyinstrument.Profiler()
        # prof.start()
        corrected_met = met_factory.build(met, corrected_jets)
        # prof.stop()
        toc = time.time()

        # print(prof.output_text(unicode=True, color=True, show_all=True))

        print("corrected_met build time =", toc - tic)

        print(corrected_met.dask)

        print(corrected_met.pt_orig.compute())
        print(corrected_met.pt.compute())
        tic = time.time()
        prof = pyinstrument.Profiler()
        prof.start()

        tocompute = {
            unc: {"up": corrected_met[unc].up.pt, "down": corrected_met[unc].down.pt}
            for unc in (jet_factory.uncertainties() + met_factory.uncertainties())
        }
        computed_uncs = dask.compute(tocompute)[0]

        for unc in jet_factory.uncertainties() + met_factory.uncertainties():
            print(unc)
            print(computed_uncs[unc]["up"])
            print(computed_uncs[unc]["down"])
        prof.stop()
        toc = time.time()

        print("build all met variations =", toc - tic)

        print(prof.output_text(unicode=True, color=True, show_all=True))


def test_corrected_met_type1():
    """Test CorrectedMETFactory in Type-1 mode with L1 and L1L2L3 JEC correctors."""
    import os

    from coffea.jetmet_tools import CorrectedJetsFactory, CorrectedMETFactory, JECStack
    from coffea.nanoevents import NanoEventsFactory

    events = NanoEventsFactory.from_root(
        {os.path.abspath("tests/samples/nano_tt_v15.root"): "Events"},
        metadata={},
        mode="eager",
    ).events()

    # --- Build JEC stack (same as existing test) ---
    jec_stack_names = [
        "Summer16_23Sep2016V3_MC_L1FastJet_AK4PFPuppi",
        "Summer16_23Sep2016V3_MC_L2Relative_AK4PFPuppi",
        "Summer16_23Sep2016V3_MC_L2L3Residual_AK4PFPuppi",
        "Summer16_23Sep2016V3_MC_L3Absolute_AK4PFPuppi",
        "Spring16_25nsV10_MC_PtResolution_AK4PFPuppi",
        "Spring16_25nsV10_MC_SF_AK4PFPuppi",
    ]
    for key in evaluator.keys():
        if "Summer16_23Sep2016V3_MC_UncertaintySources_AK4PFPuppi" in key:
            jec_stack_names.append(key)

    jec_inputs = {name: evaluator[name] for name in jec_stack_names}
    jec_stack = JECStack(jec_inputs)

    # --- Build name_map for CorrectedJetsFactory ---
    name_map = jec_stack.blank_name_map
    name_map["JetPt"] = "pt"
    name_map["JetMass"] = "mass"
    name_map["JetEta"] = "eta"
    name_map["JetA"] = "area"
    name_map["ptRaw"] = "pt_raw"
    name_map["massRaw"] = "mass_raw"
    name_map["Rho"] = "Rho"
    name_map["ptGenJet"] = "pt_gen"

    jets = events.Jet
    jets["pt_raw"] = (1 - jets["rawFactor"]) * jets.pt
    jets["mass_raw"] = (1 - jets["rawFactor"]) * jets.mass
    jets["pt_gen"] = ak.fill_none(jets.matched_gen.pt, 0)
    jets["Rho"] = events.Rho.fixedGridRhoFastjetAll

    # --- Build corrected jets ---
    jet_factory = CorrectedJetsFactory(name_map, jec_stack)
    corrected_jets = jet_factory.build(jets)

    # --- Set up MET name_map keys ---
    # NanoAODv15 uses PFMET (with sumPtUnclustered) instead of MET (with MetUnclustEnUpDeltaX/Y).
    # Compute unclustered energy dx/dy from sumPtUnclustered: dx = dy = sumPtUnclustered / sqrt(2).
    pfmet = events.PFMET
    uncl_delta = pfmet.sumPtUnclustered / np.sqrt(2.0)
    pfmet["MetUnclustEnUpDeltaX"] = uncl_delta
    pfmet["MetUnclustEnUpDeltaY"] = uncl_delta

    name_map["METpt"] = "pt"
    name_map["METphi"] = "phi"
    name_map["JetPhi"] = "phi"
    name_map["UnClusteredEnergyDeltaX"] = "MetUnclustEnUpDeltaX"
    name_map["UnClusteredEnergyDeltaY"] = "MetUnclustEnUpDeltaY"

    # Type-1 specific keys for Jet collection
    name_map["RawMETpt"] = "pt"
    name_map["RawMETphi"] = "phi"
    name_map["JetRawFactor"] = "rawFactor"
    name_map["JetMuonSubtrFactor"] = "muonSubtrFactor"
    name_map["JetMuonSubtrDeltaPhi"] = "muonSubtrDeltaPhi"
    name_map["JetChEmEF"] = "chEmEF"
    name_map["JetNeEmEF"] = "neEmEF"

    # Type-1 specific keys for CorrT1METJet collection
    name_map["CorrT1JetPt"] = "rawPt"
    name_map["CorrT1JetPhi"] = "phi"
    name_map["CorrT1JetEta"] = "eta"
    name_map["CorrT1JetArea"] = "area"
    name_map["CorrT1JetMuonSubtrFactor"] = "muonSubtrFactor"
    name_map["CorrT1JetMuonSubtrDeltaPhi"] = "muonSubtrDeltaPhi"
    name_map["CorrT1JetEmEF"] = "EmEF"

    # --- Build L1-only and L1L2L3 JEC correctors ---
    from coffea.jetmet_tools import FactorizedJetCorrector

    jec_L1 = FactorizedJetCorrector(
        **{"Summer16_23Sep2016V3_MC_L1FastJet_AK4PFPuppi": evaluator["Summer16_23Sep2016V3_MC_L1FastJet_AK4PFPuppi"]}
    )
    jec_L1L2L3 = FactorizedJetCorrector(
        **{name: evaluator[name] for name in jec_stack_names[0:4]}
    )

    # --- Build CorrectedMETFactory in Type-1 mode ---
    met_factory = CorrectedMETFactory(name_map, jec_L1L2L3=jec_L1L2L3, jec_L1=jec_L1)

    raw_met = events.RawPFMET

    # Attach Rho to CorrT1METJet for JEC evaluation
    corrt1jets = events.CorrT1METJet
    corrt1jets["Rho"] = events.Rho.fixedGridRhoFastjetAll

    corrected_met = met_factory.build(
        pfmet, corrected_jets, in_RawMET=raw_met, in_CorrT1METJets=corrt1jets
    )

    # --- Assertions ---
    # pt_orig and phi_orig should be present (storing raw MET values)
    assert "pt_orig" in ak.fields(corrected_met)
    assert "phi_orig" in ak.fields(corrected_met)

    # Corrected MET should differ from raw MET
    assert not ak.all(corrected_met.pt == corrected_met.pt_orig)
    assert not ak.all(corrected_met.phi == corrected_met.phi_orig)

    # Unclustered energy variations should be present
    assert "MET_UnclusteredEnergy" in ak.fields(corrected_met)
    assert not ak.all(
        corrected_met.MET_UnclusteredEnergy.up.pt
        == corrected_met.MET_UnclusteredEnergy.down.pt
    )

    # JES/JER variations should be present and have up/down fields
    jes_jer_fields = [f for f in ak.fields(corrected_met) if f.startswith(("JER", "JES"))]
    assert len(jes_jer_fields) > 0, "No JES/JER variations found in corrected MET"
    for unc in jes_jer_fields:
        assert "up" in ak.fields(corrected_met[unc]), f"{unc} missing 'up'"
        assert "down" in ak.fields(corrected_met[unc]), f"{unc} missing 'down'"
    # At least some JES sources should have up != down
    n_differ = sum(
        not ak.all(corrected_met[unc].up.pt == corrected_met[unc].down.pt)
        for unc in jes_jer_fields
    )
    assert n_differ > 0, "All JES/JER variations have up == down"

    print("Type-1 corrected MET (nominal):", corrected_met.pt)
    print("Raw MET (orig):", corrected_met.pt_orig)

    # --- Test validation errors ---
    # Only one of jec_L1L2L3/jec_L1 provided should raise
    with pytest.raises(ValueError, match="Both jec_L1L2L3 and jec_L1"):
        CorrectedMETFactory(name_map, jec_L1L2L3=jec_L1L2L3)

    # Type-1 mode without in_RawMET should raise
    met_factory_t1 = CorrectedMETFactory(name_map, jec_L1L2L3=jec_L1L2L3, jec_L1=jec_L1)
    with pytest.raises(ValueError, match="in_RawMET is required"):
        met_factory_t1.build(pfmet, corrected_jets)

    # --- Test without CorrT1METJets (should still work, just Jet contribution) ---
    corrected_met_no_corrt1 = met_factory.build(
        pfmet, corrected_jets, in_RawMET=raw_met
    )
    assert "pt_orig" in ak.fields(corrected_met_no_corrt1)
    assert not ak.all(corrected_met_no_corrt1.pt == corrected_met_no_corrt1.pt_orig)

    print("\nType-1 MET test passed.")


def test_corrected_met_type1_v12():
    """Test CorrectedMETFactory Type-1 mode with NanoAODv12 (missing muonSubtrDeltaPhi, EmEF)."""
    import os

    from coffea.jetmet_tools import CorrectedJetsFactory, CorrectedMETFactory, JECStack
    from coffea.nanoevents import NanoEventsFactory

    events = NanoEventsFactory.from_root(
        {os.path.abspath("tests/samples/nano_tt_v12.root"): "Events"},
        metadata={},
        mode="eager",
    ).events()

    # --- Build JEC stack ---
    jec_stack_names = [
        "Summer16_23Sep2016V3_MC_L1FastJet_AK4PFPuppi",
        "Summer16_23Sep2016V3_MC_L2Relative_AK4PFPuppi",
        "Summer16_23Sep2016V3_MC_L2L3Residual_AK4PFPuppi",
        "Summer16_23Sep2016V3_MC_L3Absolute_AK4PFPuppi",
        "Spring16_25nsV10_MC_PtResolution_AK4PFPuppi",
        "Spring16_25nsV10_MC_SF_AK4PFPuppi",
    ]
    for key in evaluator.keys():
        if "Summer16_23Sep2016V3_MC_UncertaintySources_AK4PFPuppi" in key:
            jec_stack_names.append(key)

    jec_inputs = {name: evaluator[name] for name in jec_stack_names}
    jec_stack = JECStack(jec_inputs)

    # --- Build name_map for CorrectedJetsFactory ---
    name_map = jec_stack.blank_name_map
    name_map["JetPt"] = "pt"
    name_map["JetMass"] = "mass"
    name_map["JetEta"] = "eta"
    name_map["JetA"] = "area"
    name_map["ptRaw"] = "pt_raw"
    name_map["massRaw"] = "mass_raw"
    name_map["Rho"] = "Rho"
    name_map["ptGenJet"] = "pt_gen"

    jets = events.Jet
    jets["pt_raw"] = (1 - jets["rawFactor"]) * jets.pt
    jets["mass_raw"] = (1 - jets["rawFactor"]) * jets.mass
    jets["pt_gen"] = ak.fill_none(jets.matched_gen.pt, 0)
    jets["Rho"] = events.Rho.fixedGridRhoFastjetAll

    # NanoAODv12 lacks muonSubtrDeltaPhi — fill with zeros (phi_noMuRaw = phi)
    jets["muonSubtrDeltaPhi"] = ak.zeros_like(jets.phi)

    # --- Build corrected jets ---
    jet_factory = CorrectedJetsFactory(name_map, jec_stack)
    corrected_jets = jet_factory.build(jets)

    # --- Set up MET name_map keys ---
    met = events.MET
    name_map["METpt"] = "pt"
    name_map["METphi"] = "phi"
    name_map["JetPhi"] = "phi"
    name_map["UnClusteredEnergyDeltaX"] = "MetUnclustEnUpDeltaX"
    name_map["UnClusteredEnergyDeltaY"] = "MetUnclustEnUpDeltaY"

    # Type-1 specific keys for Jet collection
    name_map["RawMETpt"] = "pt"
    name_map["RawMETphi"] = "phi"
    name_map["JetRawFactor"] = "rawFactor"
    name_map["JetMuonSubtrFactor"] = "muonSubtrFactor"
    name_map["JetMuonSubtrDeltaPhi"] = "muonSubtrDeltaPhi"
    name_map["JetChEmEF"] = "chEmEF"
    name_map["JetNeEmEF"] = "neEmEF"

    # Type-1 specific keys for CorrT1METJet collection
    name_map["CorrT1JetPt"] = "rawPt"
    name_map["CorrT1JetPhi"] = "phi"
    name_map["CorrT1JetEta"] = "eta"
    name_map["CorrT1JetArea"] = "area"
    name_map["CorrT1JetMuonSubtrFactor"] = "muonSubtrFactor"
    name_map["CorrT1JetMuonSubtrDeltaPhi"] = "muonSubtrDeltaPhi"
    name_map["CorrT1JetEmEF"] = "EmEF"

    # --- Build L1-only and L1L2L3 JEC correctors ---
    from coffea.jetmet_tools import FactorizedJetCorrector

    jec_L1 = FactorizedJetCorrector(
        **{"Summer16_23Sep2016V3_MC_L1FastJet_AK4PFPuppi": evaluator["Summer16_23Sep2016V3_MC_L1FastJet_AK4PFPuppi"]}
    )
    jec_L1L2L3 = FactorizedJetCorrector(
        **{name: evaluator[name] for name in jec_stack_names[0:4]}
    )

    # --- Build CorrectedMETFactory in Type-1 mode ---
    met_factory = CorrectedMETFactory(name_map, jec_L1L2L3=jec_L1L2L3, jec_L1=jec_L1)

    raw_met = events.RawMET

    # NanoAODv12 CorrT1METJet lacks muonSubtrDeltaPhi and EmEF — fill with zeros
    corrt1jets = events.CorrT1METJet
    corrt1jets["Rho"] = events.Rho.fixedGridRhoFastjetAll
    corrt1jets["muonSubtrDeltaPhi"] = ak.zeros_like(corrt1jets.phi)
    corrt1jets["EmEF"] = ak.zeros_like(corrt1jets.phi)

    corrected_met = met_factory.build(
        met, corrected_jets, in_RawMET=raw_met, in_CorrT1METJets=corrt1jets
    )

    # --- Assertions ---
    assert "pt_orig" in ak.fields(corrected_met)
    assert "phi_orig" in ak.fields(corrected_met)
    assert not ak.all(corrected_met.pt == corrected_met.pt_orig)
    assert not ak.all(corrected_met.phi == corrected_met.phi_orig)

    assert "MET_UnclusteredEnergy" in ak.fields(corrected_met)
    assert not ak.all(
        corrected_met.MET_UnclusteredEnergy.up.pt
        == corrected_met.MET_UnclusteredEnergy.down.pt
    )

    jes_jer_fields = [f for f in ak.fields(corrected_met) if f.startswith(("JER", "JES"))]
    assert len(jes_jer_fields) > 0
    n_differ = sum(
        not ak.all(corrected_met[unc].up.pt == corrected_met[unc].down.pt)
        for unc in jes_jer_fields
    )
    assert n_differ > 0, "All JES/JER variations have up == down"

    # Also test without CorrT1METJets
    corrected_met_no_corrt1 = met_factory.build(
        met, corrected_jets, in_RawMET=raw_met
    )
    assert not ak.all(corrected_met_no_corrt1.pt == corrected_met_no_corrt1.pt_orig)

    print("Type-1 corrected MET v12 (nominal):", corrected_met.pt)
    print("Raw MET v12 (orig):", corrected_met.pt_orig)
    print("\nType-1 MET v12 test passed.")
