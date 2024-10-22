import os

import numpy as np
import ROOT
import uproot
from awkward import JaggedArray

ROOT.PyConfig.IgnoreCommandLineOptions = True

cfname = "rochester/roccor.Run2.v3/RoccoR.cc"
txtname = "rochester/roccor.Run2.v3/RoccoR2018.txt"
treename = "Events"
datafname = "tests/samples/nano_dimuon.root"
mcfname = "tests/samples/nano_dy.root"


# official version of rochester
PWD = os.getcwd()
ROOT.gROOT.ProcessLine(f".L {PWD}/{cfname}")
roccor = ROOT.RoccoR(f"{PWD}/{txtname}")

for isData, fname in [(True, datafname), (False, mcfname)]:
    branches = ["Muon_charge", "Muon_pt", "Muon_eta", "Muon_phi"]
    if not isData:
        branches += ["Muon_genPartIdx", "GenPart_pt", "Muon_nTrackerLayers"]
    res = []
    err = []
    fullu = []
    for i, arrays in enumerate(
        uproot.iterate(
            fname, treename, branches=branches, namedecode="utf-8", entrysteps=200000
        )
    ):
        charge = arrays["Muon_charge"]
        pt = arrays["Muon_pt"]
        eta = arrays["Muon_eta"]
        phi = arrays["Muon_phi"]
        if not isData:
            # for default if gen present
            gid = arrays["Muon_genPartIdx"]
            gpt = arrays["GenPart_pt"]
            # for backup w/o gen
            nl = arrays["Muon_nTrackerLayers"]
            u = np.random.rand(*pt.flatten().shape)
            u = JaggedArray.fromoffsets(pt.offsets, u)
            fullu += [u]
        for ie in range(len(pt)):
            subres = []
            suberr = []
            for im in range(len(pt[ie])):
                if isData:
                    subres += [
                        roccor.kScaleDT(
                            int(charge[ie][im]),
                            float(pt[ie][im]),
                            float(eta[ie][im]),
                            float(phi[ie][im]),
                        )
                    ]
                    suberr += [
                        roccor.kScaleDTerror(
                            int(charge[ie][im]),
                            float(pt[ie][im]),
                            float(eta[ie][im]),
                            float(phi[ie][im]),
                        )
                    ]
                else:
                    if gid[ie][im] >= 0:
                        subres += [
                            roccor.kSpreadMC(
                                int(charge[ie][im]),
                                float(pt[ie][im]),
                                float(eta[ie][im]),
                                float(phi[ie][im]),
                                float(gpt[ie][gid[ie][im]]),
                            )
                        ]
                        suberr += [
                            roccor.kSpreadMCerror(
                                int(charge[ie][im]),
                                float(pt[ie][im]),
                                float(eta[ie][im]),
                                float(phi[ie][im]),
                                float(gpt[ie][gid[ie][im]]),
                            )
                        ]
                    else:
                        subres += [
                            roccor.kSmearMC(
                                int(charge[ie][im]),
                                float(pt[ie][im]),
                                float(eta[ie][im]),
                                float(phi[ie][im]),
                                int(nl[ie][im]),
                                float(u[ie][im]),
                            )
                        ]
                        suberr += [
                            roccor.kSmearMCerror(
                                int(charge[ie][im]),
                                float(pt[ie][im]),
                                float(eta[ie][im]),
                                float(phi[ie][im]),
                                int(nl[ie][im]),
                                float(u[ie][im]),
                            )
                        ]
            res += [subres]
            err += [suberr]
    res = JaggedArray.fromiter(res)
    err = JaggedArray.fromiter(err)
    outres = res.flatten()
    outerr = err.flatten()
    np.save(fname.replace(".root", "_rochester.npy"), outres)
    np.save(fname.replace(".root", "_rochester_err.npy"), outerr)
    if not isData:
        outrand = np.concatenate([ui.flatten() for ui in fullu])
        np.save(fname.replace(".root", "_rochester_rand.npy"), outrand)
