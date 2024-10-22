import numpy as np
import ROOT
from dummy_distributions import dummy_pt_eta

counts, test_in1, test_in2 = dummy_pt_eta()

f = ROOT.TFile.Open("samples/testSF2d.root")
sf = f.Get("scalefactors_Tight_Electron")

xmin, xmax = sf.GetXaxis().GetXmin(), sf.GetXaxis().GetXmax()
ymin, ymax = sf.GetYaxis().GetXmin(), sf.GetYaxis().GetXmax()

test_out = np.empty_like(test_in1)
for i, (eta, pt) in enumerate(zip(test_in1, test_in2)):
    if xmax <= eta:
        eta = xmax - 1.0e-5
    elif eta < xmin:
        eta = xmin
    if ymax <= pt:
        pt = ymax - 1.0e-5
    elif pt < ymin:
        pt = ymin
    ib = sf.FindBin(eta, pt)
    test_out[i] = sf.GetBinContent(ib)

print(repr(test_out))
