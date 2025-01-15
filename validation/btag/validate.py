import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gInterpreter.LoadFile("BTagCalibrationStandalone.cpp")

import numpy

from coffea.btag_tools import BTagScaleFactor


def stdvec(stringlist):
    out = ROOT.std.vector("string")()
    for item in stringlist:
        out.push_back(item)
    return out


def makesf(btagReader):
    def btv_sf(syst, flavor, abseta, pt, discr=None):
        def wrap(flavor, abseta, pt, discr):
            if flavor == 5:
                btvflavor = ROOT.BTagEntry.FLAV_B
            elif flavor == 4:
                btvflavor = ROOT.BTagEntry.FLAV_C
            elif flavor == 0:
                btvflavor = ROOT.BTagEntry.FLAV_UDSG
            else:
                raise ValueError
            if discr is None:
                return btagReader.eval_auto_bounds(syst, btvflavor, abseta, pt)
            return btagReader.eval_auto_bounds(syst, btvflavor, abseta, pt, discr)

        return numpy.vectorize(wrap, otypes="d")(flavor, abseta, pt, discr)

    return btv_sf


def validate_btag(filename, btagtype, etamax):
    btagData = ROOT.BTagCalibration(btagtype, filename)
    npts = 10000
    flavor = numpy.full(npts, 5)
    abseta = numpy.random.uniform(0, etamax, size=npts)
    pt = numpy.random.exponential(50, size=npts) + numpy.random.exponential(
        20, size=npts
    )
    pt = numpy.maximum(20.1, pt)
    discr = numpy.random.rand(npts)

    coffea_sf = BTagScaleFactor(
        filename, BTagScaleFactor.RESHAPE, "iterativefit", keep_df=True
    )
    btagReader = ROOT.BTagCalibrationReader(
        ROOT.BTagEntry.OP_RESHAPING, "central", stdvec(["up_jes", "down_jes"])
    )
    btagReader.load(btagData, ROOT.BTagEntry.FLAV_B, "iterativefit")
    btv_sf = makesf(btagReader)

    for syst in ["central", "up_jes", "down_jes"]:
        csf = coffea_sf.eval(syst, flavor, abseta, pt, discr)
        bsf = btv_sf(syst, flavor, abseta, pt, discr)
        print(abs(csf - bsf).max())

    flavor = numpy.random.choice([0, 4, 5], size=npts)
    coffea_sf = BTagScaleFactor(
        filename, BTagScaleFactor.TIGHT, "comb,mujets,incl", keep_df=True
    )
    btagReader = ROOT.BTagCalibrationReader(
        ROOT.BTagEntry.OP_TIGHT, "central", stdvec(["up", "down"])
    )
    btagReader.load(btagData, ROOT.BTagEntry.FLAV_B, "comb")
    btagReader.load(btagData, ROOT.BTagEntry.FLAV_C, "mujets")
    btagReader.load(btagData, ROOT.BTagEntry.FLAV_UDSG, "incl")
    btv_sf = makesf(btagReader)

    for syst in ["central", "up", "down"]:
        csf = coffea_sf.eval(syst, flavor, abseta, pt, discr)
        bsf = btv_sf(syst, flavor, abseta, pt, discr)
        print(abs(csf - bsf).max())


validate_btag(
    filename="DeepCSV_102XSF_V1.btag.csv",
    btagtype="DeepCSV",
    etamax=2.5,
)
validate_btag(
    filename="testBTagSF.btag.csv",
    btagtype="CSVv2",
    etamax=2.4,
)
