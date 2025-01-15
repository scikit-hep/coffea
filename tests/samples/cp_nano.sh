# Current NanoAOD: v6
# for MC, we need to have at least 1 muon with Muon_genPartIdx == -1 for Rochester tests
first=1
last=40
datafname=root://cmsxrootd.fnal.gov///store/data/Run2018A/DoubleMuon/NANOAOD/Nano25Oct2019-v1/230000/9C013FBF-416D-204C-BE6F-36B009DC25DD.root
mcfname=root://cmsxrootd.fnal.gov///store/mc/RunIIAutumn18NanoAODv6/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/Nano25Oct2019_102X_upgrade2018_realistic_v20_ext2-v1/260000/244E20C4-9694-9840-830B-8513C1FB5448.root

rooteventselector --recreate --first $first --last $last $datafname:Events nano_dimuon.root
rooteventselector $datafname:Runs nano_dimuon.root
rooteventselector --recreate --first $first --last $last $mcfname:Events nano_dy.root
rooteventselector $mcfname:Runs nano_dy.root
