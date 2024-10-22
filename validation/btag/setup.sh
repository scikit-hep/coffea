wget https://github.com/cms-sw/cmssw/raw/CMSSW_10_0_X/CondTools/BTau/test/BTagCalibrationStandalone.cpp
wget https://github.com/cms-sw/cmssw/raw/CMSSW_10_0_X/CondTools/BTau/test/BTagCalibrationStandalone.h
cp ../../tests/samples/DeepCSV_102XSF_V1.btag.csv.gz .
gzip -d DeepCSV_102XSF_V1.btag.csv.gz
cp ../../tests/samples/testBTagSF.btag.csv .
