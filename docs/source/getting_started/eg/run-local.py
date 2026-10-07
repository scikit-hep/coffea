# necessary coffea infrastructure support
# my analysis code
from dimuonmass import DiMuonMassCalculator

from coffea import processor
from coffea.nanoevents import NanoAODSchema
from coffea.util import save

# define the different data sets along with their
# metadata and the files that they contain
fileset = {
    "DYJets": {
        "files": {"nano_dy.root": "Events"},
        "metadata": {"is_mc": True},
    },
    "Data": {
        "files": {"nano_dimuon.root": "Events"},
        "metadata": {"is_mc": False},
    },
}

# define how we are going to run the analysis over the data
runner = processor.Runner(
    # iterative is the simplest to help us debug our code
    executor=processor.IterativeExecutor(),
    # we are reading CMS NanoAOD files
    schema=NanoAODSchema,
    # we want to save the metrics to also view the performance
    savemetrics=True,
)

# this is the line that actually does the processing
result, metrics = runner(fileset, processor_instance=DiMuonMassCalculator())

# view the metrics which shows which columns were read,
# number of entries, number of chunks, etc...
print(metrics)

# the actual analysis result (histograms and other accumulators)
print(result)
# save the result to a file
save(result, "dimuonmass.coffea")
