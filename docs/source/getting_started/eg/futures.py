import json

# my analysis code
from dimuonmass import DiMuonMassCalculator

# necessary coffea infrastructure support
from coffea import processor
from coffea.nanoevents import NanoAODSchema
from coffea.util import save

# input-sample.json is the fileset constructed
# from coffea.dataset_tools.dataset_query
with open("input-sample.json") as f:
    fileset = json.load(f)

# define how we are going to run the analysis over the data
runner = processor.Runner(
    # just use the 8 cores on this interactive node
    executor=processor.FuturesExecutor(),
    schema=NanoAODSchema,
    savemetrics=True,
)

# this is the line that actually does the processing
result, metrics = runner(fileset, processor_instance=DiMuonMassCalculator())
print(metrics)
print(result)
save(result, "dimuonmass.coffea")
