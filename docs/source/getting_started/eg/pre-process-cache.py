import json
import pickle
from pathlib import Path

# my analysis code
from dimuonmass import DiMuonMassCalculator

# necessary coffea infrastructure support
from coffea import processor
from coffea.nanoevents import NanoAODSchema
from coffea.util import save


def load(filepath, runner):
    """load the input JSON fileset as a pre-processed file-set

    We cache the list of coffea WorkItems into a pickle file so that
    future runs with the exact same JSON fileset can skip the pre-processing step.
    The cache is re-written if it doesn't exist yet or if the input
    JSON filepath has been modified more recently than the cache file.

    Parameters
    ----------
    filepath: str or Path
        path to the JSON fileset to use
    runner: coffea.processor.Runner
        use to do the pre-processing if needed
    """

    # first we just open the file to hash it quick
    filepath = Path(filepath)
    cache_fp = Path(f"{filepath.stem}-pre-processed-cache.pkl")
    if cache_fp.is_file() and cache_fp.stat().st_mtime > filepath.stat().st_mtime:
        # cache hit: it exists and is newer than source filepath
        with open(cache_fp, "rb") as f:
            return pickle.load(f)

    # cache miss: do pre-processing
    with open(filepath) as f:
        fileset = json.load(f)

    preprocessed_fileset = list(runner.preprocess(fileset))
    with open(cache_fp, "wb") as f:
        pickle.dump(preprocessed_fileset, f)
    return preprocessed_fileset


# define how we are going to run the analysis over the data
runner = processor.Runner(
    # just use the 8 cores on this interactive node
    executor=processor.FuturesExecutor(),
    schema=NanoAODSchema,
    savemetrics=True,
)

# this is the line that actually does the processing
result, metrics = runner(
    load("input-sample.json", runner), processor_instance=DiMuonMassCalculator()
)
print(metrics)
print(result)
save(result, "dimuonmass.coffea")
