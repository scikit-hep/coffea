import os.path as osp
import random
from pathlib import Path

import awkward as ak
import fsspec
import numpy as np
import pytest

from coffea import processor
from coffea.nanoevents import schemas

# we want repeatable failures, and know that we never run indefinitely
random.seed(1234)


class UnstableNanoEventsProcessor(processor.ProcessorABC):
    @property
    def accumulator(self):
        return {"cutflow": {}}

    def process(self, events):
        if random.random() < 0.5:
            raise RuntimeError("Random failure for testing checkpointing")

        output = self.accumulator
        dataset = events.metadata["dataset"]
        output["cutflow"]["%s_pt" % dataset] = ak.sum(ak.num(events.Muon, axis=1))
        return output

    def postprocess(self, accumulator):
        return accumulator


@pytest.mark.parametrize("filetype", ["root", "parquet"])
def test_checkpointing(filetype):
    suffix = ".root" if filetype == "root" else ".parquet"
    filelist = {
        "ZJets": {
            "treename": "Events",
            "files": [osp.abspath(f"tests/samples/nano_dy{suffix}")],
        },
        "Data": {
            "treename": "Events",
            "files": [osp.abspath(f"tests/samples/nano_dimuon{suffix}")],
        },
    }

    executor = processor.IterativeExecutor()

    checkpoint_dir = str(Path(__file__).parent / f"test_checkpointing_{filetype}")
    # checkpoint_dir = "root://cmseos.fnal.gov//store/user/ikrommyd/test"
    checkpointer = processor.SimpleCheckpointer(checkpoint_dir)
    run = processor.Runner(
        executor=executor,
        schema=schemas.NanoAODSchema,
        chunksize=10,
        format=filetype,
        checkpointer=checkpointer,
    )
    # use the chunk generator to not re-run the preprocessing step
    chunks = list(run.preprocess(filelist, treename="Events"))

    def chunk_gen():
        yield from chunks

    # number of WorkItems
    n_expected_checkpoints = len(chunks)
    ntries = 0
    fs, path = fsspec.url_to_fs(checkpoint_dir)

    # keep trying until we have as many checkpoints as WorkItems
    while len(list(filter(fs.isfile, fs.glob(f"{path}/**")))) != n_expected_checkpoints:
        fs.invalidate_cache()
        ntries += 1
        try:
            out = run(chunk_gen(), UnstableNanoEventsProcessor(), treename="Events")
        except Exception:
            print(f"Run failed, trying again, try number {ntries}...")
            continue

    # make sure we have as many checkpoints as WorkItems
    fs.invalidate_cache()
    assert len(list(filter(fs.isfile, fs.glob(f"{path}/**")))) == n_expected_checkpoints

    # make sure we got the right answer
    assert out == {"cutflow": {"Data_pt": np.int64(84), "ZJets_pt": np.int64(18)}}

    # cleanup
    fs.rm(path, recursive=True)
