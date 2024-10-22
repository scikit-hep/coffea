import os

import awkward as ak
import toml
import uproot

from coffea import processor
from coffea.nanoevents import schemas
from coffea.processor.test_items import NanoEventsProcessor

if __name__ == "__main__":
    config_dict = {
        "skyhook": {
            "ceph_config_path": "/tmp/testskyhookjob/ceph.conf",
            "ceph_data_pool": "cephfs_data",
        }
    }
    with open("/root/.coffea.toml", "w") as f:
        toml.dump(config_dict, f)

    ak.to_parquet(
        uproot.lazy("tests/samples/nano_dy.root:Events"),
        "nano_dy.parquet",
        list_to32=True,
        use_dictionary=False,
        compression="GZIP",
        compression_level=1,
    )

    ak.to_parquet(
        uproot.lazy("tests/samples/nano_dimuon.root:Events"),
        "nano_dimuon.parquet",
        list_to32=True,
        use_dictionary=False,
        compression="GZIP",
        compression_level=1,
    )

    os.makedirs("/mnt/cephfs/nanoevents/ZJets")
    os.makedirs("/mnt/cephfs/nanoevents/Data")
    for i in range(6):
        os.system(
            f"cp nano_dy.parquet /mnt/cephfs/nanoevents/ZJets/nano_dy.{i}.parquet"
        )
        os.system(
            f"cp nano_dimuon.parquet /mnt/cephfs/nanoevents/Data/nano_dimuon.{i}.parquet"
        )

    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(processes=True, threads_per_worker=1)
    client = Client(cluster)

    executor = processor.DaskExecutor(client=client)

    run = processor.Runner(
        executor=executor,
        use_skyhook=True,
        format="parquet",
        schema=schemas.NanoAODSchema,
    )

    hists = run(
        {
            "ZJets": "/mnt/cephfs/nanoevents/ZJets",
            "Data": "/mnt/cephfs/nanoevents/Data",
        },
        "Events",
        processor_instance=NanoEventsProcessor(),
    )

    assert hists["cutflow"]["ZJets_pt"] == 108
    assert hists["cutflow"]["ZJets_mass"] == 36
    assert hists["cutflow"]["Data_pt"] == 504
    assert hists["cutflow"]["Data_mass"] == 396

    # now run again on parquet files in cephfs (without any pushdown)
    executor_args = {"client": client}

    run = processor.Runner(
        executor=executor,
        format="parquet",
        schema=schemas.NanoAODSchema,
        use_skyhook=True,
    )

    hists = run(
        {
            "ZJets": "/mnt/cephfs/nanoevents/ZJets",
            "Data": "/mnt/cephfs/nanoevents/Data",
        },
        "Events",
        processor_instance=NanoEventsProcessor(),
    )

    assert hists["cutflow"]["ZJets_pt"] == 108
    assert hists["cutflow"]["ZJets_mass"] == 36
    assert hists["cutflow"]["Data_pt"] == 504
    assert hists["cutflow"]["Data_mass"] == 396
