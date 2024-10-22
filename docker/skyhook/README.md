# Reading Nanoevents from Parquet files stored in CephFS through the Arrow Datasets API by pushing down Scan operations and processing with Coffea

We added a new executor API called the [`run_parquet_job`](../../coffea/processor/executor.py#L1377) that allows reading columns in parallel using Dask out of  Parquet files stored in CephFS by pushing down projection operations into the Ceph OSDs to minimize the overhead of moving data through the network.

# Getting Started

**NOTE:** Please make sure that [Docker](https://www.docker.com/) is installed and running.

1. Clone the repository.
```bash
git clone https://github.com/CoffeaTeam/coffea
cd coffea/
```

1. Install the [Popper](https://github.com/getpopper/popper) container-native workflow engine.

```bash
curl -sSfL https://raw.githubusercontent.com/getpopper/popper/master/install.sh | sh
```

2. Run a single step workflow. Running the workflow will create a single-node Ceph cluster in a Docker container, mount CephFS, and will open up a Jupyter environment.

```bash
cd docker/skyhook/
popper run -f skyhook-coffea-demo.yml -w ../../
```

3. Navigate to [`binder/nanoevents_pq.ipynb`](../../binder/nanoevents_pq.ipynb) and open the notebook. This notebook based guide starts with a brief hands-on on PyArrow with the   [`SkyhookFileFormat`](https://github.com/uccross/arrow/blob/rados-dataset-dev/cpp/src/arrow/dataset/file_rados_parquet.h#L126) API and then shows how to read data into Coffea processors from CephFS by doing projection pushdowns using the Arrow Dataset API and the `SkyhookFileFormat`.
