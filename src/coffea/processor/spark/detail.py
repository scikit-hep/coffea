from concurrent.futures import ThreadPoolExecutor

import pyspark.sql
import pyspark.sql.functions as fn
from pyarrow.util import guid
from tqdm import tqdm

try:
    from collections.abc import Sequence
except ImportError:
    from collections.abc import Sequence

from coffea.processor.executor import _futures_handler

# this is a reasonable local spark configuration
_default_config = (
    pyspark.sql.SparkSession.builder.appName("coffea-analysis-%s" % guid())
    .master("local[*]")
    .config("spark.sql.execution.arrow.enabled", "true")
    .config("spark.sql.execution.arrow.maxRecordsPerBatch", 200000)
)


def _spark_initialize(config=_default_config, **kwargs):
    spark_progress = False
    if "spark_progress" in kwargs.keys():
        spark_progress = kwargs["spark_progress"]

    cfg_actual = config
    # get spark to not complain about missing log configs
    cfg_actual = cfg_actual.config(
        "spark.driver.extraJavaOptions", "-Dlog4jspark.root.logger=ERROR,console"
    )
    if not spark_progress:
        cfg_actual = cfg_actual.config("spark.ui.showConsoleProgress", "false")

    kwargs.setdefault("bindAddress", None)
    if kwargs["bindAddress"] is not None:
        cfg_actual = cfg_actual.config(
            "spark.driver.bindAddress", kwargs["bindAddress"]
        )
    kwargs.setdefault("host", None)
    if kwargs["host"] is not None:
        cfg_actual = cfg_actual.config("spark.driver.host", kwargs["host"])

    session = cfg_actual.getOrCreate()
    sc = session.sparkContext

    if "log_level" in kwargs.keys():
        sc.setLogLevel(kwargs["log_level"])
    else:
        sc.setLogLevel("ERROR")

    return session


def _read_df(
    spark, dataset, files_or_dirs, ana_cols, partitionsize, file_type, treeName
):
    flist = files_or_dirs
    tname = treeName
    if isinstance(files_or_dirs, dict):
        tname = files_or_dirs["treename"]
        flist = files_or_dirs["files"]
    if not isinstance(flist, Sequence):
        raise ValueError("spark dataset file list must be a Sequence (like list())")
    df = (
        spark.read.format(file_type)
        .option("tree", tname)
        .option("threadCount", "-1")
        .load(flist)
    )
    count = df.count()

    df_cols = set(df.columns)
    cols_in_df = ana_cols.intersection(df_cols)
    df = df.select(*cols_in_df)
    missing_cols = ana_cols - cols_in_df
    for missing in missing_cols:
        df = df.withColumn(missing, fn.lit(0.0))
    # compatibility with older pyarrow which doesn't understand array<boolean>
    for col, dtype in df.dtypes:
        if dtype == "array<boolean>":
            tempcol = col + "tempbool"
            df = df.withColumnRenamed(col, tempcol)
            df = df.withColumn(col, df[tempcol].cast("array<tinyint>")).drop(tempcol)
    df = df.withColumn("dataset", fn.lit(dataset))
    npartitions = (count // partitionsize) + 1
    actual_partitions = df.rdd.getNumPartitions()
    avg_counts = count / actual_partitions
    if actual_partitions > 1.50 * npartitions or avg_counts > partitionsize:
        df = df.repartition(npartitions)

    return df, dataset, count


def _spark_make_dfs(
    spark,
    fileset,
    partitionsize,
    columns,
    thread_workers,
    file_type,
    treeName,
    status=True,
):
    dfs = {}
    ana_cols = set(columns)

    with ThreadPoolExecutor(max_workers=thread_workers) as executor:
        futures = {
            executor.submit(
                _read_df, spark, ds, files, ana_cols, partitionsize, file_type, treeName
            )
            for ds, files in fileset.items()
        }

        for df, ds, count in tqdm(
            _futures_handler(futures, timeout=None),
            disable=not status,
            unit="dataset",
            total=len(fileset),
            desc="loading",
        ):
            dfs[ds] = (df, count)

    return dfs


def _spark_stop(spark):
    # this may do more later?
    spark._jvm.SparkSession.clearActiveSession()
    spark.stop()
