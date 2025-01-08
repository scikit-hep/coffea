import parsl

from parsl.app.app import python_app

from parsl.providers import LocalProvider
from parsl.config import Config
from parsl.executors import HighThroughputExecutor

from ..executor import _futures_handler
from .timeout import timeout


_default_cfg = Config(
    executors=[
        HighThroughputExecutor(
            label="coffea_parsl_default",
            cores_per_worker=1,
            provider=LocalProvider(
                init_blocks=1,
                max_blocks=1,
            ),
        )
    ],
    strategy=None,
)


def _parsl_initialize(config=None):
    parsl.clear()
    parsl.load(config)


def _parsl_stop():
    parsl.dfk().cleanup()
    parsl.clear()


@timeout
@python_app
def derive_chunks(filename, treename, chunksize, ds, timeout=10):
    import uproot
    from collections.abc import Sequence

    uproot.XRootDSource.defaults["parallel"] = False

    afile = uproot.open({filename: None})

    tree = None
    if isinstance(treename, str):
        tree = afile[treename]
    elif isinstance(treename, Sequence):
        for name in reversed(treename):
            if name in afile:
                tree = afile[name]
    else:
        raise Exception(
            "treename must be a str or Sequence but is a %s!" % repr(type(treename))
        )

    if tree is None:
        raise Exception(
            "No tree found, out of possible tree names: %s" % repr(treename)
        )

    nentries = tree.numentries
    return (
        ds,
        treename,
        [(filename, chunksize, index) for index in range(nentries // chunksize + 1)],
    )


def _parsl_get_chunking(filelist, chunksize, status=True, timeout=10):
    futures = set(
        derive_chunks(fn, tn, chunksize, ds, timeout=timeout) for ds, fn, tn in filelist
    )

    items = []

    def chunk_accumulator(total, result):
        ds, treename, chunks = result
        for chunk in chunks:
            total.append((ds, chunk[0], treename, chunk[1], chunk[2]))

    _futures_handler(
        futures, items, status, "files", "Preprocessing", chunk_accumulator, None
    )

    return items
