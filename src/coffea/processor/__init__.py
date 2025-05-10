"""A framework for analysis scale-out"""

from .accumulator import (
    Accumulatable,
    AccumulatorABC,
    accumulate,
    column_accumulator,
    defaultdict_accumulator,
    dict_accumulator,
    list_accumulator,
    set_accumulator,
    value_accumulator,
)
from .executor import (
    DaskExecutor,
    FuturesExecutor,
    IterativeExecutor,
    Runner,
)
from .processor import ProcessorABC

__all__ = [
    "ProcessorABC",
    "Weights",
    "PackedSelection",
    "IterativeExecutor",
    "FuturesExecutor",
    "DaskExecutor",
    "Runner",
    "accumulate",
    "Accumulatable",
    "AccumulatorABC",
    "value_accumulator",
    "list_accumulator",
    "set_accumulator",
    "dict_accumulator",
    "defaultdict_accumulator",
    "column_accumulator",
]
