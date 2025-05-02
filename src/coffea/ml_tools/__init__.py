"""Tools to interface with various ML inference services

Providing the interfaces to the run ML inference such that user can simply
handle data mangling in awkward/numpy formats. Specifics of passing numpy arrays
conversion and the handling of dask are mostly abstract away.
"""

from coffea.ml_tools.helper import numpy_call_wrapper
from coffea.ml_tools.tf_wrapper import tf_wrapper
from coffea.ml_tools.torch_wrapper import torch_wrapper
from coffea.ml_tools.triton_wrapper import triton_wrapper
from coffea.ml_tools.xgboost_wrapper import xgboost_wrapper

__all__ = [
    "numpy_call_wrapper",
    "torch_wrapper",
    "triton_wrapper",
    "xgboost_wrapper",
    "tf_wrapper",
]
