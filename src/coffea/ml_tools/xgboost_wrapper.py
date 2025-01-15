import warnings
from typing import Dict, Optional

import numpy

from coffea.ml_tools.helper import nonserializable_attribute, numpy_call_wrapper

_xgboost_import_error = None
try:
    import xgboost
except (ImportError, ModuleNotFoundError) as err:
    _xgboost_import_error = err


class xgboost_wrapper(numpy_call_wrapper, nonserializable_attribute):
    """
    Very simple wrapper for xgbooster inference. The xgboost.Booster object is
    nonserializable, so the users should pass in the xgboost model file.

    Parameters
    ----------
        fname: str
            Path to the xgboost model file, such that an `xgbooster` can be created
            via `xgboost.Booster(model_file=fname)`.
    """

    def __init__(self, fname):
        if _xgboost_import_error is not None:
            warnings.warn(
                "Users should make sure the xgboost package is installed before proceeding!\n"
                "> pip install xgboost==1.5.1\n"
                "or\n"
                "> conda install xgboost==1.5.1",
                UserWarning,
            )
            raise _xgboost_import_error

        nonserializable_attribute.__init__(self, ["xgbooster"])
        self.xgboost_file = fname

    def _create_xgbooster(self) -> "xgboost.Booster":
        # Automatic detection of compressed model file
        return xgboost.Booster(model_file=self.xgboost_file)

    def validate_numpy_input(
        self,
        data: numpy.ndarray,
        dmat_args: Optional[Dict] = None,
        predict_args: Optional[Dict] = None,
    ):
        """
        Check that the arguments to be passed into the actual xgboost inference
        request are valid.

        The inner most dimension of the data array should be smaller than the
        number of features of the xgboost model. (Will raise a warning if
        mismatched). We will not attempt to parse the kwargs passed to the
        construction of a DMatrix, or the predict call, as those advanced
        features are expected to be properly handled by the user.

        Parameters
        ----------
            data: np.ndarray
                The data to pass into the `xgboost.DMatrix` construction.
            dmat_args: dict[str,str], optional
                Keyword arguments to pass into the `xgboost.DMatrix` construction.
            predict_args: dict[str,str], optional
                Keyword arguments to pass to the actual prediction step of `xgboost`,
                ie: the `predict` method of `xgbooster.Booster.predict`. Note that the
                first argument of that method is handled by this method.
        """
        ndims = data.shape[-1]
        nfeat = self.xgbooster.num_features()
        if ndims > nfeat:
            raise ValueError(
                f"Input shape {data.shape} exceeded number of features ({nfeat})"
            )
        elif ndims < nfeat:
            warnings.warn(
                f"Input shape {data.shape} smaller than number of features ({nfeat})",
                UserWarning,
            )

    def numpy_call(
        self,
        data: numpy.ndarray,
        dmat_args: Optional[Dict] = None,
        predict_args: Optional[Dict] = None,
    ):
        """
        Pass the numpy array data as-is to the construction of an
        xgboost.DMatrix constructor (with additional keyword arguments should
        they be specified), then run the xgboost.Booster.predict method (with
        additional keyword arguments).

        Parameters
        ----------
            data: np.ndarray
                The data to pass into the `xgboost.DMatrix` construction.
            dmat_args: dict[str,str], optional
                Keyword arguments to pass into the `xgboost.DMatrix` construction.
            predict_args: dict[str,str], optional
                Keyword arguments to pass to the actual prediction step of `xgboost`,
                ie: the `predict` method of `xgbooster.Booster.predict`. Note that the
                first argument of that method is handled by this method.
        """
        if dmat_args is None:
            dmat_args = {}
        if predict_args is None:
            predict_args = {}
        mat = xgboost.DMatrix(data, **dmat_args)
        return self.xgbooster.predict(mat, **predict_args)
