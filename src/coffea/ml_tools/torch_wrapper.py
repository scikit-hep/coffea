import warnings

import numpy

_torch_import_error = None
try:
    import torch
except (ImportError, ModuleNotFoundError) as err:
    _torch_import_error = err

from .helper import nonserializable_attribute, numpy_call_wrapper


class torch_wrapper(nonserializable_attribute, numpy_call_wrapper):
    """
    Wrapper for running pytorch with awkward/dask-awkward inputs.

    As torch models are not guaranteed to be serializable we load the model
    using torch save-state files. Notice that we only support TorchScript
    files for this wrapper class [1]. If the user is attempting to run on
    the clusters, the TorchScript file will need to be passed to the worker
    nodes in a way which preserves the file path.

    Once an instance `wrapper` of this class is created, it can be called on inputs
    like `wrapper(*args)`, where `args` are the inputs to `prepare_awkward` (see
    next paragraph).

    In order to actually use the class, the user must override the method
    `prepare_awkward`. The input to this method is an arbitrary number of awkward
    arrays or dask awkward arrays (but never a mix of dask/non-dask array). The
    output is two objects: a tuple `a` and a dictionary `b` such that the underlying
    `pytorch` model instance calls like `model(*a,**b)`. The contents of a and b
    should be numpy-compatible awkward-like arrays: if the inputs are non-dask awkward
    arrays, the return should also be non-dask awkward arrays that can be trivially
    converted to numpy arrays via a ak.to_numpy call; if the inputs are dask awkward
    arrays, the return should be still be dask awkward arrays that can be trivially
    converted via a to_awkward().to_numpy() call.

    [1]
    https://pytorch.org/tutorials/beginner/saving_loading_models.html#export-load-model-in-torchscript-format

    Parameters
    ----------
        torch_jit: str
            Path to the TorchScript file to load
    """

    def __init__(self, torch_jit: str):
        if _torch_import_error is not None:
            warnings.warn(
                "Users should make sure the torch package is installed before proceeding!\n"
                "> pip install torch\n"
                "or\n"
                "> conda install torch",
                UserWarning,
            )
            raise _torch_import_error

        nonserializable_attribute.__init__(self, ["model", "device"])
        self.torch_jit = torch_jit

    def _create_device(self):
        """
        Torch device run calculations on. This wrapper class will always attempt
        to use GPU if possible. Setting this as a "lazy object" so that remote
        worker can have a different configuration the interactive session.
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_model(self):
        """
        Loading in the model from the TorchScript file.

        #TODO: Move to weakref to better performance.
        """
        if torch.cuda.is_available():
            model = torch.jit.load(self.torch_jit).cuda()
        else:
            model = torch.jit.load(self.torch_jit)
        model.eval()
        return model

    def validate_numpy_input(self, *args: numpy.array, **kwargs: numpy.array) -> None:
        # Pytorch's model.parameters is not a reliable way to extract input
        # information for arbitrary models, so we will leave this to the user.
        super().validate_numpy_input(*args, **kwargs)

    def numpy_call(self, *args: numpy.array, **kwargs: numpy.array) -> numpy.array:
        """
        Evaluating the numpy inputs via the model. Returning the results also as
        as numpy array.
        """
        args = [
            (
                torch.from_numpy(arr)
                if arr.flags["WRITEABLE"]
                else torch.from_numpy(numpy.copy(arr))
            )
            for arr in args
        ]
        kwargs = {
            key: (
                torch.from_numpy(arr)
                if arr.flags["WRITEABLE"]
                else torch.from_numpy(numpy.copy(arr))
            )
            for key, arr in kwargs.items()
        }
        with torch.no_grad():
            return self.model(*args, **kwargs).detach().numpy()
