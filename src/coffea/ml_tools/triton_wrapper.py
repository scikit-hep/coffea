# For python niceties
import warnings
from typing import Dict, List, Optional

import numpy

# For triton specific handling
_triton_import_error = None
try:
    import tritonclient.grpc
    import tritonclient.http
    import tritonclient.utils
except (ImportError, ModuleNotFoundError) as err:
    _triton_import_error = err

from .helper import nonserializable_attribute, numpy_call_wrapper


class triton_wrapper(nonserializable_attribute, numpy_call_wrapper):
    """
    Wrapper for running triton inference.

    The target of this class is such that all triton specific operations are
    wrapped and abstracted-away from the users. The user should then only need
    to handle awkward-level operations to mangle the arrays into the expected
    input format required by the the model of interest. This must be done by
    overriding the `prepare_awkward` method.

    Once an instance `wrapper` of this class is created, it can be called on inputs
    like `wrapper(*args)`, where `args` are the inputs to `prepare_awkward` (see
    next paragraph).

    In order to actually use the class, the user must override the method
    `prepare_awkward`. The input to this method is an arbitrary number of awkward
    arrays or dask awkward arrays (but never a mix of dask/non-dask array). The
    output is two objects: a tuple `a` and a dictionary `b` such that the underlying
    `tritonclient` instance calls like `client(*a,**b)`. The contents of a and b
    should be numpy-compatible awkward-like arrays: if the inputs are non-dask awkward
    arrays, the return should also be non-dask awkward arrays that can be trivially
    converted to numpy arrays via a ak.to_numpy call; if the inputs are dask awkward
    arrays, the return should be still be dask awkward arrays that can be trivially
    converted via a to_awkward().to_numpy() call.

    Parameters
    ----------
        model_url: str
            A string in the format of: `triton+<protocol>://<address>/<model>/<version>`

        client_args: dict[str,str], optional
            Optional keyword arguments to pass to the underlying `InferenceServerClient` objects.

        batch_size: int, default -1
            How the input arrays should be split up for analysis processing. Leave negative to
            have this automatically resolved.
    """

    batch_size_fallback = 10  # Fall back should batch size not be determined.
    http_client_concurrency = 12  # TODO: check whether this value is optimum

    def __init__(
        self, model_url: str, client_args: Optional[Dict] = None, batch_size=-1
    ):
        if _triton_import_error is not None:
            warnings.warn(
                "Users should make sure the tritonclient package is installed before proceeding!\n"
                "> pip install tritonclient[grpc,http]\n"
                "or\n"
                "> conda install tritonclient[grpc,http]",
                UserWarning,
            )
            raise _triton_import_error

        nonserializable_attribute.__init__(
            self, ["client", "model_metadata", "model_inputs", "model_outputs"]
        )

        fullprotocol, location = model_url.split("://")
        _, self.protocol = fullprotocol.split("+")
        self.address, self.model, self.version = location.split("/")

        # Additional pseudo-lazy objects that requires additional parsing after
        # lazy objects have been initialized or additional parsing.
        self._batch_size = batch_size
        self._client_args = client_args

    """
    Spawning the unserializable triton client, as well as other helper objects
    that require the triton client to be present.
    """

    @property
    def pmod(self):
        """Getting the protocol module based on the url protocol string."""
        if self.protocol == "grpc":
            return tritonclient.grpc
        elif self.protocol == "http":
            return tritonclient.http
        else:
            raise ValueError(
                f"{self.protocol} does not encode a valid protocol (grpc or http)"
            )

    def _create_client(self):
        return self.pmod.InferenceServerClient(url=self.address, **self.client_args)

    @property
    def client_args(self) -> Dict:
        """
        Function for adding default arguments to the client constructor kwargs.
        """
        if self.protocol == "grpc":
            kwargs = dict(verbose=False, ssl=True)
        elif self.protocol == "http":
            kwargs = dict(verbose=False, concurrency=self.http_client_concurrency)
        if self._client_args is not None:
            kwargs.update(self._client_args)
        return kwargs

    def _create_model_metadata(self) -> Dict:
        return self.client.get_model_metadata(self.model, self.version, as_json=True)

    def _create_model_inputs(self) -> Dict[str, Dict]:
        """
        Extracting the model input data formats from the model_metatdata. Here
        we slightly change the input formats the objects in a format that is
        easier to manipulate and compare with numpy arrays.
        """
        return {
            x["name"]: {
                "shape": tuple(int(i) for i in x["shape"]),
                "datatype": x["datatype"],
            }
            for x in self.model_metadata["inputs"]
        }

    def _create_model_outputs(self) -> List[int]:
        """Getting a list of names of possible outputs"""
        return [x["name"] for x in self.model_metadata["outputs"]]

    @property
    def batch_size(self) -> int:
        """
        Getting the batch size to be used for array splitting. If it is
        explicitly set by the users, use that; otherwise, extract from the model
        configuration hosted on the server.
        """
        if self._batch_size < 0:
            model_config = self.client.get_model_config(
                self.model, self.version, as_json=True
            )["config"]
            if "dynamic_batching" in model_config:
                self._batch_size = model_config["dynamic_batching"][
                    "preferred_batch_size"
                ][0]
            elif "max_batch_size" in model_config:
                self._batch_size = model_config["max_batch_size"]
            else:
                warnings.warn(
                    f"Batch size not set by model! Setting to default value {self.batch_size_fallback}. "
                    "Contact model maintainer to check if this is expected",
                    UserWarning,
                )
                self._batch_size = self.batch_size_fallback

        return self._batch_size

    """
    Numpy/awkward/dask_awkward inference
    """

    def validate_numpy_input(
        self, output_list: List[str], input_dict: Dict[str, numpy.array]
    ) -> None:
        """
        Check that tritonclient can return the expected input array dimensions and
        available output values. Can be useful when ensuring that data is being properly
        mangled for Triton. This method is called just before passing to the Triton client
        when an inference request is made.

        If no errors are raised, it is understood that the input is validated by this function.

        Parameters
        ----------
            output_list: list[str]
                List of string corresponding to the name of the outputs
                of interest. These strings will be automatically translated into the
                required `tritonclient.InferRequestedOutput` objects. This is identical
                to the first argument the user passes in when calling the `triton_wrapper`
                instance.

            input_dict: dict[str,np.array]
                Dictionary with the model's input-names as the key and the
                appropriate numpy array as the dictionary value. This dictionary is
                automatically translated into a list of `tritonclient.InferInput`
                objects. This is identical to the second argument the user passes in when
                calling the `triton_wrapper` instance.
        """
        # Input value checking
        for iname, iarr in input_dict.items():
            # Checking the name
            if iname not in self.model_inputs.keys():
                raise ValueError(
                    f"Input [{iname}] not defined in model! "
                    f"Inputs defined by model: {[x for x in self.model_inputs.keys()]}"
                )
            # Checking the shape
            ishape = numpy.array(iarr.shape)
            mshape = numpy.array(self.model_inputs[iname]["shape"])
            if len(ishape) != len(mshape):
                raise ValueError(
                    f"Input [{iname}] got wrong dimension: {len(ishape)} "
                    f"(Expected {len(mshape)})"
                )
            if not all(numpy.where(mshape > 0, ishape == mshape, True)):
                raise ValueError(
                    f"Input [{iname}] got array of shape {ishape} "
                    f"(Expected: {mshape}, -1 means arbitrary)"
                )
            # Checking data type. Notice that this will only raise a warning! Data
            # type defined by triton can be found here:
            # https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#datatypes
            itype = iarr.dtype
            mtype = tritonclient.utils.triton_to_np_dtype(
                self.model_inputs[iname]["datatype"]
            )
            if itype != mtype:
                warnings.warn(
                    f"Input [{iname}] got array of type [{itype}] (Expected [{mtype.__name__}])."
                    " Automatic conversion will be performed using numpy.array.astype.",
                    UserWarning,
                )

        # Checking for missing inputs
        for mname in self.model_inputs.keys():
            if mname not in input_dict.keys():
                raise ValueError(f"Input [{mname}] not given in input dictionary!")

        # Checking output
        for oname in output_list:
            if oname not in self.model_outputs:
                raise ValueError(
                    f"Requested output [{oname}] not defined by model (Defined: {[x for x in self.model_outputs]})"
                )

    def numpy_call(
        self, output_list: List[str], input_dict: Dict[str, numpy.array]
    ) -> Dict[str, numpy.array]:
        """
        Parameters
        ----------
            output_list: list[str]
                List of string corresponding to the name of the outputs
                of interest. These strings will be automatically translated into the
                required `tritonclient.InferRequestedOutput` objects.

            input_dict: dict[str,np.array]
                Dictionary with the model's input-names as the key and the
                appropriate numpy array as the dictionary value. This dictionary is
                automatically translated into a list of `tritonclient.InferInput`
                objects.


        Returns
        -------
            dict[str,np.array]
                The return will be the dictionary of numpy arrays that have the
                output_list arguments as keys.
        """

        # Setting up the inference input containers
        def _get_infer_shape(name):
            ishape = numpy.array(input_dict[name].shape)
            mshape = numpy.array(self.model_inputs[name]["shape"])
            mshape = numpy.where(mshape < 0, ishape, mshape)
            mshape[0] = self.batch_size
            return mshape

        infer_inputs = [
            self.pmod.InferInput(name, _get_infer_shape(name), prop["datatype"])
            for name, prop in self.model_inputs.items()
        ]

        # Setting up the inference output containers
        infer_outputs = [
            self.pmod.InferRequestedOutput(output) for output in output_list
        ]

        # Setting up container for storing output.
        output = None

        # Padding the outermost dimension to a multiple of of the batch size
        orig_len = list(input_dict.values())[0].shape[0]  # saving original length
        for start_idx in range(0, orig_len, self.batch_size):
            stop_idx = min([start_idx + self.batch_size, orig_len])

            for idx, name in enumerate(self.model_inputs.keys()):
                mtype = tritonclient.utils.triton_to_np_dtype(
                    self.model_inputs[name]["datatype"]
                )
                shape = list(input_dict[name].shape)
                shape[0] = self.batch_size  # Always pad to fixed length
                infer_inputs[idx].set_data_from_numpy(
                    numpy.resize(
                        input_dict[name][start_idx:stop_idx],  # We need a copy here
                        tuple(shape),
                    ).astype(mtype)
                )

            # Making request to server
            request = self.client.infer(
                self.model,
                model_version=self.version,
                inputs=infer_inputs,
                outputs=infer_outputs,
            )
            if output is None:
                output = {
                    o: request.as_numpy(o)[start_idx:stop_idx] for o in output_list
                }
            else:
                for o in output_list:
                    output[o] = numpy.concatenate(
                        (output[o], request.as_numpy(o)), axis=0
                    )
        return {k: v[:orig_len] for k, v in output.items()}
