import warnings

import numpy

_tf_import_error = None
try:
    import tensorflow
except (ImportError, ModuleNotFoundError) as err:
    _tf_import_error = err

from .helper import nonserializable_attribute, numpy_call_wrapper


class tf_wrapper(nonserializable_attribute, numpy_call_wrapper):
    """
    Wrapper for running tensorflow inference with awkward/dask-awkward inputs.
    """

    def __init__(self, tf_model: str, skip_length_zero: bool = False):
        """
        As models are not guaranteed to be directly serializable, the use will
        need to pass the model as files saved using the `tf.keras.save` method
        [1]. If the user is attempting to run on the clusters, the model file
        will need to be passed to the worker nodes in a way which preserves the
        file path.

        [1]
        https://www.tensorflow.org/guide/keras/serialization_and_saving#saving

        Parameters
        ----------

        tf_model:
            Path to the tensorflow model file for computation
        skip_length_zero:
            Generating a default length 0 numpy array if the input array is
            detected to be length-0 instead of passing it into the tensorflow
            model. This option should only be used if the model uses tensorflow
            functions that does not properly implement behaviors for length 0
            inputs.
        """
        if _tf_import_error is not None:
            warnings.warn(
                "Users should make sure the torch package is installed before proceeding!\n"
                "> pip install tensorflow\n"
                "or\n"
                "> conda install tensorflow",
                UserWarning,
            )
            raise _tf_import_error

        nonserializable_attribute.__init__(self, ["model"])
        self.tf_model = tf_model
        self.skip_length_zero = skip_length_zero

    def _create_model(self):
        """
        Loading in the model from the model file. We simply rely on Tensorflow
        to automatically load the accelerator resources.
        # TODO: More control over accelerator resources?
        """
        return tensorflow.keras.models.load_model(self.tf_model)

    def validate_numpy_input(self, *args: numpy.array, **kwargs: numpy.array) -> None:
        """
        Here we are assuming that the model contains the required information
        for parsing the input numpy array(s), and that the input numpy array(s)
        is the first argument of the user method call.
        """
        model_input = self.model.input_shape
        input_arr = args[0]  # Getting the input array

        def _equal_shape(mod_in: tuple, arr_shape: tuple) -> None:
            """Tuple of input shape and array shape"""
            assert len(mod_in) == len(
                arr_shape
            ), f"Mismatch number of axis (model: {mod_in}; received: {arr_shape})"
            match = [
                (m == a if m is not None else True) for m, a in zip(mod_in, arr_shape)
            ]
            assert numpy.all(
                match
            ), f"Mismatch shape (model: {mod_in}; received: {arr_shape})"

        if isinstance(model_input, tuple):
            # Single input model
            _equal_shape(model_input, input_arr.shape)
        else:
            assert len(input_arr) == len(
                model_input
            ), f"Mismatch number of inputs (model: {len(model_input)}; received: {len(input_arr)})"
            for model_shape, arr in zip(model_input, input_arr):
                _equal_shape(model_shape, arr.shape)

    def numpy_call(self, *args: numpy.array, **kwargs: numpy.array) -> numpy.array:
        """
        Evaluating the numpy inputs via the `model.__call__` method. With a
        trivial conversion for tensors for the numpy inputs.

        TODO: Do we need to evaluate using `predict` [1]? Since array batching
        is already handled by dask.

        [1]
        https://keras.io/getting_started/faq/#whats-the-difference-between-model-methods-predict-and-call
        """
        first_arg = args[0] if len(args) else next(iter(kwargs.values()))
        if len(first_arg) == 0 and self.skip_length_zero:
            return numpy.zeros(shape=(0, *self.model.output_shape[1:]))
        args = [
            (
                tensorflow.convert_to_tensor(arr)
                if arr.flags["WRITEABLE"]
                else tensorflow.convert_to_tensor(numpy.copy(arr))
            )
            for arr in args
        ]
        kwargs = {
            key: (
                tensorflow.convert_to_tensor(arr)
                if arr.flags["WRITABLE"]
                else tensorflow.convert_to_tensor(numpy.copy(arr))
            )
            for key, arr in kwargs.items()
        }
        return self.model(*args, **kwargs).numpy()
