from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
from coffea.lookup_tools.dense_evaluated_lookup import dense_evaluated_lookup
from coffea.lookup_tools.dense_lookup import dense_lookup
from coffea.lookup_tools.jec_uncertainty_lookup import jec_uncertainty_lookup
from coffea.lookup_tools.jersf_lookup import jersf_lookup
from coffea.lookup_tools.jme_standard_function import jme_standard_function
from coffea.lookup_tools.rochester_lookup import rochester_lookup

lookup_types = {
    "dense_lookup": dense_lookup,
    "dense_evaluated_lookup": dense_evaluated_lookup,
    "jme_standard_function": jme_standard_function,
    "jersf_lookup": jersf_lookup,
    "jec_uncertainty_lookup": jec_uncertainty_lookup,
    "rochester_lookup": rochester_lookup,
    "correctionlib_wrapper": correctionlib_wrapper,
}


class evaluator:
    """
    The evaluator class serves as a single point of entry for
    looking up values of histograms and other functions read in
    with the extractor class. Stored look ups can be indexed by
    name and then called through an overloaded __call__ function.

    Example::

        #assuming 'eta' and 'pt' are array like objects
        wgts = "testSF2d scalefactors_Tight_Electron tests/samples/testSF2d.histo.root"
        extractor.add_weight_sets([wgts])
        extractor.finalize(reduce_list=['testSF2d'])
        evaluator = extractor.make_evaluator()
        out = evaluator["testSF2d"](eta, pt)

    The returned value has the same shape as the input arguments. `lookup_types` is a map of possible
    constructors for extracted data. The arguments used when calling the evaluator depend on which named
    weight is being used (eg. in the above example, the "testSF2d" weight requires `eta` and `pt` be
    passed when calling the evaluator).

    It is recommended to construct an evaluator from an extractor, so ensure that inputs to the
    constructor are properly ordered and formatted.

    Parameters
    ----------
    names: dict[str, int]
        A dictionary mapping the names of weights to the index of that weight in `primitives`.
    types: list[str]
        A list of the types of weights, ordered in the same way as `primitives`.
    primitives: list[Varies]
        A list of primitives, whose type and structure depend on types. Should be order in the
        same way as `primitives`.
    """

    def __init__(self, names, types, primitives):
        """
        initialize the evaluator from a list of inputs names,
        lookup types, and input primitive data
        """
        self._functions = {}
        for key in names.keys():
            lookup_type = types[names[key]]
            lookup_def = primitives[names[key]]
            self._functions[key] = lookup_types[lookup_type](*lookup_def)

    def __dir__(self):
        """dir is overloaded to list all available functions
        in the evaluator
        """
        return self._functions.keys()

    def __getitem__(self, key):
        """return a function named 'key'"""
        return self._functions[key]

    def keys(self):
        """returns the available functions"""
        return self._functions.keys()

    def __contains__(self, item):
        """item in X"""
        return item in self._functions
