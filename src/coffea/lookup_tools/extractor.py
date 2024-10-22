import os

from coffea.lookup_tools.evaluator import evaluator
from coffea.lookup_tools.json_converters import (
    convert_correctionlib_file,
    convert_histo_json_file,
    convert_pileup_json_file,
)
from coffea.lookup_tools.root_converters import convert_histo_root_file
from coffea.lookup_tools.txt_converters import *

file_converters = {
    "root": {"default": convert_histo_root_file, "histo": convert_histo_root_file},
    "json": {
        "default": convert_histo_json_file,
        "histo": convert_histo_json_file,
        "corr": convert_correctionlib_file,
        "pileup": convert_pileup_json_file,
    },
    "txt": {
        "default": convert_jec_txt_file,
        "jec": convert_jec_txt_file,
        "jersf": convert_jersf_txt_file,
        "jr": convert_jr_txt_file,
        "junc": convert_junc_txt_file,
        "ea": convert_effective_area_file,
        "pileup": convert_pileup_json_file,
        "l5flavor": convert_l5flavor_jes_txt_file,
    },
}


class extractor:
    """
    This class defines a common entry point for defining functions that extract
    the inputs to build lookup tables from various kinds of files.

    The files that can be converted are presently defined in the "file_converters" dict.

    The file names are used to determine the converter that is used, i.e.:
        something.TYPE.FORMAT will apply the TYPE extractor to a file of given FORMAT

    If there is no file type specifier the 'default' value is used.

    The extractor class supports a number of useful file formats by default:
        - **.histo.root** : 1,2, and 3 dimensional histograms in root files.
        - **.histo.json** : N-dimensional histograms stored in JSON format.
        - **.ea.txt**     : CMS EGM effective area text files.
        - **'.[jec, jersf, jr, junc].txt'** : CMS JME jet energy corrections and systematic error text files.

    It is possible to extend the functionality of lookup_tools.extractor by editing coffea.lookup_tools.file_converters to add new types and formats.

    You can add sets of lookup tables / weights by calling:
        extractor.add_weight_set(<description>)

    <description> is formatted like '<nickname> <name-in-file> <the file to extract>'
        ``*`` can be used as a wildcard to import all available lookup tables in a file
    """

    def __init__(self):
        self._weights = []
        self._names = {}
        self._types = []
        self._filecache = {}
        self._finalized = False

    def add_weight_set(self, local_name, thetype, weights):
        """
        Adds one extracted weight to the extractor.

        Parameters
        ----------
        local_name: str
            The name of the weight.
        thetype: str
            The type of weight (eg: jme_standard_function).
        weights: Varies
            The weights themselves. Type and structure depends on thetype.
        """
        if self._finalized:
            raise Exception("extractor is finalized cannot add new weights!")
        if local_name in self._names.keys():
            raise Exception(f'weights name "{local_name}" already defined')
        self._names[local_name] = len(self._weights)
        self._types.append(thetype)
        self._weights.append(weights)

    def add_weight_sets(self, weightsdescs):
        """
        Add multiple weight sets at once, coming from one or more files.

        Parameters
        ----------
        weightsdescs: Iterable[str]
            Expects a list of text lines to be formatted as '<local name> <name> <weights file>'.
            Allows * * <file> and <prefix> * <file> to do easy imports of whole file.
        """
        for weightdesc in weightsdescs:
            if weightdesc[0] == "#":
                continue  # skip comment lines
            temp = weightdesc.strip().split(" ")
            if len(temp) != 3:
                raise Exception(
                    '"{}" not formatted as "<local name> <name> <weights file>"'.format(
                        weightdesc
                    )
                )
            (local_name, name, thefile) = tuple(temp)
            if name == "*":
                self.import_file(thefile)
                weights = self._filecache[thefile]
                for key, value in weights.items():
                    if local_name == "*":
                        self.add_weight_set(key[0], key[1], value)
                    else:
                        keyfilename, keymyname = key[0], key[1]
                        if isinstance(keyfilename, bytes):
                            keyfilename = keyfilename.decode()
                        if isinstance(keymyname, bytes):
                            keymyname = keymyname.decode()
                        self.add_weight_set(local_name + keyfilename, keymyname, value)
            else:
                weights, thetype = self.extract_from_file(thefile, name)
                self.add_weight_set(local_name, thetype, weights)
                if thetype == "json_lookup":
                    self._names[local_name] = 0

    def import_file(self, thefile):
        """
        Cache the whole contents of a file for later processing

        Parameters
        ----------
        thefile: str
            The path to the file to be imported
        """
        if thefile not in self._filecache.keys():
            drop_gz = thefile.replace(".gz", "")
            file_dots = os.path.basename(drop_gz).split(".")
            theformat = file_dots[-1].strip()
            thetype = "default"
            if len(file_dots) > 2:
                thetype = file_dots[-2]
            if "pileup" in thefile and "pileup" in file_converters[theformat]:
                thetype = "pileup"
            if "_SF_" in thefile and "jersf" in file_converters[theformat]:
                thetype = "jersf"
            if "_L5Flavor_" in thefile and "l5flavor" in file_converters[theformat]:
                thetype = "l5flavor"
            self._filecache[thefile] = file_converters[theformat][thetype](thefile)

    def extract_from_file(self, thefile, name):
        """
        Import a file and then extract a lookup set

        Parameters
        ----------
        thefile: str
            The path to the file to import
        name: str
            The name of the weights to extract, as named in the file
        """
        self.import_file(thefile)
        weights = self._filecache[thefile]
        names = {key[0]: key[1] for key in weights.keys()}
        if name not in names.keys():
            raise Exception(f'Weights named "{name}" not in {thefile}!')
        return (weights[(name, names[name])], names[name])

    def finalize(self, reduce_list=None):
        """
        Stop any further imports and, if requested, pare down
        the stored histograms to those specified in reduce_list.

        Parameters
        ----------
        reduce_list: list[str], optional
            Reduce the weights contained in this extractor to only those with names
            in reduce_list. If not provided, no such reduction takes place.
        """
        if self._finalized:
            raise Exception("extractor is already finalized!")
        del self._filecache
        if reduce_list is not None:
            names = {}
            types = []
            weights = []
            for i, name in enumerate(reduce_list):
                if name not in self._names:
                    raise Exception(f'Weights named "{name}" not in extractor!')
                names[name] = i
                types.append(self._types[self._names[name]])
                weights.append(self._weights[self._names[name]])
            self._names = names
            self._types = types
            self._weights = weights
        self._finalized = True

    def make_evaluator(self):
        """
        Produce an evaluator based on the finalized extractor

        Returns
        -------
        An evaluator based on the names, weight types, and weights of the finalized extractor.
        """
        if self._finalized:
            return evaluator(self._names, self._types, self._weights)
        else:
            raise Exception("Cannot make an evaluator from unfinalized extractor!")
