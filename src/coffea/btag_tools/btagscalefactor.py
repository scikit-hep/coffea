import numpy
import pandas

from coffea.lookup_tools.dense_mapped_lookup import dense_mapped_lookup


class BTagScaleFactor:
    """A class holding one complete BTag scale factor for a given working point

    Parameters
    ----------
        filename : str
            The BTag-formatted CSV file to read (accepts .csv, .csv.gz, etc.)
            See pandas read_csv for all supported compressions.
        workingpoint : str or int
            The working point, one of LOOSE, MEDIUM, TIGHT, or RESHAPE (0-3, respectively)
        methods : str, optional
            The scale factor derivation method to use for each flavor, 'b,c,light' respectively.
            Defaults to 'comb,comb,incl'
        keep_df : bool, optional
            If set true, keep the parsed dataframe as an attribute (.df) for later inspection

    Attributes
    ----------
        LOOSE: int
            Value is 0. This is the integer for the loose WP
        MEDIUM: int
            Value is 1. This is the integer for the medium WP
        TIGHT: int
            Value is 2. This is the integer for the tight WP
        RESHAPE: int
            Value is 3. This is the integer for the reshape WP
        FLAV_B: int
            Value is 0. This is the integer to represent the b flavor. Input choice to some methods.
        FLAV_C: int
            Value is 1. This is the integer to represent the c flavor. Input choice to some methods.
        FLAV_UDSG: int
            Value is 2. This is the integer to represent u, d, and s flavors, as well as gluons. Input choice to some methods.
    """

    LOOSE, MEDIUM, TIGHT, RESHAPE = range(4)
    FLAV_B, FLAV_C, FLAV_UDSG = range(3)
    _flavor = numpy.array([0, 4, 5, 6])
    _flavor2btvflavor = {0: FLAV_UDSG, 4: FLAV_C, 5: FLAV_B}
    _wpString = {"loose": LOOSE, "medium": MEDIUM, "tight": TIGHT, "reshape": RESHAPE}
    _expectedColumns = [
        "OperatingPoint",
        "measurementType",
        "sysType",
        "jetFlavor",
        "etaMin",
        "etaMax",
        "ptMin",
        "ptMax",
        "discrMin",
        "discrMax",
        "formula",
    ]

    @classmethod
    def readcsv(cls, filename):
        """Reads a BTag-formmated CSV file into a pandas dataframe

        This function also merges the bin min and max into a tuple representing the bin

        Parameters
        ----------
            filename : str
                The file to open

        Returns
        -------
            df : pandas.DataFrame
                A dataframe containing all info in the file
            discriminator : str
                The name of the discriminator the correction map is for
        """
        df = pandas.read_csv(filename, skipinitialspace=True)
        discriminator = df.columns[0].split(";")[0]

        def cleanup(colname):
            if ";" in colname:
                _, colname = colname.split(";")
            return colname.strip()

        df.rename(columns=cleanup, inplace=True)
        if not list(df.columns) == BTagScaleFactor._expectedColumns:
            raise RuntimeError(
                "Columns in BTag scale factor file %s as expected" % filename
            )
        for var in ["eta", "pt", "discr"]:
            df[var + "Bin"] = list(zip(df[var + "Min"], df[var + "Max"]))
            del df[var + "Min"]
            del df[var + "Max"]
        return df, discriminator

    def __init__(self, filename, workingpoint, methods="comb,comb,incl", keep_df=False):
        if workingpoint not in [0, 1, 2, 3]:
            try:
                workingpoint = BTagScaleFactor._wpString[workingpoint.lower()]
            except (KeyError, AttributeError):
                raise ValueError("Unrecognized working point")
        methods = methods.split(",")
        self.workingpoint = workingpoint
        df, self.discriminator = BTagScaleFactor.readcsv(filename)
        if set(df["OperatingPoint"].unique()).intersection({"L", "M", "T"}):
            raise RuntimeError(
                f"The BTag csv file {filename} is in the new UL format which is not supported by coffea.btag_tools.\n"
                "Instead one can use correctionlib for UL scale factors."
            )
        cut = (df["jetFlavor"] == self.FLAV_B) & (df["measurementType"] == methods[0])
        if len(methods) > 1:
            cut |= (df["jetFlavor"] == self.FLAV_C) & (
                df["measurementType"] == methods[1]
            )
        if len(methods) > 2:
            cut |= (df["jetFlavor"] == self.FLAV_UDSG) & (
                df["measurementType"] == methods[2]
            )
        cut &= df["OperatingPoint"] == workingpoint
        df = df[cut]
        mavailable = list(df["measurementType"].unique())
        if not all(m in mavailable for m in methods):
            raise ValueError(
                "Unrecognized jet correction method, available: %r" % mavailable
            )
        df = df.set_index(
            ["sysType", "jetFlavor", "etaBin", "ptBin", "discrBin"]
        ).sort_index()
        if keep_df:
            self.df = df
        self._corrections = {}
        for syst in list(df.index.levels[0]):
            corr = df.loc[syst]
            allbins = list(corr.index)
            edges_eta = numpy.array(
                sorted({x for tup in corr.index.levels[1] for x in tup})
            )
            if numpy.all(edges_eta >= 0):
                assert (
                    edges_eta[0] == 0.0
                ), "BTV correction doesn't cover the middle of the detector!"
                edges_eta = numpy.concatenate([-edges_eta[:0:-1], edges_eta])
            edges_pt = numpy.array(
                sorted({x for tup in corr.index.levels[2] for x in tup})
            )
            edges_discr = numpy.array(
                sorted({x for tup in corr.index.levels[3] for x in tup})
            )
            bin_low_edges = numpy.meshgrid(
                self._flavor[:-1],
                edges_eta[:-1],
                edges_pt[:-1],
                edges_discr[:-1],
                indexing="ij",
            )
            mapping = numpy.full(bin_low_edges[0].shape, -1)

            def findbin(flavor, eta, pt, discr):
                btvflavor = self._flavor2btvflavor[flavor]
                for i, (fbin, ebin, pbin, dbin) in enumerate(allbins):
                    if (
                        btvflavor == fbin
                        and ebin[0] <= eta < ebin[1]
                        and pbin[0] <= pt < pbin[1]
                        and dbin[0] <= discr < dbin[1]
                    ):
                        return i
                if eta < 0:
                    # maybe in this region we have only abseta
                    for i, (fbin, ebin, pbin, dbin) in enumerate(allbins):
                        if (
                            btvflavor == fbin
                            and -ebin[1] <= eta < -ebin[0]
                            and pbin[0] <= pt < pbin[1]
                            and dbin[0] <= discr < dbin[1]
                        ):
                            return i
                return -1

            for idx, _ in numpy.ndenumerate(mapping):
                flavor, eta, pt, discr = (x[idx] for x in bin_low_edges)
                mapping[idx] = findbin(flavor, eta, pt, discr)

            if self.workingpoint == BTagScaleFactor.RESHAPE:
                self._corrections[syst] = dense_mapped_lookup(
                    (self._flavor, edges_eta, edges_pt, edges_discr),
                    mapping,
                    numpy.array(corr["formula"]),
                    3,
                )
            else:
                self._corrections[syst] = dense_mapped_lookup(
                    (self._flavor, edges_eta, edges_pt),
                    mapping[..., 0],
                    numpy.array(corr["formula"]),
                    2,
                )

    def eval(self, systematic, flavor, eta, pt, discr=None, ignore_missing=False):
        """Evaluate this scale factor as a function of jet properties

        Parameters
        ----------
            systematic : str
                Which systematic to evaluate. Nominal correction is 'central', the options depend
                on the scale factor and method
            flavor : numpy.ndarray or awkward.Array
                The generated jet hadron flavor, following the enumeration:
                0: uds quark or gluon, 4: charm quark, 5: bottom quark
            eta : numpy.ndarray or awkward.Array
                The jet pseudorapitiy
            pt : numpy.ndarray or awkward.Array
                The jet transverse momentum
            discr : numpy.ndarray or awkward.Array, optional
                The jet tagging discriminant value (default None), optional for all scale factors except
                the reshaping scale factor
            ignore_missing : bool, optional
                If set true, any values that have no correction will return 1. instead of throwing
                an exception. Out-of-bounds values are always clipped to the nearest bin.

        Returns
        -------
            out : numpy.ndarray or awkward.Array
                An array with shape matching ``pt``, containing the per-jet scale factor
        """
        if systematic not in self._corrections:
            raise ValueError("Unrecognized systematic: %s" % systematic)
        if self.workingpoint == BTagScaleFactor.RESHAPE:
            if discr is None:
                raise ValueError("RESHAPE scale factor requires a discriminant array")
            return self._corrections[systematic](
                flavor, eta, pt, discr, ignore_missing=ignore_missing
            )
        else:
            return self._corrections[systematic](
                flavor, eta, pt, ignore_missing=ignore_missing
            )

    def __call__(self, systematic, flavor, eta, pt, discr=None, ignore_missing=False):
        return self.eval(systematic, flavor, eta, pt, discr, ignore_missing)
