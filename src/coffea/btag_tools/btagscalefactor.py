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

    Notes
    -----
    Class constants for working points: LOOSE (0), MEDIUM (1), TIGHT (2), RESHAPE (3)

    Class constants for flavors: FLAV_B (0), FLAV_C (1), FLAV_UDSG (2)
    """

    # Working point constants
    LOOSE, MEDIUM, TIGHT, RESHAPE = range(4)
    # Flavor constants
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

            fbins = numpy.array([b[0] for b in allbins])
            eta_lo = numpy.array([b[1][0] for b in allbins])
            eta_hi = numpy.array([b[1][1] for b in allbins])
            pt_lo = numpy.array([b[2][0] for b in allbins])
            pt_hi = numpy.array([b[2][1] for b in allbins])
            discr_lo = numpy.array([b[3][0] for b in allbins])
            discr_hi = numpy.array([b[3][1] for b in allbins])

            eta_cell = bin_low_edges[1].reshape(-1, 1)
            pt_cell = bin_low_edges[2].reshape(-1, 1)
            discr_cell = bin_low_edges[3].reshape(-1, 1)
            btvflavor_cell = numpy.empty(bin_low_edges[0].size, dtype=fbins.dtype)
            for flav, btv in self._flavor2btvflavor.items():
                btvflavor_cell[bin_low_edges[0].reshape(-1) == flav] = btv
            btvflavor_cell = btvflavor_cell.reshape(-1, 1)

            base_match = (
                (btvflavor_cell == fbins)
                & (pt_lo <= pt_cell)
                & (pt_cell < pt_hi)
                & (discr_lo <= discr_cell)
                & (discr_cell < discr_hi)
            )
            match = base_match & (eta_lo <= eta_cell) & (eta_cell < eta_hi)
            found = match.any(axis=1)
            result = numpy.where(found, match.argmax(axis=1), -1)

            # maybe in this region we have only abseta
            abseta_match = base_match & (-eta_hi <= eta_cell) & (eta_cell < -eta_lo)
            fallback = (~found) & (eta_cell[:, 0] < 0)
            fallback &= abseta_match.any(axis=1)
            result[fallback] = abseta_match.argmax(axis=1)[fallback]

            mapping = result.reshape(mapping.shape)

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
