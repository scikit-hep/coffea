import os
import sys
import warnings

import awkward
import numpy

try:
    import cStringIO as io
except ImportError:
    import io

# for later
# func = numbaize(formula,['p%i'%i for i in range(nParams)]+[varnames[i] for i in range(nEvalVars)])


def is_gz_file(filename):
    with open(filename, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


def _parse_jme_formatted_file(
    jmeFilePath, interpolatedFunc=False, paramsFromColumns=False, jme_f=None
):
    if jme_f is None:
        fopen = open
        fmode = "rt"
        if is_gz_file(jmeFilePath):
            import gzip

            fopen = gzip.open
            fmode = (
                "r"
                if sys.platform.startswith("win") and sys.version_info.major < 3
                else fmode
            )
        jme_f = fopen(jmeFilePath, fmode)
    layoutstr = jme_f.readline().strip().strip("{}")

    name = jmeFilePath.split("/")[-1].split(".")[0]

    layout = layoutstr.split()
    if not layout[0].isdigit():
        raise Exception("First column of JME File Header must be a digit!")

    # setup the file format
    nBinnedVars = int(layout[0])
    nBinColumns = 2 * nBinnedVars
    nEvalVars = int(layout[nBinnedVars + 1])
    formula = layout[nBinnedVars + nEvalVars + 2]
    nParams = 0
    while formula.count("[%i]" % nParams):
        formula = formula.replace("[%i]" % nParams, "p%i" % nParams)
        nParams += 1
    # get rid of TMath
    tmath = {
        "TMath::Max": "max",
        "TMath::Log": "log",
        "TMath::Power": "pow",
        "TMath::Erf": "erf",
    }
    for key, rpl in tmath.items():
        formula = formula.replace(key, rpl)
    # protect function names with vars in them
    funcs_to_cap = ["max", "exp", "pow"]

    # parse the columns
    minMax = ["Min", "Max"]
    columns = []
    dtypes = []
    offset = 1
    prefix = "" if paramsFromColumns else "binned_"
    for i in range(nBinnedVars):
        columns.extend([f"{prefix}{layout[i + offset]}{mm}" for mm in minMax])
        dtypes.extend(["<f4", "<f4"])
    columns.append("NVars")
    dtypes.append("<i8")
    offset += nBinnedVars + 1
    prefix = "" if paramsFromColumns else "eval_"
    if not interpolatedFunc:
        for i in range(nEvalVars):
            columns.extend([f"{prefix}{layout[i + offset]}{mm}" for mm in minMax])
            dtypes.extend(["<f4", "<f4"])
    for i in range(nParams):
        columns.append("p%i" % i)
        dtypes.append("<f4")

    for f in funcs_to_cap:
        formula = formula.replace(f, f.upper())

    templatevars = ["x", "y", "z", "t", "w", "s"]
    varnames = [layout[i + nBinnedVars + 2] for i in range(nEvalVars)]
    for find, replace in zip(templatevars, varnames):
        formula = formula.replace(find, replace.upper())
        funcs_to_cap.append(replace)
    # restore max
    for f in funcs_to_cap:
        formula = formula.replace(f.upper(), f)

    if paramsFromColumns:
        pars = numpy.genfromtxt(jme_f, encoding="ascii")
        if len(pars.shape) == 1:
            pars = pars[numpy.newaxis, :]
        nParams = pars.shape[1] - len(columns)
        for i in range(nParams):
            columns.append("p%i" % i)
            dtypes.append("<f4")
        pars = numpy.core.records.fromarrays(
            pars.transpose(), names=columns, formats=dtypes
        )
    else:
        pars = numpy.genfromtxt(
            jme_f,
            dtype=tuple(dtypes),
            names=tuple(columns),
            encoding="ascii",
        )
        if len(pars.shape) == 0:
            pars = pars[numpy.newaxis]

    outs = [
        name,
        layout,
        pars,
        nBinnedVars,
        nBinColumns,
        nEvalVars,
        formula,
        nParams,
        columns,
        dtypes,
    ]
    jme_f.close()
    return tuple(outs)


def _build_standard_jme_lookup(
    name,
    layout,
    pars,
    nBinnedVars,
    nBinColumns,
    nEvalVars,
    formula,
    nParams,
    columns,
    dtypes,
    interpolatedFunc=False,
):
    # the first bin is always usual for JECs
    # the next bins may vary in number, so they're jagged arrays... yay
    bins = {}
    offset_col = 0
    offset_name = 1
    bin_order = []
    for i in range(nBinnedVars):
        binMins = None
        binMaxs = None
        if i == 0:
            binMins = numpy.unique(pars[columns[0]])
            binMaxs = numpy.unique(pars[columns[1]])
            if numpy.all(binMins[1:] == binMaxs[:-1]):
                bins[layout[i + offset_name]] = numpy.union1d(binMins, binMaxs)
            else:
                warnings.warn(
                    "binning for file for %s is malformed in variable %s"
                    % (name, layout[i + offset_name])
                )
                bins[layout[i + offset_name]] = numpy.union1d(binMins, binMaxs[-1:])
        else:
            counts = numpy.zeros(0, dtype=numpy.int64)
            allBins = numpy.zeros(0, dtype=numpy.double)
            for binMin in bins[bin_order[0]][:-1]:
                binMins = numpy.unique(
                    pars[numpy.where(pars[columns[0]] == binMin)][
                        columns[i + offset_col]
                    ]
                )
                binMaxs = numpy.unique(
                    pars[numpy.where(pars[columns[0]] == binMin)][
                        columns[i + offset_col + 1]
                    ]
                )
                theBins = None
                if numpy.all(binMins[1:] == binMaxs[:-1]):
                    theBins = numpy.union1d(binMins, binMaxs)
                else:
                    warnings.warn(
                        "binning for file for %s is malformed in variable %s"
                        % (name, layout[i + offset_name])
                    )
                    theBins = numpy.union1d(binMins, binMaxs[-1:])
                allBins = numpy.append(allBins, theBins)
                counts = numpy.append(counts, theBins.size)
            bins[layout[i + offset_name]] = awkward.to_packed(
                awkward.unflatten(allBins, counts)
            )
        bin_order.append(layout[i + offset_name])
        offset_col += 1

    # skip nvars to the variable columns
    # the columns here define clamps for the variables defined in columns[]
    # ----> clamps can be different from bins
    # ----> if there is more than one binning variable this array is jagged
    # ----> just make it jagged all the time
    clamp_mins = {}
    clamp_maxs = {}
    var_order = []
    offset_col = 2 * nBinnedVars + 1
    offset_name = nBinnedVars + 2
    jagged_counts = numpy.ones(bins[bin_order[0]].size - 1, dtype=numpy.int64)
    if len(bin_order) > 1:
        jagged_counts = numpy.maximum(
            awkward.num(bins[bin_order[1]]) - 1, 0
        )  # need counts-1 since we only care about Nbins
    for i in range(nEvalVars):
        var_order.append(layout[i + offset_name])
        if not interpolatedFunc:
            clamp_mins[layout[i + offset_name]] = awkward.to_packed(
                awkward.unflatten(
                    numpy.atleast_1d(pars[columns[i + offset_col]]), jagged_counts
                )
            )
            clamp_maxs[layout[i + offset_name]] = awkward.to_packed(
                awkward.unflatten(
                    numpy.atleast_1d(pars[columns[i + offset_col + 1]]), jagged_counts
                )
            )
            assert awkward.is_valid(clamp_mins[layout[i + offset_name]])
            assert awkward.is_valid(clamp_maxs[layout[i + offset_name]])
            offset_col += 1

    # now get the parameters, which we will look up with the clamped values
    params = []
    param_order = []
    offset_col = 2 * nBinnedVars + 1 + int(not interpolatedFunc) * 2 * nEvalVars
    for i in range(nParams):
        jag = awkward.to_packed(
            awkward.unflatten(pars[columns[i + offset_col]], jagged_counts)
        )
        assert awkward.is_valid(jag)
        params.append(jag)
        param_order.append("p%i" % (i))

    wrapped_up = {}
    wrapped_up[(name, "jme_standard_function")] = (
        formula,
        (bins, bin_order),
        (clamp_mins, clamp_maxs, var_order),
        (params, param_order),
    )
    return wrapped_up


def _convert_standard_jme_txt_file(jmeFilePath):
    return _build_standard_jme_lookup(*_parse_jme_formatted_file(jmeFilePath))


convert_jec_txt_file = _convert_standard_jme_txt_file
convert_jr_txt_file = _convert_standard_jme_txt_file


def convert_jersf_txt_file(jersfFilePath):
    (
        name,
        layout,
        pars,
        nBinnedVars,
        nBinColumns,
        nEvalVars,
        formula,
        nParams,
        columns,
        dtypes,
    ) = _parse_jme_formatted_file(jersfFilePath, paramsFromColumns=True)

    temp = _build_standard_jme_lookup(
        name,
        layout,
        pars,
        nBinnedVars,
        nBinColumns,
        nEvalVars,
        formula,
        nParams,
        columns,
        dtypes,
    )
    wrapped_up = {}
    for key, val in temp.items():
        newkey = (key[0], "jersf_lookup")
        vallist = list(val)
        vals, names = vallist[-1]
        if len(vals) > 3:
            warnings.warn(
                "JERSF file is in the new format with split-out systematic, only parsing totals!!!"
            )
            vals = vals[:3]
        names = ["central-up-down"]
        central, down, up = vals
        vallist[-1] = ((central, up, down), names)
        wrapped_up[newkey] = tuple(vallist)

    return wrapped_up


def convert_l5flavor_jes_txt_file(juncFilePath):
    components = []
    basename = os.path.basename(juncFilePath).split(".")[0]
    fopen = open
    fmode = "rt"
    if is_gz_file(juncFilePath):
        import gzip

        fopen = gzip.open
        fmode = (
            "r"
            if sys.platform.startswith("win") and sys.version_info.major < 3
            else fmode
        )
    with fopen(juncFilePath, fmode) as uncfile:
        for line in uncfile:
            if line.startswith("#"):
                continue
            elif line.startswith("["):
                component_name = line.strip()[1:-1]  # remove leading and trailing []
                cname = f"{basename}_{component_name}.txt"
                components.append((cname, []))
            elif components:
                components[-1][1].append(line)
            else:
                continue
    if not components:  # there are no components in the file
        components.append((juncFilePath, None))
    else:
        components = [(i, io.StringIO("".join(j))) for i, j in components]

    retval = {}
    #     return components
    for name, ifile in components:
        wrapped_up = _build_standard_jme_lookup(
            *_parse_jme_formatted_file(
                name, interpolatedFunc=False, paramsFromColumns=False, jme_f=ifile
            )
        )
        retval.update(wrapped_up)
    return retval


def convert_junc_txt_file(juncFilePath):
    components = []
    basename = os.path.basename(juncFilePath).split(".")[0]
    fopen = open
    fmode = "rt"
    if is_gz_file(juncFilePath):
        import gzip

        fopen = gzip.open
        fmode = (
            "r"
            if sys.platform.startswith("win") and sys.version_info.major < 3
            else fmode
        )
    with fopen(juncFilePath, fmode) as uncfile:
        for line in uncfile:
            if line.startswith("#"):
                continue
            elif line.startswith("["):
                component_name = line.strip()[1:-1]  # remove leading and trailing []
                cname = "just/sum/dummy/dir/{}_{}.junc.txt".format(
                    basename, component_name
                )
                components.append((cname, []))
            elif components:
                components[-1][1].append(line)
            else:
                continue

    if not components:  # there are no components in the file
        components.append((juncFilePath, None))
    else:
        components = [(i, io.StringIO("".join(j))) for i, j in components]

    retval = {}
    for name, ifile in components:
        retval.update(convert_junc_txt_component(name, ifile))
    return retval


def convert_junc_txt_component(juncFilePath, uncFile):
    (
        name,
        layout,
        pars,
        nBinnedVars,
        nBinColumns,
        nEvalVars,
        formula,
        nParams,
        columns,
        dtypes,
    ) = _parse_jme_formatted_file(
        juncFilePath, interpolatedFunc=True, paramsFromColumns=True, jme_f=uncFile
    )

    temp = _build_standard_jme_lookup(
        name,
        layout,
        pars,
        nBinnedVars,
        nBinColumns,
        nEvalVars,
        formula,
        nParams,
        columns,
        dtypes,
        interpolatedFunc=True,
    )
    wrapped_up = {}
    for key, val in temp.items():
        newkey = (key[0], "jec_uncertainty_lookup")
        vallist = list(val)
        vals, names = vallist[-1]
        knots = vals[0 : len(vals) : 3]
        downs = vals[1 : len(vals) : 3]
        ups = vals[2 : len(vals) : 3]
        downs = numpy.array([numpy.array(awkward.flatten(down)) for down in downs])
        ups = numpy.array([numpy.array(awkward.flatten(up)) for up in ups])
        for knotv in knots:
            knot = numpy.unique(numpy.array(awkward.flatten(knotv)))
            if knot.size != 1:
                raise Exception("Multiple bin low edges found")
        knots = numpy.array(
            [numpy.unique(numpy.array(awkward.flatten(k)))[0] for k in knots]
        )
        vallist[2] = ({"knots": knots, "ups": ups.T, "downs": downs.T}, vallist[2][-1])
        vallist = vallist[:-1]
        wrapped_up[newkey] = tuple(vallist)
    return wrapped_up


def convert_effective_area_file(eaFilePath):
    fopen = open
    fmode = "rt"
    if is_gz_file(eaFilePath):
        import gzip

        fopen = gzip.open
        fmode = (
            "r"
            if sys.platform.startswith("win") and sys.version_info.major < 3
            else fmode
        )
    ea_f = fopen(eaFilePath, fmode)
    layoutstr = ea_f.readline().strip().strip("{}")
    ea_f.close()

    name = eaFilePath.split("/")[-1].split(".")[0]

    layout = layoutstr.split()
    if not layout[0].isdigit():
        raise Exception("First column of Effective Area File Header must be a digit!")

    # setup the file format
    nBinnedVars = int(layout[0])
    nEvalVars = int(layout[nBinnedVars + 1])

    minMax = ["Min", "Max"]
    columns = []
    dtypes = []
    offset = 1
    for i in range(nBinnedVars):
        columns.extend([f"{layout[i + offset]}{mm}" for mm in minMax])
        dtypes.extend(["<f4", "<f4"])
    offset += nBinnedVars + 1
    for i in range(nEvalVars):
        columns.append("%s" % (layout[i + offset]))
        dtypes.append("<f4")

    pars = numpy.genfromtxt(
        eaFilePath,
        dtype=tuple(dtypes),
        names=tuple(columns),
        skip_header=1,
        encoding="ascii",
    )

    bins = {}
    offset_col = 0
    offset_name = 1
    bin_order = []
    for i in range(nBinnedVars):
        binMins = None
        binMaxs = None
        if i == 0:
            binMins = numpy.unique(pars[columns[0]])
            binMaxs = numpy.unique(pars[columns[1]])
            bins[layout[i + offset_name]] = numpy.union1d(binMins, binMaxs)
        else:
            counts = numpy.zeros(0, dtype=numpy.int)
            allBins = numpy.zeros(0, dtype=numpy.double)
            for binMin in bins[bin_order[0]][:-1]:
                binMins = numpy.unique(
                    pars[numpy.where(pars[columns[0]] == binMin)][
                        columns[i + offset_col]
                    ]
                )
                binMaxs = numpy.unique(
                    pars[numpy.where(pars[columns[0]] == binMin)][
                        columns[i + offset_col + 1]
                    ]
                )
                theBins = numpy.union1d(binMins, binMaxs)
                allBins = numpy.append(allBins, theBins)
                counts = numpy.append(counts, theBins.size)
            bins[layout[i + offset_name]] = awkward.to_packed(
                awkward.unflatten(allBins, counts)
            )
        bin_order.append(layout[i + offset_name])
        offset_col += 1

    # again this is only for one dimension of binning, fight me
    # we can figure out a 2D EA when we get there
    offset_name += 1
    wrapped_up = {}
    lookup_type = "dense_lookup"
    dims = bins[layout[1]]
    for i in range(nEvalVars):
        ea_name = "_".join([name, columns[offset_name + i]])
        values = pars[columns[offset_name + i]]
        wrapped_up[(ea_name, lookup_type)] = (values, dims)

    return wrapped_up


def convert_rochester_file(path, loaduncs=True):
    initialized = False

    fopen = open
    fmode = "rt"
    if is_gz_file(path):
        import gzip

        fopen = gzip.open
        fmode = (
            "r"
            if sys.platform.startswith("win") and sys.version_info.major < 3
            else fmode
        )

    with fopen(path, fmode) as f:
        for line in f:
            line = line.strip("\n")
            # the number of sets available
            if line.startswith("NSET"):
                nsets = int(line.split()[1])
            # the number of members in a given set
            elif line.startswith("NMEM"):
                members = [int(x) for x in line.split()[1:]]
                assert len(members) == nsets
            # the type of the values provided: 0 is default, 1 is replicas (for stat unc), 2 is Symhes (for various systematics)
            elif line.startswith("TVAR"):
                tvars = [int(x) for x in line.split()[1:]]
                assert len(tvars) == nsets
            # number of phi bins
            elif line.startswith("CPHI"):
                nphi = int(line.split()[1])
                phiedges = [
                    float(x) * 2 * numpy.pi / nphi - numpy.pi for x in range(nphi + 1)
                ]
            # number of eta bins and edges
            elif line.startswith("CETA"):
                neta = int(line.split()[1])
                etaedges = [float(x) for x in line.split()[2:]]
                assert len(etaedges) == neta + 1
            # minimum number of tracker layers with measurement
            elif line.startswith("RMIN"):
                nmin = int(line.split()[1])
            # number of bins in the number of tracker layers measurements
            elif line.startswith("RTRK"):
                ntrk = int(line.split()[1])
            # number of abseta bins and edges
            elif line.startswith("RETA"):
                nabseta = int(line.split()[1])
                absetaedges = [float(x) for x in line.split()[2:]]
                assert len(absetaedges) == nabseta + 1
            # load the parameters
            # the structure will be
            # SETNUMBER MEMBERNUMBER TAG[T,R,F,C] [TAG specific indices] [values]
            else:
                if not initialized:
                    # don't want to necessarily load uncertainties
                    toload = nsets if loaduncs else 1
                    M = {
                        s: {m: {t: {} for t in range(2)} for m in range(members[s])}
                        for s in range(toload)
                    }
                    A = {
                        s: {m: {t: {} for t in range(2)} for m in range(members[s])}
                        for s in range(toload)
                    }
                    kRes = {
                        s: {m: {t: [] for t in range(2)} for m in range(members[s])}
                        for s in range(toload)
                    }
                    rsPars = {
                        s: {m: {t: {} for t in range(3)} for m in range(members[s])}
                        for s in range(toload)
                    }
                    cbS = {s: {m: {} for m in range(members[s])} for s in range(toload)}
                    cbA = {s: {m: {} for m in range(members[s])} for s in range(toload)}
                    cbN = {s: {m: {} for m in range(members[s])} for s in range(toload)}
                    initialized = True
                remainder = line.split()
                setn, membern, tag, *remainder = remainder
                setn = int(setn)
                membern = int(membern)
                tag = str(tag)
                # tag T has 2 indices corresponding to TYPE BINNUMBER and has RTRK+1 values each
                # these correspond to the nTrk[2] parameters of RocRes (and BINNUMBER is the abseta bin)
                if tag == "T":
                    t, b, *remainder = remainder
                    t = int(t)
                    b = int(b)
                    values = [float(x) for x in remainder]
                    assert len(values) == ntrk + 1

                # tag R has 2 indices corresponding to VARIABLE BINNUMBER and has RTRK values each
                # these variables correspond to the rsPar[3] and crystal ball (std::vector<CrystalBall> cb) of RocRes where CrystalBall has values s, a, n
                # (and BINNUMBER is the abseta bin)
                # Note: crystal ball here is a symmetric double-sided crystal ball
                elif tag == "R":
                    v, b, *remainder = remainder
                    v = int(v)
                    b = int(b)
                    values = [float(x) for x in remainder]
                    assert len(values) == ntrk
                    if v in range(3):
                        if setn in rsPars:
                            rsPars[setn][membern][v][b] = values
                            if v == 2:
                                rsPars[setn][membern][v][b] = [x * 0.01 for x in values]
                    elif v == 3:
                        if setn in cbS:
                            cbS[setn][membern][b] = values
                    elif v == 4:
                        if setn in cbA:
                            cbA[setn][membern][b] = values
                    elif v == 5:
                        if setn in cbN:
                            cbN[setn][membern][b] = values

                # tag F has 1 index corresponding to TYPE and has RETA values each
                # these correspond to the kRes[2] of RocRes
                elif tag == "F":
                    t, *remainder = remainder
                    t = int(t)
                    values = [float(x) for x in remainder]
                    assert len(values) == nabseta
                    if setn in kRes:
                        kRes[setn][membern][t] = values

                # tag C has 3 indices corresponding to TYPE VARIABLE BINNUMBER and has NPHI values each
                # these correspond to M and A values of CorParams (and BINNUMBER is the eta bin)
                # These are what are used to get the scale factor for kScaleDT (and kScaleMC)
                #       scale = 1.0 / (M+Q*A*pT)
                # For the kSpreadMC (gen matched, recommended) and kSmearMC (not gen matched), we need all of the above parameters
                elif tag == "C":
                    t, v, b, *remainder = remainder
                    t = int(t)
                    v = int(v)
                    b = int(b)
                    values = [float(x) for x in remainder]
                    assert len(values) == nphi
                    if v == 0:
                        if setn in M:
                            M[setn][membern][t][b] = [1.0 + x * 0.01 for x in values]
                    elif v == 1:
                        if setn in A:
                            A[setn][membern][t][b] = [x * 0.01 for x in values]

                else:
                    raise ValueError(line)

    # now build the lookup tables
    # for data scale, simple, just M A in bins of eta,phi
    _scaleedges = (numpy.array(etaedges), numpy.array(phiedges))
    _Mvalues = {
        s: {
            m: {t: numpy.array([M[s][m][t][b] for b in range(neta)]) for t in M[s][m]}
            for m in M[s]
        }
        for s in M
    }
    _Avalues = {
        s: {
            m: {t: numpy.array([A[s][m][t][b] for b in range(neta)]) for t in A[s][m]}
            for m in A[s]
        }
        for s in A
    }

    # for mc scale, more complicated
    # version 1 if gen pt available
    # only requires the kRes lookup
    _resedges = numpy.array(absetaedges)
    _kResvalues = {
        s: {m: {t: numpy.array(kRes[s][m][t]) for t in kRes[s][m]} for m in kRes[s]}
        for s in kRes
    }

    # version 2 if gen pt not available
    trkedges = [0] + [nmin + x + 0.5 for x in range(ntrk)]
    _cbedges = (numpy.array(absetaedges), numpy.array(trkedges))
    _rsParsvalues = {
        s: {
            m: {
                t: numpy.array([rsPars[s][m][t][b] for b in range(nabseta)])
                for t in rsPars[s][m]
            }
            for m in rsPars[s]
        }
        for s in rsPars
    }
    _cbSvalues = {
        s: {m: numpy.array([cbS[s][m][b] for b in range(nabseta)]) for m in cbS[s]}
        for s in cbS
    }
    _cbAvalues = {
        s: {m: numpy.array([cbA[s][m][b] for b in range(nabseta)]) for m in cbA[s]}
        for s in cbA
    }
    _cbNvalues = {
        s: {m: numpy.array([cbN[s][m][b] for b in range(nabseta)]) for m in cbN[s]}
        for s in cbN
    }

    wrapped_up = {
        "nsets": nsets,
        "members": members,
        "edges": {
            "scales": _scaleedges,
            "res": _resedges,
            "cb": _cbedges,
        },
        "values": {
            "M": _Mvalues,
            "A": _Avalues,
            "kRes": _kResvalues,
            "rsPars": _rsParsvalues,
            "cbS": _cbSvalues,
            "cbA": _cbAvalues,
            "cbN": _cbNvalues,
        },
    }
    return wrapped_up
