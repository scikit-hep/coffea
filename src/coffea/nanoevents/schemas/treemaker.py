from coffea.nanoevents.schemas.base import BaseSchema, nest_jagged_forms, zip_forms


class TreeMakerSchema(BaseSchema):
    """TreeMaker schema builder

    The TreeMaker schema is built from all branches found in the supplied file,
    based on the naming pattern of the branches. There are two steps of to the
    generation of array collections:

    - Objects with vector-like quantities (momentum, coordinate points) in the
      TreeMaker n-tuples are stored using ROOT PtEtaPhiEVectors and XYZPoint
      classes with maximum TTree splitting. These variable branches are grouped
      into a single collection with the original object name, with the
      corresponding coordinate variables names mapped to the standard variable
      names for coffea.nanoevents.methods.vector behaviors. For example:

      - The "Jets" branch in a TreeMaker n-tuple branch stores 'PtEtaPhiEVector's
        corresponding to the momentum of AK4 jets. The resulting collection after
        this first step would contain the vector variables in the form of
        Jets.pt, Jets.eta, Jets.phi, Jets.energy, and addition vector quantities
        (px) can be accessed via the usual vector behavior methods.

      - The "PrimaryVertices" branch in a TreeMaker n-tuple branch stores
        'XYZPoint's corresponding to the coordinates of the primary vertices, The
        resulting collection after this first step wold contain the coordinate
        variables in the form of PrimaryVertices.x, PrimaryVertices.y,
        PrimaryVertices.z.

    - Extended quantities of physic objects are stored in the format
      <Object>_<variable>, such as "Jets_jecFactor". Such variables will be
      merged into the collection <Object>, so the branch "Jets_jetFactor" will be
      access to in the array format as "Jets.jecFactor". An exception to the

    All collections are then zipped into one `base.NanoEvents` record and
    returned.
    """

    __dask_capable__ = True

    def __init__(self, base_form, *args, **kwargs):
        super().__init__(base_form, *args, **kwargs)
        old_style_form = {
            k: v for k, v in zip(self._form["fields"], self._form["contents"])
        }
        output = self._build_collections(old_style_form)
        self._form["fields"] = [k for k in output.keys()]
        self._form["contents"] = [v for v in output.values()]

    def _build_collections(self, branch_forms):
        # Turn any special classes into the appropriate awkward form
        composite_objects = list(
            {k.split("/")[0].rstrip("_") for k in branch_forms if "/" in k}
        )

        composite_behavior = {  # Dictionary for overriding the default behavior
            "Tracks": "LorentzVector"
        }
        for objname in composite_objects:
            components = {  # Extracting the various composite object names
                k.split(".")[-1]: k
                for k in branch_forms
                if k.startswith(objname + "/") or
                # Second case for skimming
                k.startswith(objname + "_/")
            }

            if set(components.keys()) == {
                "fPt",
                "fEta",
                "fPhi",
                "fE",
            }:
                form = zip_forms(
                    {
                        "pt": branch_forms.pop(components["fPt"]),
                        "eta": branch_forms.pop(components["fEta"]),
                        "phi": branch_forms.pop(components["fPhi"]),
                        "energy": branch_forms.pop(components["fE"]),
                    },
                    objname,
                    composite_behavior.get(objname, "PtEtaPhiELorentzVector"),
                )
                branch_forms[objname] = form
            elif {x.split(".")[-1] for x in components} == {
                "fX",
                "fY",
                "fZ",
            }:
                form = zip_forms(
                    {
                        "x": branch_forms.pop(components["fX"]),
                        "y": branch_forms.pop(components["fY"]),
                        "z": branch_forms.pop(components["fZ"]),
                    },
                    objname,
                    composite_behavior.get(objname, "ThreeVector"),
                )
                branch_forms[objname] = form
            else:
                raise ValueError(
                    f"Unrecognized class with split branches of object {objname}: {components.values()}"
                )

        # Generating collection from branch name
        collections = [k for k in branch_forms if "_" in k and not k.startswith("n")]
        collections = {
            "_".join(k.split("_")[:-1])
            for k in collections
            if k.split("_")[-1] != "AK8"
            # Excluding per-event variables with AK8 variants like Mjj and MT
        }

        subcollections = []

        for cname in collections:
            items = sorted(k for k in branch_forms if k.startswith(cname + "_"))
            if len(items) == 0:
                continue

            # Special pattern parsing for <collection>_<subcollection>Counts branches
            countitems = [x for x in items if x.endswith("Counts")]
            subcols = {x[:-6] for x in countitems}  # List of subcollection names
            for subcol in subcols:
                items = [
                    k for k in items if not k.startswith(subcol) or k.endswith("Counts")
                ]
                subname = subcol[len(cname) + 1 :]
                subcollections.append(
                    {
                        "colname": cname,
                        "subcol": subcol,
                        "countname": subname + "Counts",
                        "subname": subname,
                    }
                )

            if cname not in branch_forms:
                collection = zip_forms(
                    {k[len(cname) + 1]: branch_forms.pop(k) for k in items}, cname
                )
                branch_forms[cname] = collection
            else:
                collection = branch_forms[cname]
                if not collection["class"].startswith("ListOffsetArray"):
                    print(collection["class"])
                    raise NotImplementedError(
                        f"{cname} isn't a jagged array, not sure what to do"
                    )
                for item in items:
                    Itemname = item[len(cname) + 1 :]
                    collection["content"]["fields"].append(Itemname)
                    collection["content"]["contents"].append(
                        branch_forms.pop(item)["content"]
                    )

        for sub in subcollections:
            nest_jagged_forms(
                branch_forms[sub["colname"]],
                branch_forms.pop(sub["subcol"]),
                sub["countname"],
                sub["subname"],
            )

        return branch_forms

    @classmethod
    def behavior(cls):
        """Behaviors necessary to implement this schema (dict)"""
        from coffea.nanoevents.methods import base, vector

        behavior = {}
        behavior.update(base.behavior)
        behavior.update(vector.behavior)
        return behavior

    @classmethod
    def uproot_writeable(cls, events):
        """
        Converting a TreeMakerSchema event into something that is uproot
        writeable. Based off the discussion thread here [1], but added specific
        cased to handled the nested structures define for TreeMaker n-tuples.
        [1] https://github.com/scikit-hep/coffea/discussions/735

        Parameters
        ----------
            events: TreeMakerSchema events
                The TreeMakerSchema events to be turned into something uproot-writeable

        Returns
        -------
            out: dict
                An uproot-writeable dictionary representing the same information as the input
                TreeMakerSchema events
        """
        import awkward as ak

        def _is_compat(a):
            """Is it a flat or 1-d jagged array?"""
            t = ak.type(a)
            if isinstance(t, ak.types.ArrayType):
                if isinstance(t._content, ak.types.NumpyType):
                    return True
                if isinstance(t._content, ak.types.ListType) and isinstance(
                    t._content._content, ak.types.NumpyType
                ):
                    return True
            return False

        def _make_packed(arr):
            return ak.ak_to_packed.to_packed(ak.without_parameters(arr))

        def zip_composite(array):
            # Additional naming scheme to allow composite object read back
            _rename_lookup = {
                "pt": "/.fPt",
                "eta": "/.fEta",
                "phi": "/.fPhi",
                "energy": "/.fE",
                "x": "/.fX",
                "y": "/.fY",
                "z": "/.fZ",
            }
            return ak.zip(
                {
                    _rename_lookup.get(n, n): _make_packed(array[n])
                    for n in array.fields
                    if _is_compat(array[n])
                }
            )

        # Looping over events structure
        out = {}
        for bname in events.fields:
            if events[bname].fields:
                sub_collection = [  # Handing nested structures first
                    x.replace("Counts", "")
                    for x in events[bname].fields
                    if x.endswith("Counts")
                ]
                if sub_collection:
                    for subname in sub_collection:
                        if events[bname][subname].fields:
                            out[f"{bname}_{subname}"] = zip_composite(
                                ak.flatten(events[bname][subname], axis=-1)
                            )
                        else:
                            out[f"{bname}_{subname}"] = _make_packed(
                                ak.flatten(events[bname][subname], axis=-1)
                            )
                out[bname] = zip_composite(events[bname])
            else:
                out[bname] = _make_packed(events[bname])
        return out
