import copy
import warnings
from collections import defaultdict

from coffea.nanoevents.schemas.base import BaseSchema, zip_forms
from coffea.nanoevents.util import quote


class PHYSLITESchema(BaseSchema):
    """PHYSLITE schema builder - work in progress.

    This is a schema for the `ATLAS DAOD_PHYSLITE derivation
    <https://gitlab.cern.ch/atlas/athena/-/blob/release/21.2.108.0/PhysicsAnalysis/DerivationFramework/DerivationFrameworkPhys/share/PHYSLITE.py>`_.
    Closely following `schemas.nanoaod.NanoAODSchema`, it is mainly build from
    naming patterns where the "Analysis" prefix has been removed, so the
    collections will be named Electrons, Muons, instead of AnalysisElectrons,
    AnalysisMunos, etc. The collection fields correspond to the "Aux" and
    "AuxDyn" columns.

    Collections are assigned mixin types according to the `mixins` mapping.
    All collections are then zipped into one `base.NanoEvents` record and returned.

    Cross references are build from ElementLink columns. Global indices are
    created dynamically, using an ``_eventindex`` field that is attached to
    each collection.
    """

    __dask_capable__ = True

    truth_collections = [
        "TruthPhotons",
        "TruthMuons",
        "TruthNeutrinos",
        "TruthTaus",
        "TruthElectrons",
        "TruthBoson",
        "TruthBottom",
        "TruthTop",
    ]
    """TRUTH3 collection names.

    TruthParticle behavior is assigned to all of them and global index forms
    for parent/children relations are created for all combinations.
    """

    mixins = {
        "Electrons": "Electron",
        "Muons": "Muon",
        "Jets": "Particle",
        "TauJets": "Particle",
        "CombinedMuonTrackParticles": "TrackParticle",
        "ExtrapolatedMuonTrackParticles": "TrackParticle",
        "GSFTrackParticles": "TrackParticle",
        "InDetTrackParticles": "TrackParticle",
        "MuonSpectrometerTrackParticles": "TrackParticle",
        "egammaClusters": "NanoCollection",
    }
    """Default configuration for mixin types, based on the collection name.

    The types are implemented in the `coffea.nanoevents.methods.physlite` module.
    """

    for _k in truth_collections:
        mixins[_k] = "TruthParticle"

    def __init__(self, base_form, *args, **kwargs):
        super().__init__(base_form)
        form_dict = {
            key: form for key, form in zip(self._form["fields"], self._form["contents"])
        }
        output = self._build_collections(form_dict)
        self._form["fields"] = [k for k in output.keys()]
        self._form["contents"] = [v for v in output.values()]

    def _build_collections(self, branch_forms):
        zip_groups = defaultdict(list)
        has_eventindex = defaultdict(bool)
        for key, ak_form in branch_forms.items():
            # Normal fields
            key_fields = key.split("/")[-1].split(".")
            top_key = key_fields[0]
            sub_key = ".".join(key_fields[1:])
            if ak_form["class"] == "RecordArray" and not ak_form["fields"]:
                # skip empty records (e.g. the branches ending in "." only containing the base class)
                continue
            objname = (
                top_key.replace("Analysis", "").replace("AuxDyn", "").replace("Aux", "")
            )

            zip_groups[objname].append(((key, sub_key), ak_form))

            # add eventindex form, based on the first single-jagged list column
            if (
                not has_eventindex[objname]
                and "List" in ak_form["class"]
                and "List" not in ak_form["content"]["class"]
            ):
                zip_groups[objname].append(
                    ((key, "_eventindex"), self._create_eventindex_form(ak_form, key))
                )
                has_eventindex[objname] = True

        # zip the forms
        contents = {}
        for objname, keys_and_form in zip_groups.items():
            if len(keys_and_form) == 1:
                # don't zip if there is only one item
                contents[objname] = keys_and_form[0][1]
                continue
            to_zip = {}
            for (key, sub_key), form in keys_and_form:
                if "." in sub_key:
                    # we can skip fields with '.' in the name since they will come again as records
                    # e.g. truthParticleLink.m_persKey will also appear in truthParticleLink
                    # (record with fields m_persKey and m_persIndex)
                    continue
                if form["class"] == "RecordArray" and form["fields"]:
                    # single-jagged ElementLinks come out as RecordArray(ListOffsetArray)
                    # the zipping converts the forms to ListOffsetArray(RecordArray)
                    fields = [field.split(".")[-1] for field in form["fields"]]
                    form = zip_forms(
                        dict(zip(fields, form["contents"])),
                        sub_key,
                    )
                to_zip[sub_key] = form
            try:
                contents[objname] = zip_forms(
                    to_zip,
                    objname,
                    self.mixins.get(objname, None),
                    bypass=False,
                )
            except NotImplementedError:
                warnings.warn(f"Can't zip collection {objname}")
            if "content" in contents[objname]:
                # in this case we were able to zip everything together to a ListOffsetArray(RecordArray)
                assert "List" in contents[objname]["class"]
                content = contents[objname]["content"]
            else:
                # in this case this was not possible (e.g. because we also had non-list fields)
                assert contents[objname]["class"] == "RecordArray"
                content = contents[objname]
            content["parameters"] = dict(
                content.get("parameters", {}), collection_name=objname
            )
        return contents

    @staticmethod
    def _create_eventindex_form(base_form, key):
        form = copy.deepcopy(base_form)
        form["content"] = {
            "class": "NumpyArray",
            "parameters": {},
            "form_key": quote(f"{key},!load,!eventindex,!content"),
            "itemsize": 8,
            "primitive": "int64",
        }
        return form

    @classmethod
    def behavior(cls):
        """Behaviors necessary to implement this schema (dict)"""
        from coffea.nanoevents.methods import physlite

        return physlite.behavior
