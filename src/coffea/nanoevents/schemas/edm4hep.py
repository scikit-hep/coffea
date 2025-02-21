import copy
import re
import warnings
from difflib import SequenceMatcher

import numpy

from coffea.nanoevents import transforms
from coffea.nanoevents.methods import vector
from coffea.nanoevents.schemas.base import BaseSchema, zip_forms
from coffea.nanoevents.util import concat

# Load and Parse EDM4HEP.yaml


def parse_Members_and_Relations(Members_and_Relation_List, target_text=False):
    parsed = {}
    for i in Members_and_Relation_List:
        # Separate the declaration and the comment
        separated = i.split("//", 1)
        declaration = separated[0].strip()
        doc_str = ""
        if len(separated) > 1:
            doc_str = separated[1].strip()

        type_str = declaration.split()[0]
        name_str = declaration.split()[1]

        if ("::" in declaration) and ("<" in declaration) and (">" in declaration):
            type_str = declaration.split(">", 1)[0] + ">"
            name_str = declaration.split(">", 1)[1]

        parsed[name_str.strip()] = {"type": type_str.strip(), "doc": doc_str.strip()}
        if target_text:
            parsed[name_str.strip()] = {
                "type": type_str.strip(),
                "target": type_str.strip().split("::")[1],
                "doc": doc_str.strip(),
            }
    return parsed


# from coffea.nanoevents.assets import edm4hep
from coffea.nanoevents.assets import edm4hep_ver


def process_further(loaded_dict, parsed_dict):
    for key in loaded_dict.keys():
        if not isinstance(loaded_dict[key], dict):
            continue
        for subkey in loaded_dict[key].keys():
            if not isinstance(loaded_dict[key][subkey], dict):
                continue
            for subsubkey in loaded_dict[key][subkey].keys():
                if subsubkey in ["Members", "VectorMembers"]:
                    parsed_dict[key][subkey][subsubkey] = parse_Members_and_Relations(
                        loaded_dict[key][subkey][subsubkey]
                    )
                elif subsubkey in ["OneToOneRelations", "OneToManyRelations"]:
                    parsed_dict[key][subkey][subsubkey] = parse_Members_and_Relations(
                        loaded_dict[key][subkey][subsubkey], target_text=True
                    )
    # Add extra datatypes
    parsed_dict["datatypes"]["edm4hep::ObjectID"] = (
        {  # Actually from podio, but, for parsing compatibility, keep as edm4hep
            "Description": "The Monte Carlo particle - based on the lcio::MCParticle.",
            "Author": "Prayag Yadav",
            "Members": {
                "index": {"type": "int64", "doc": "indices to the target collection"},
                "collectionID": {
                    "type": "int64",
                    "doc": "indices to the target collection",
                },
            },
        }
    )
    return parsed_dict


# Collection Regex #
# Any branch name with a forward slash '/'
# Example: 'ReconstructedParticles/ReconstructedParticles.energy'
_all_collections = re.compile(r".*[\/]+.*")

# Any branch name with '[' and ']'
# Example: 'ReconstructedParticles/ReconstructedParticles.covMatrix[10]'
_square_braces = re.compile(r".*\[.*\]")

# Any branch name that starts with an underscore has no '/'
# This is the characteristic of a vector member branch that stores all the vector elements
# Example: _EventHeader_weights'
_vector_members = re.compile(r"^_[^/]*$")

__dask_capable__ = True


def generate_collection_info_from_branches(branch_forms):
    all_collections = {
        collection_name.split("/")[0]
        for collection_name in branch_forms.keys()
        if _all_collections.match(collection_name)
    }
    collections = {}
    for name in all_collections:
        # skip Links, because they all look the same with same branches, so can't really compare them
        if name.endswith("Link"):
            continue
        if name.startswith("_"):
            continue
        # Gather
        collections[name] = {"relations": []}
        for other_name in all_collections:
            if other_name.startswith("_" + name + "_"):
                collections[name]["relations"].append(other_name[2 + len(name) :])
        list_relations = collections[name]["relations"]
        collections[name]["relations"] = set(list_relations)

    for name in collections.keys():
        # Gather members
        members = {
            collection_name.split("/")[-1].split(".")[1].split("[")[0]
            for collection_name in branch_forms.keys()
            if _all_collections.match(collection_name)
            and collection_name.split("/")[0] == name
        }

        collections[name]["members"] = {
            name
            for name in members
            if (not name.endswith("_begin")) and (not name.endswith("_end"))
        }
        # Gather vector members
        vector_members = {
            branch_name.split("_")[-1]
            for branch_name in branch_forms.keys()
            if _vector_members.match(branch_name) and branch_name.startswith("_" + name)
        }
        collections[name]["vector_members"] = vector_members
    return collections


def generate_collection_info_from_yaml(parsed_dict):
    out = {}
    datatypes = parsed_dict["datatypes"]
    for datatype in datatypes.keys():
        # skip Links, because they all look the same with same branches, so can't really compare them
        if datatype.endswith("Link"):
            continue

        members = set(datatypes[datatype]["Members"].keys())
        vector_members = set()
        if datatypes[datatype].get("VectorMembers", None) is not None:
            vector_members = set(datatypes[datatype]["VectorMembers"].keys())
        OneToOneRelations = set()
        if datatypes[datatype].get("OneToOneRelations", None) is not None:
            OneToOneRelations = set(datatypes[datatype]["OneToOneRelations"].keys())
        OneToManyRelations = set()
        if datatypes[datatype].get("OneToManyRelations", None) is not None:
            OneToManyRelations = set(datatypes[datatype]["OneToManyRelations"].keys())
        relations = OneToOneRelations.union(OneToManyRelations)

        out[datatype] = {
            "members": members,
            "vector_members": vector_members,
            "relations": relations,
        }
        # Do a check if a vector member has an 'edm4hep' type
        # because, then, the vector member info is saved like a one to many relation in the underscored branches
        # So copy that vector member to relations instead, to facilitate comparisons
        for member in copy.copy(vector_members):
            type_of_the_vector_member = datatypes[datatype]["VectorMembers"][member][
                "type"
            ]
            if type_of_the_vector_member.startswith("edm4hep::"):
                # Move this vector member to the relations
                out[datatype]["vector_members"].remove(member)
                out[datatype]["relations"].add(member)

    return out


def match_collection_to_datatype(branch_forms, edm4hep_dict):
    """
    The idea is to check if all the branch names (along with relations) are exactly the same as described in the edm4hep.yaml.
    """

    from_root_file = generate_collection_info_from_branches(branch_forms)
    from_yaml_file = generate_collection_info_from_yaml(edm4hep_dict)

    mixin_match = {}
    for collection_name in from_root_file.keys():
        for datatype in from_yaml_file.keys():
            if from_root_file[collection_name] == from_yaml_file[datatype]:
                mixin_match[collection_name] = datatype.split("::")[-1]

    # Matching links is tough, because they all have the same branch names
    # Here is a hacky attempt to match them
    # Best string match
    all_link_collections = {
        collection_name.split("/")[0]
        for collection_name in branch_forms.keys()
        if (_all_collections.match(collection_name))
        and (not collection_name.startswith("_"))
        and ("Link" in collection_name.split("/")[0])
    }
    link_datatypes = {
        name.split("::")[-1]
        for name in edm4hep_dict["datatypes"].keys()
        if name.endswith("Link")
    }
    for link in all_link_collections:
        similarities = []
        for datatype in link_datatypes:
            similarities.append(
                SequenceMatcher(None, link, datatype).ratio()
            )  # returns a value that quantifies the similarity between the two strings
        best_match_index = numpy.argmax(numpy.array(similarities))
        mixin_match[link] = list(link_datatypes)[best_match_index]
    return mixin_match


def sort_dict(d):
    """Sort a dictionary by key"""
    return {k: d[k] for k in sorted(d)}


class EDM4HEPSchema(BaseSchema):
    """
    Schema-builder for EDM4HEP root file structure.
    """

    __dask_capable__ = True

    # Latest (default) edm4hep_version
    edm4hep_version = "00-99-01"

    # EDM4HEP components mixins
    _components_mixins = {
        "Vector4f": "LorentzVector",
        "Vector3f": "ThreeVector",
        "Vector3d": "ThreeVector",
        "Vector2i": "TwoVector",
        "Vector2f": "TwoVector",
        "TrackState": "TrackState",
        "Quantity": "Quantity",
        "covMatrix2f": "covMatrix",
        "covMatrix3f": "covMatrix",
        "covMatrix4f": "covMatrix",
        "covMatrix6f": "covMatrix",
    }

    # # EDM4HEP datatype mixins
    # _datatype_mixins = {  # Example {'TrackCollection' : 'Track', 'MCParticleCollection':'MCParticleCollection', etc}
    #     name.split("::")[1] + "Collection": name.split("::")[1]
    #     for name in self.parsed_edm4hep["datatypes"].keys()
    #     if name.split("::")[1] != "EventHeader"
    # }
    # _datatype_mixins["EventHeader"] = "EventHeader"

    extra_mixins = {"*idx": "ObjectID"}

    _momentum_fields_e = {
        "energy": "E",
        "momentum.x": "px",
        "momentum.y": "py",
        "momentum.z": "pz",
    }
    _two_vec_replacement = {"a": "x", "b": "y"}
    _replacement = {**_momentum_fields_e, **_two_vec_replacement}

    copy_links_to_target_datatype = False

    # Which collection to match if there are multiple matching collections for a given datatype
    _datatype_priority = {}

    def __init__(self, base_form, *args, **kwargs):
        super().__init__(base_form)

        # Detect Collection Datatypes and create a datatype mixin
        self.edm4hep = edm4hep_ver[self.edm4hep_version]
        self.parsed_edm4hep = process_further(self.edm4hep, copy.deepcopy(self.edm4hep))
        self._create_mixin(base_form)

        self._form["fields"], self._form["contents"] = self._build_collections(
            self._form["fields"], self._form["contents"]
        )

    @classmethod
    def version(cls, ver="latest"):
        match ver:
            case "latest":
                return EDM4HEPSchema
            case "00.99.01":
                return EDM4HEPSchema
            case "00.99.00":
                return EDM4HEPSchema_v00_99_00
            case "00.10.05":
                return EDM4HEPSchema_v00_10_05
            case "00.10.04":
                return EDM4HEPSchema_v00_10_04
            case "00.10.03":
                return EDM4HEPSchema_v00_10_03
            case "00.10.02":
                return EDM4HEPSchema_v00_10_02
            case "00.10.01":
                return EDM4HEPSchema_v00_10_01

    def _create_mixin(self, base_form):
        """Extract mixin dictionary from typename info"""
        eager_mode_typenames = base_form.get("typenames", None)
        if eager_mode_typenames is None:
            # Dask mode has typename stored in each branch
            # Collect all those typenames into a single dictionary
            collected_branch_typenames = {}
            for name, form in zip(self._form["fields"], self._form["contents"]):
                matched = form["parameters"].get("typename", "unknown")
                collected_branch_typenames[name] = matched
            typenames = collected_branch_typenames
        else:
            typenames = eager_mode_typenames

        all_collections = {
            collection_name.split("/")[0]
            for collection_name in self._form["fields"]
            if _all_collections.match(collection_name)
        }

        collections = {
            collection_name
            for collection_name in all_collections
            if not collection_name.startswith("_")
        }

        mixins = {}

        for name in collections:
            datatype = typenames.get(name, "NanoCollection")
            if datatype.startswith(r"vector<edm4hep::"):
                if datatype.endswith("Data>"):
                    mixins[name] = datatype.split("::")[-1][:-5]
                else:
                    raise RuntimeError("Unknown datatype:", datatype)
            elif datatype.startswith(r"vector<podio::"):
                mixins[name] = datatype.split("::")[-1][:-1]
            else:
                mixins[name] = datatype

        mixins_dictionary = {**mixins, **self.extra_mixins}
        self._datatype_mixins = mixins_dictionary

    def _zip_components(self, collection_name, component_branches, branch_forms):
        """
        Zip the members of a component collection
        Eg. referencePoint (edm4hep::Vector3f) has the referencePoint.x, referencePoint.y and referencePoint.z branches
            They are zipped together to return the referencePoint collection
        """
        inverted_dict = {}
        for name in component_branches.keys():
            var = component_branches[name]["branch_var"]
            subvar = component_branches[name]["branch_subvar"]
            type = component_branches[name]["type"]
            doc = component_branches[name]["doc"]
            if var + "@" + type not in inverted_dict.keys():
                inverted_dict[var + "@" + type] = []
            inverted_dict[var + "@" + type].append(
                {"name": name, "branch_subvar": subvar, "doc": doc}
            )

        for var, branch_list in inverted_dict.items():
            assign_name = var.split("@")[0]
            type_name = var.split("@")[1].split("::")[1]
            mixin = self._components_mixins.get(type_name, None)
            if assign_name == "momentum":
                continue  # Used to create 4 vector for the whole collection, later.
            if var.split("@")[1] == "unknown":
                continue

            to_zip_raw = {
                item["branch_subvar"]: branch_forms.pop(item["name"])
                for item in branch_list
            }
            # replace keys if needed
            to_zip = {
                self._replacement.get(name, name): form
                for name, form in to_zip_raw.items()
            }

            replaced_branch = (
                collection_name + "/" + collection_name + "." + assign_name
            )
            branch_forms[replaced_branch] = zip_forms(
                sort_dict(to_zip), str(assign_name), str(mixin)
            )

            branch_forms[replaced_branch]["content"]["parameters"].update(
                {"collection_name": assign_name, "__doc__": branch_list[0]["doc"]}
            )
        return branch_forms

    def _lookup_branch(self, collection_name, branch_name, key=None):
        """
        Returns 'type' or 'doc' of a branch component, given a collection_name, branch_name and key('type' or 'doc')
        """
        datatype = self._datatype_mixins.get(collection_name, None)
        if collection_name.startswith("_"):
            col_name = collection_name[1:].split("_")[0]
            subcol_name = collection_name[1:].split("_")[-1]
            datatype = self._datatype_mixins.get(col_name, None)
        if datatype is None:
            raise FileNotFoundError(f"No datatype found for {collection_name}!")
        collection_edm4hep = self.parsed_edm4hep["datatypes"]["edm4hep::" + datatype]
        Members = collection_edm4hep.get("Members", {})
        VectorMembers = collection_edm4hep.get("VectorMembers", {})
        OneToOneRelations = collection_edm4hep.get("OneToOneRelations", {})
        OneToManyRelations = collection_edm4hep.get("OneToOneRelations", {})
        composite_dict = {
            **Members,
            **VectorMembers,
            **OneToOneRelations,
            **OneToManyRelations,
        }
        if collection_name.startswith("_"):
            matched_subcol = composite_dict.get(
                subcol_name, {"type": "unknown", "doc": "unknown"}
            )
            components_edm4hep = self.parsed_edm4hep["components"].get(
                matched_subcol["type"], {"Members": {}}
            )["Members"]
            composite_dict = {**composite_dict, **components_edm4hep}

        if key is not None:
            return composite_dict.get(
                branch_name, {"type": "unknown", "doc": "unknown"}
            )[key]
        return composite_dict.get(branch_name, {"type": "unknown", "doc": "unknown"})

    def _doc_strings(self, branch_forms, collections):
        """
        Assign docstrings for all branches
        Docstrings are taken from the comments in edm4hep.yaml
        """

        def assign_doc(branch, doc):
            branch["content"]["parameters"]["__doc__"] = doc
            return branch

        fieldnames = list(branch_forms.keys())
        for collection in collections:
            for name in fieldnames:
                slash_split = name.split("/")
                if slash_split[0] == collection:
                    if (
                        len(slash_split) > 1
                    ):  # Ensure that no placeholder branch is allowed
                        branch_name_split = slash_split[1].split(".")
                        var_name = branch_name_split[1]
                        if "[" in branch_name_split[1] and "]" in branch_name_split[1]:
                            var_name = branch_name_split[1].split("[")[0]
                        # Assign doc strings to all branches
                        doc = self._lookup_branch(collection, var_name, "doc")
                        branch_forms[name] = assign_doc(branch_forms[name], doc)
        return branch_forms

    def _process_components(self, branch_forms, all_collections):
        """
        Zip all the component types (except if the component is a VectorMember for a datatype)
        """

        def _process(branch_forms, collections):
            fieldnames = branch_forms.keys()
            for collection in collections:
                component_branches = {}
                for name in fieldnames:
                    slash_split = name.split("/")
                    if slash_split[0] == collection:
                        if (
                            len(slash_split) > 1
                        ):  # Ensure that no placeholder branch is allowed
                            branch_name_split = slash_split[1].split(".")
                            if len(branch_name_split) > 2:
                                branch_var = branch_name_split[-2]
                                branch_subvar = branch_name_split[-1]
                                # skip momentum because it will be used
                                # later to create 4 vector with E or mass
                                if branch_var == "momentum":
                                    continue
                                component = self._lookup_branch(collection, branch_var)
                                component_type = component["type"]
                                component_doc = component["doc"]

                                component_branches[name] = {
                                    "type": component_type,
                                    "branch_var": branch_var,
                                    "branch_subvar": branch_subvar,
                                    "doc": component_doc,
                                }
                branch_forms = self._zip_components(
                    collection, component_branches, branch_forms
                )
            return branch_forms

        branch_forms = _process(branch_forms, all_collections)
        branch_forms = _process(
            branch_forms, all_collections
        )  # Doing it twice to deal with nested components, if at all present

        return branch_forms

    def _process_VectorMembers(self, branch_forms, all_collections):
        fieldnames = list(branch_forms.keys())

        for collection in all_collections:
            if collection.startswith("_"):
                continue
            branch_var = {
                name.split("/")[1].split(".")[1]: branch_forms[name]
                for name in fieldnames
                if (name.split("/")[0] == collection) and (len(name.split("/")) > 1)
            }
            datatype = self._datatype_mixins.get(collection, None)
            if datatype is None:
                continue
            vec_members = self.parsed_edm4hep["datatypes"]["edm4hep::" + datatype].get(
                "VectorMembers", None
            )
            if vec_members is None:
                continue
            for member in vec_members.keys():
                target_contents = {
                    name.split("/")[1][1:].split("_")[1]: branch_forms.pop(name)
                    for name in fieldnames
                    if name.startswith(f"_{collection}_{member}")
                    and (len(name.split("/")) > 1)
                }
                begin_form = branch_var[member + "_begin"]
                branch_forms.pop(f"{collection}/{collection}.{member}_begin")
                end_form = branch_var[member + "_end"]
                branch_forms.pop(f"{collection}/{collection}.{member}_end")

                vars = list(target_contents.keys())
                if len(vars) == 0:
                    if not vec_members[member]["type"].startswith("edm4hep::"):
                        # Example : _EventHeader_weights where 'weights'
                        # is the VectorMember of 'EventHeader' datatype
                        # ('weights' not to be confused with 'weight' )
                        associated_target_form = branch_forms.get(
                            f"_{collection}_{member}", None
                        )
                        if associated_target_form is None:
                            continue
                        branch_forms.pop(f"_{collection}_{member}")
                        target_form = transforms.begin_end_mapping_form(
                            begin_form, end_form, associated_target_form
                        )
                    else:
                        raise RuntimeError(f"_{collection}_{member} not found!")
                elif len(vars) == 1:
                    target_form = transforms.begin_end_mapping_form(
                        begin_form, end_form, target_contents[vars[0]]
                    )
                else:
                    # Example : _TrackCollection_trackStates.D0, _TrackCollection_trackStates.phi, etc.
                    #  where 'trackStates' is the VectorMember of 'TrackStates' component in 'Track' datatype
                    vec_contents = {
                        name.split(".")[1]: transforms.begin_end_mapping_form(
                            begin_form, end_form, targetform
                        )
                        for name, targetform in target_contents.items()
                    }
                    vec_contents = {}
                    for name, targetform in target_contents.items():
                        if name.endswith("covMatrix"):
                            vec_contents[name.split(".")[1]] = (
                                transforms.begin_end_mapping_nested_target_form(
                                    begin_form, end_form, targetform
                                )
                            )
                        elif name.endswith("referencePoint"):
                            vec_contents[name.split(".")[1]] = (
                                transforms.begin_end_mapping_with_xyzrecord_form(
                                    begin_form, end_form, targetform
                                )
                            )
                        else:
                            vec_contents[name.split(".")[1]] = (
                                transforms.begin_end_mapping_form(
                                    begin_form, end_form, targetform
                                )
                            )
                    target_form = zip_forms(vec_contents, member)

                branch_forms[f"{collection}/{collection}.{member}"] = target_form
                branch_forms[f"{collection}/{collection}.{member}"]["content"][
                    "parameters"
                ] = {"__doc__": vec_members[member]["doc"]}

        return branch_forms

    def _process_OneToOneRelations(self, branch_forms, all_collections):
        fieldnames = list(branch_forms.keys())

        for collection in all_collections:
            if collection.startswith("_"):
                continue
            datatype = self._datatype_mixins.get(collection, None)
            if datatype is None:
                continue
            OneToOneRelations = self.parsed_edm4hep["datatypes"][
                "edm4hep::" + datatype
            ].get("OneToOneRelations", None)
            if OneToOneRelations is None:
                continue
            for member in OneToOneRelations.keys():
                if member in ["from", "to"]:
                    continue  # Skip Link Collections
                target_contents = {
                    name.split("/")[1][1:].split("_")[-1]: branch_forms.pop(name)
                    for name in fieldnames
                    if name.startswith(f"_{collection}_{member}")
                    and (len(name.split("/")) > 1)
                }

                vars = list(target_contents.keys())
                if not OneToOneRelations[member]["type"].startswith("edm4hep::"):
                    raise RuntimeError(
                        f"{member} does not point to a valid datatype({OneToOneRelations[member]['type']})!"
                    )
                if len(vars) == 0:
                    raise RuntimeError(f"_{collection}_{member} not found!")
                else:
                    target_datatype = OneToOneRelations[member]["type"]
                    matched_collections = [
                        collection_name
                        for collection_name, datatype in self._datatype_mixins.items()
                        if "edm4hep::" + datatype == target_datatype
                    ]
                    # Potential Bug: What if there are more than one collections with the same datatype
                    # Since We can't get the collection ID from the events branch, truly matching collections
                    # seems impossible here
                    # For now, lets add the relation to all the matched collections
                    # One can provide the priority with the self._datatype_priority dictionary
                    if len(matched_collections) == 0:
                        warnings.warn(
                            f"No matched collection for {target_datatype} found!\n skipping ..."
                        )
                        continue
                    for matched_collection in matched_collections:

                        # grab the offset from one of the branches of the target datatype
                        target_vars = self.parsed_edm4hep["datatypes"][target_datatype][
                            "Members"
                        ]
                        first_var = list(target_vars.keys())[0]
                        offset_form = branch_forms[
                            f"{matched_collection}/{matched_collection}.{first_var}"
                        ]
                        target_datatype_offset_form = {
                            "class": "NumpyArray",
                            "itemsize": 8,
                            "format": "i",
                            "primitive": "int64",
                            "form_key": concat(
                                offset_form["form_key"],
                                "!offsets",
                            ),
                        }

                        OneToOneRelations_content = {
                            name.split(".")[1]: targetform
                            for name, targetform in target_contents.items()
                        }
                        OneToOneRelations_content.update(
                            {
                                "index_Global": transforms.local2global_form(
                                    index=OneToOneRelations_content["index"],
                                    target_offsets=target_datatype_offset_form,
                                )
                            }
                        )

                        for name, form in OneToOneRelations_content.items():
                            branch_forms[
                                f"{collection}/{collection}.{member}_idx_{matched_collection}_{name}"
                            ] = form
                            branch_forms[
                                f"{collection}/{collection}.{member}_idx_{matched_collection}_{name}"
                            ]["content"]["parameters"] = {
                                "__doc__": OneToOneRelations[member]["doc"]
                            }

        return branch_forms

    def _process_OneToManyRelations(self, branch_forms, all_collections):
        fieldnames = list(branch_forms.keys())

        for collection in all_collections:
            if collection.startswith("_"):
                continue
            branch_var = {
                name.split("/")[1].split(".")[1]: branch_forms[name]
                for name in fieldnames
                if (name.split("/")[0] == collection) and (len(name.split("/")) > 1)
            }
            datatype = self._datatype_mixins.get(collection, None)
            if datatype is None:
                continue
            OneToManyRelations = self.parsed_edm4hep["datatypes"][
                "edm4hep::" + datatype
            ].get("OneToManyRelations", None)
            if OneToManyRelations is None:
                continue
            for member in OneToManyRelations.keys():
                if member in ["from", "to"]:
                    continue  # Skip Link Collections
                target_contents = {
                    name.split("/")[1][1:].split("_")[-1]: branch_forms.pop(name)
                    for name in fieldnames
                    if name.startswith(f"_{collection}_{member}")
                    and (len(name.split("/")) > 1)
                }

                begin_form = branch_var[member + "_begin"]
                end_form = branch_var[member + "_end"]
                branch_forms.pop(f"{collection}/{collection}.{member}_begin")
                branch_forms.pop(f"{collection}/{collection}.{member}_end")

                vars = list(target_contents.keys())
                if not OneToManyRelations[member]["type"].startswith("edm4hep::"):
                    raise RuntimeError(
                        f"{member} does not point to a valid datatype({OneToManyRelations[member]['type']})!"
                    )
                if len(vars) == 0:
                    raise RuntimeError(f"_{collection}_{member} not found!")
                else:
                    target_datatype = OneToManyRelations[member]["type"]
                    matched_collections = [
                        collection_name
                        for collection_name, datatype in self._datatype_mixins.items()
                        if "edm4hep::" + datatype == target_datatype
                    ]
                    # Potential problem: What if there are more than one collections with the same datatype
                    # Since We can't get the collection ID from the events branch, truly matching collections
                    # seems impossible here
                    # For now, lets add the relation to all the matched collections
                    if len(matched_collections) == 0:
                        # Might be the TrackerHit Interface
                        if (
                            self.edm4hep["schema_version"] == "1"
                        ):  # version 1 doesn't have interfaces
                            pass
                        elif self.edm4hep["schema_version"] == "2":
                            if target_datatype in list(
                                self.parsed_edm4hep["interfaces"].keys()
                            ):
                                # What datatypes does it interface to?
                                interfaced_datatypes = self.parsed_edm4hep[
                                    "interfaces"
                                ][target_datatype]["Types"]
                                matched_collections = []
                                for i in interfaced_datatypes:
                                    for (
                                        col_name,
                                        datatype_name,
                                    ) in self._datatype_mixins.items():
                                        if "edm4hep::" + datatype_name == i:
                                            matched_collections.append(col_name)
                            else:
                                raise RuntimeError(
                                    f"No matched collection for {target_datatype} found!"
                                )

                    for matched_collection in matched_collections:

                        # grab the offset from one of the branches of the target datatype
                        target_datatype = self._datatype_mixins.get(
                            matched_collection, None
                        )
                        if target_datatype is None:
                            raise RuntimeError()
                        target_vars = self.parsed_edm4hep["datatypes"][
                            "edm4hep::" + target_datatype
                        ]["Members"]
                        first_var = list(target_vars.keys())[0]
                        offset_form = branch_forms[
                            f"{matched_collection}/{matched_collection}.{first_var}"
                        ]
                        target_datatype_offset_form = {
                            "class": "NumpyArray",
                            "itemsize": 8,
                            "format": "i",
                            "primitive": "int64",
                            "form_key": concat(
                                offset_form["form_key"],
                                "!offsets",
                            ),
                        }
                        first_var = list(branch_var.keys())[0]
                        zip_offset_form = branch_var[first_var]
                        zip_datatype_offset_form = copy.deepcopy(
                            target_datatype_offset_form
                        )
                        zip_datatype_offset_form["form_key"] = concat(
                            zip_offset_form["form_key"], "!offsets"
                        )

                        OneToManyRelations_content = {
                            name.split(".")[1]: transforms.begin_end_mapping_form(
                                begin_form, end_form, targetform
                            )
                            for name, targetform in target_contents.items()
                        }
                        OneToManyRelations_content_global = {
                            name
                            + "_Global": transforms.nested_local2global_form(
                                form,
                                target_datatype_offset_form,
                            )
                            for name, form in OneToManyRelations_content.items()
                            if name == "index"
                        }

                        to_zip = {
                            **OneToManyRelations_content,
                            **OneToManyRelations_content_global,
                        }

                        for key, form in to_zip.items():
                            branch_forms[
                                f"{collection}/{collection}.{member}_idx_{matched_collection}_{key}"
                            ] = form
                            branch_forms[
                                f"{collection}/{collection}.{member}_idx_{matched_collection}_{key}"
                            ]["content"]["parameters"] = {
                                "__doc__": OneToManyRelations[member]["doc"]
                            }

        return branch_forms

    def _process_Links(self, branch_forms, all_collections):
        fieldnames = list(branch_forms.keys())

        for collection in all_collections:
            if collection.startswith("_"):
                continue
            datatype = self._datatype_mixins.get(collection, None)
            if datatype is None:
                continue
            OneToOneRelations = self.parsed_edm4hep["datatypes"][
                "edm4hep::" + datatype
            ].get("OneToOneRelations", None)
            if OneToOneRelations is None:
                continue
            if not all(
                link_name in OneToOneRelations.keys() for link_name in ["from", "to"]
            ):
                continue
            set_matched_collections = set()
            dict_branches_to_copy = {}
            dict_docs_of_branches = {}
            for member in OneToOneRelations.keys():
                if member in ["from", "to"]:
                    target_contents = {
                        name.split("/")[1][1:].split("_")[1]: branch_forms.pop(name)
                        for name in fieldnames
                        if name.startswith(f"_{collection}_{member}")
                        and (len(name.split("/")) > 1)
                    }

                    vars = list(target_contents.keys())
                    if not OneToOneRelations[member]["type"].startswith("edm4hep::"):
                        raise RuntimeError(
                            f"{member} does not point to a valid datatype({OneToOneRelations[member]['type']})!"
                        )
                    if len(vars) == 0:
                        raise RuntimeError(f"_{collection}_{member} not found!")
                    else:
                        target_datatype = OneToOneRelations[member]["type"]
                        matched_collections = [
                            collection_name
                            for collection_name, datatype in self._datatype_mixins.items()
                            if "edm4hep::" + datatype == target_datatype
                        ]

                        if len(matched_collections) == 0:
                            # Might be the TrackerHit Interface
                            if target_datatype in list(
                                self.parsed_edm4hep["interfaces"].keys()
                            ):
                                # What datatypes does it interface to?
                                interfaced_datatypes = self.parsed_edm4hep[
                                    "interfaces"
                                ][target_datatype]["Types"]
                                matched_collections = []
                                for i in interfaced_datatypes:
                                    for (
                                        col_name,
                                        datatype_name,
                                    ) in self._datatype_mixins.items():
                                        if "edm4hep::" + datatype_name == i:
                                            matched_collections.append(col_name)
                            # Or maybe not
                            else:
                                raise RuntimeError(
                                    f"No matched collection for {target_datatype} found!"
                                )

                        for matched_collection in matched_collections:

                            # grab the offset from one of the branches of the target datatype
                            target_datatype = self._datatype_mixins.get(
                                matched_collection, None
                            )
                            if target_datatype is None:
                                raise RuntimeError()
                            target_vars = self.parsed_edm4hep["datatypes"][
                                "edm4hep::" + target_datatype
                            ]["Members"]
                            first_var = list(target_vars.keys())[0]
                            offset_form = branch_forms[
                                f"{matched_collection}/{matched_collection}.{first_var}"
                            ]

                            target_datatype_offset_form = {
                                "class": "NumpyArray",
                                "itemsize": 8,
                                "format": "i",
                                "primitive": "int64",
                                "form_key": concat(
                                    offset_form["form_key"],
                                    "!offsets",
                                ),
                            }

                            OneToOneRelations_content = {
                                name.split(".")[1]: targetform
                                for name, targetform in target_contents.items()
                            }
                            OneToOneRelations_content.update(
                                {
                                    "index_Global": transforms.local2global_form(
                                        index=OneToOneRelations_content["index"],
                                        target_offsets=target_datatype_offset_form,
                                    )
                                }
                            )

                            target_form = zip_forms(OneToOneRelations_content, member)
                            branch_forms[
                                f"{collection}/{collection}.Link_{member}_{matched_collection}"
                            ] = target_form
                            branch_forms[
                                f"{collection}/{collection}.Link_{member}_{matched_collection}"
                            ]["parameters"] = {
                                "__doc__": OneToOneRelations[member]["doc"]
                            }
                            # Also copy this to the matched_collections
                            # First collect all the branches that need to be copied
                            do_copy = False
                            if self.copy_links_to_target_datatype:
                                if len(self._datatype_priority.keys()) == 0:
                                    raise RuntimeError(
                                        "Cannot copy links if no priority is given!"
                                    )
                                if len(matched_collections) > 1:
                                    # Choose which one to copy
                                    priority = self._datatype_priority[target_datatype]
                                    if matched_collection == priority:
                                        do_copy = True
                                else:
                                    do_copy = True
                            if do_copy:
                                if member == "from":
                                    set_matched_collections.update({matched_collection})
                                dict_branches_to_copy.update(
                                    {f"Link_{member}_{matched_collection}": target_form}
                                )
                                dict_docs_of_branches.update(
                                    {
                                        f"Link_{member}_{matched_collection}": OneToOneRelations[
                                            member
                                        ][
                                            "doc"
                                        ]
                                    }
                                )

            # Finally, copy the available branches to the set of matched_collections
            if self.copy_links_to_target_datatype:
                for matched_collection in set_matched_collections:
                    for name in dict_branches_to_copy.keys():
                        branch_forms[
                            f"{matched_collection}/{matched_collection}.{name}"
                        ] = dict_branches_to_copy[name]
                        branch_forms[
                            f"{matched_collection}/{matched_collection}.{name}"
                        ]["content"]["parameters"] = {
                            "__doc__": dict_docs_of_branches[name]
                        }

        return branch_forms

    def _make_collections(self, output, branch_forms):
        """
        Process branches to form a collection
        Example:
            "ReconstructedParticles/ReconstructedParticles.energy",
            "ReconstructedParticles/ReconstructedParticles.charge",
            "ReconstructedParticles/ReconstructedParticles.mass",
            "ReconstructedParticles/ReconstructedParticles.referencePoint"(subcollection containing x,y,z),
            ...
            etc
            are zipped together to form the "ReconstructedParticles" collection.
        The momentum.[x,y,z] branches along with the energy branch (if available) are used to
        provide the vector.LorentzVector behavior to the collection.
        """
        field_names = list(branch_forms.keys())

        # Extract the regular collection names
        # Example collections: {'Jet', 'ReconstructedParticles', 'MCRecoAssociations', ...}
        collections = {
            collection_name.split("/")[0]
            for collection_name in field_names
            if _all_collections.match(collection_name)
        }

        # Zip the collections
        # Example: 'ReconstructedParticles'
        for name in collections:
            # Get the mixin class for the collection, if available, otherwise "NanoCollection" by default
            mixin = self._datatype_mixins.get(name, "NanoCollection")

            # Content to be zipped together
            # Example collection_content: {'type':<type form>, 'energy':<energy form>, 'momentum.x':<momentum.x form> ...}
            collection_content = {
                k[(2 * len(name) + 2) :]: branch_forms.pop(k)
                for k in field_names
                if k.startswith(f"{name}/{name}.")
            }

            # Change the name of momentum fields, to facilitate the vector.LorentzVector behavior
            # 'energy' --> 'E'
            # 'momentum.x' --> 'px'
            # 'momentum.y' --> 'py'
            # 'momentum.z' --> 'pz'
            collection_content = {
                (k.replace(k, self._replacement[k]) if k in self._replacement else k): v
                for k, v in collection_content.items()
            }

            first_var_form = collection_content[list(collection_content.keys())[0]]
            offset_form = {
                "class": "NumpyArray",
                "itemsize": 8,
                "format": "i",
                "primitive": "int64",
                "form_key": concat(
                    first_var_form["form_key"],
                    "!offsets",
                ),
            }

            output[name] = zip_forms(
                sort_dict(collection_content),
                name,
                record_name=mixin,
                offsets=offset_form,
            )
            # Update some metadata
            if mixin != "NanoCollection":
                output[name]["content"]["parameters"].update(
                    {
                        "collection_name": name,
                        "__doc__": self.parsed_edm4hep["datatypes"][
                            "edm4hep::" + mixin
                        ].get("Description", mixin),
                    }
                )

            # Remove grouping branches which are generated from BaseSchema and contain no usable info
            # Example: Along with the "Jet/Jet.type","Jet/Jet.energy",etc., BaseSchema may produce "Jet" grouping branch.
            # It is an empty branch and needs to be removed
            if name in field_names:
                branch_forms.pop(name)

        return output, branch_forms

    def _unknown_collections(self, output, branch_forms, all_collections):
        """
        Process all the unknown, empty or faulty branches that remain
        after creating all the collections.
        Should be called only after creating all the other relevant collections.

        Note: It is not a neat implementation and needs more testing.
        """
        unlisted = copy.deepcopy(branch_forms)
        for name, content in unlisted.items():
            if content["class"] == "ListOffsetArray":
                if content["content"]["class"] == "RecordArray":
                    # Remove empty branches
                    if len(content["content"]["fields"]) == 0:
                        branch_forms.pop(name)
                        continue
                elif content["content"]["class"] == "RecordArray":
                    # Remove empty branches
                    if len(content["contents"]) == 0:
                        continue
                # If a branch is non-empty and is one of its kind (i.e. has no other associated branch)
                # call it a singleton and assign it directly to the output
                else:
                    # Singleton branch
                    output[name] = branch_forms.pop(name)
            elif content["class"] == "RecordArray":
                # Remove empty branches
                if len(content["contents"]) == 0:
                    continue
                else:
                    # If the branch is not empty, try to make a collection
                    # assuming good behavior of the branch
                    # Note: It's unlike that such a branch exists

                    # Extract the collection name from the branch
                    record_name = name.split("/")[0]

                    # Contents to be zipped
                    contents = {
                        k[2 * len(record_name) + 2 :]: branch_forms.pop(k)
                        for k in unlisted.keys()
                        if k.startswith(record_name + "/")
                    }
                    if len(list(contents.keys())) == 0:
                        continue
                    output[record_name] = zip_forms(
                        sort_dict(contents),
                        record_name,
                        self._datatype_mixins.get(record_name, "NanoCollection"),
                    )
            # If a branch is non-empty and is one of its kind (i.e. has no other associated branch)
            # call it a singleton and assign it directly to the output
            else:
                output[name] = content

        return output, branch_forms

    def _build_collections(self, field_names, input_contents):
        """
        Builds all the collections with the necessary behaviors defined in the mixins dictionary
        """
        branch_forms = {k: v for k, v in zip(field_names, input_contents)}
        # All collection names
        # Example: ReconstructedParticles or _ReconstructedParticle_clusters, etc
        all_collections = {
            collection_name.split("/")[0]
            for collection_name in field_names
            if _all_collections.match(collection_name)
        }

        output = {}
        branch_forms = self._doc_strings(branch_forms, all_collections)

        branch_forms = self._process_components(branch_forms, all_collections)
        branch_forms = self._process_VectorMembers(branch_forms, all_collections)
        branch_forms = self._process_OneToOneRelations(branch_forms, all_collections)
        branch_forms = self._process_OneToManyRelations(branch_forms, all_collections)
        branch_forms = self._process_Links(branch_forms, all_collections)

        output, branch_forms = self._make_collections(output, branch_forms)

        # Process all the other unknown/faulty/empty/singleton branches
        output, branch_forms = self._unknown_collections(
            output, branch_forms, all_collections
        )

        # sort the output by key
        output = sort_dict(output)

        return output.keys(), output.values()

    @classmethod
    def behavior(cls):
        """Behaviors necessary to implement this schema"""
        from coffea.nanoevents.methods import base, edm4hep

        behavior = {}
        behavior.update(base.behavior)
        behavior.update(vector.behavior)
        behavior.update(edm4hep.behavior)
        return behavior


class EDM4HEPSchema_v00_99_00(EDM4HEPSchema):
    edm4hep_version = "00-99-00"


class EDM4HEPSchema_v00_10_05(EDM4HEPSchema):
    """EDM4HEPSchema for edm4hep version 00.10.05"""

    edm4hep_version = "00-10-05"


class EDM4HEPSchema_v00_10_04(EDM4HEPSchema):
    edm4hep_version = "00-10-04"


class EDM4HEPSchema_v00_10_03(EDM4HEPSchema):
    edm4hep_version = "00-10-03"


class EDM4HEPSchema_v00_10_02(EDM4HEPSchema):
    edm4hep_version = "00-10-02"


class EDM4HEPSchema_v00_10_01(EDM4HEPSchema):
    edm4hep_version = "00-10-01"
