# import warnings
# from collections import defaultdict
# import copy
# from coffea.nanoevents.util import quote
import collections.abc
import fnmatch

from coffea.nanoevents.schemas.base import BaseSchema, zip_forms


class PDUNESchema(BaseSchema):
    __dask_capable__ = False
    mixins = {
        # branch tag : obj. class
        "RecoBeam": "Beam",
        "Tracks": "Tracks",
        "Showers": "Showers",
        "reco_beam": "RecoBeam",
        "reco_daughter_allTrack": "Tracks",
        "reco_daughter_allShower": "Showers",
        "start3D": "ThreeVector",
        "end3D": "ThreeVector",
        "start4D": "LorentzVector",
        "end4D": "LorentzVector",
        "vtx3D": "ThreeVector",
    }

    top_objects = {
        "reco_beam": "RecoBeam",
        "reco_daughter_allTrack": "Tracks",
        "reco_daughter_allShower": "Showers",
        "true_beam": "TrueBeam",
    }

    def __init__(self, base_form):
        super().__init__(base_form)
        # print(self.branch_behavior_dict)
        old_style_contents = {
            k: v for k, v in zip(self._form["fields"], self._form["contents"])
        }
        output = self._build_collections(old_style_contents)
        self._form["fields"] = [k for k in output.keys()]
        self._form["contents"] = [v for v in output.values()]

    # build a dictionary of hierarchy i.e.
    # dict["RecoBeam"]["start"]["x/y/z"]=form
    def _recursion(self, key_list, obj, key_dict, i):
        if i < len(key_list) - 1:  # not the last element
            curr_key = key_list[i]
            if curr_key not in key_dict.keys():
                key_dict[curr_key] = {}
            self._recursion(key_list, obj, key_dict[curr_key], i + 1)
        else:
            if i > 0:
                if key_dict.get(key_list[i]) is None:
                    key_dict[key_list[i]] = obj

    # RecoBeam--> start/end/len --> x,y,z
    def _recursive_zip(self, forms, hierarchy, key, final_zip=False):
        # print(key,'\n')
        for k, v in hierarchy.items():
            if isinstance(v, collections.abc.Mapping):
                # print('if:', k,v)
                forms[k] = self._recursive_zip(
                    forms.get(k, {}), hierarchy.get(k, {}), k, True
                )
                # form = zip_forms(form,"events",record_name=None)
            else:
                name = self.mixins[k] if k in self.mixins.keys() else None
                # print('else:', k,v,name)
                forms[k] = zip_forms(forms[k], k, record_name=name)
        if final_zip:
            forms = zip_forms(forms, key, record_name=None)
        return forms

    # forms --> forms
    # first level
    # k,v -> RecoBeam, map
    # second level
    #   forms --> forms[RecoBeam]
    #   k, v -> start:obj
    #   goto else
    #   forms[RecoBeam][start]=zip_forms
    #   forms[RecoBeam][end]=zip_forms
    #   then needs to zip forms[RecoBeam]

    def _filter_branches(self, branches, wildcard):
        return [b for b in branches if fnmatch.fnmatch(b, wildcard)]

    def _type_dictionary_builder(self, branch_forms):
        all_branches = branch_forms.keys()

        # 1. Build ThreeVector Suffix
        V3Var = ["X", "Y", "Z"]
        V4Var = ["Px", "Py", "Pz", "E"]
        # get suffix, check if it ends with X,Y, or Z
        V3Sets = [
            {
                b.split("_")[-1][:-1]
                for b in self._filter_branches(all_branches, "*%s" % V)
            }
            for V in V3Var
        ]
        V3Set = V3Sets[0].intersection(V3Sets[1], V3Sets[2])

        V4Sets = [
            {
                b.split("_")[-1][: -len(V)]
                for b in self._filter_branches(all_branches, "*%s" % V)
            }
            for V in V4Var
        ]
        V4Set = V4Sets[0].intersection(V4Sets[1], V4Sets[2])

        V3Names = [s + v for s in V3Set for v in V3Var]
        V4Names = [s + v for s in V4Set for v in V4Var]
        # note v4 is a subset of v3

        branch_behavior_dict = {}
        # print(V3Set, V4Set)
        for b in all_branches:
            b_fields = b.split("_")
            b_end = b_fields[-1]

            behavior = ""
            if b_end in V3Names:
                behavior = "ThreeVector"
            elif b_end in V4Names:
                behavior = "FourVector"
            branch_behavior_dict[b] = behavior

        return branch_behavior_dict

    def _sort_branches(self, branches):
        sorted_branches = sorted(
            branches,
            key=lambda x: len(x.split("_")) + (self.branch_behavior_dict[x] != ""),
            reverse=True,
        )
        return sorted_branches

    def _build_collections(self, branch_forms):
        self.branch_behavior_dict = self._type_dictionary_builder(branch_forms)

        key_form_dict = (
            {}
        )  # finally formed nested dictionary, the input to build branch form structure
        key_dict = (
            {}
        )  # instead of having `form` in the leaf, using the string 'ak_form'

        # i.e. reco_beam.startPoint --> reco_beam, startPoint
        # branch_keys = {}  # branch_name:[keys, pos]--> a data struct to store

        obj_lists = list(self.top_objects.keys())

        branches = self._sort_branches(
            branch_forms.keys()
        )  # sorted branches used to create nanoevent

        for key in branches:
            ak_form = branch_forms[key]

            behavior = self.branch_behavior_dict[key]

            which_top_key = list(map(lambda v: v in key, obj_lists))
            if sum(which_top_key) == 0:
                continue

            top_key = "".join([t * w for t, w in zip(which_top_key, obj_lists)])
            objname = self.top_objects[top_key]
            sub_keys = key.replace(top_key, objname).split("_")[1:]
            last_key = sub_keys[-1]
            # modify last key to X/Y/Z/Px/Py/Pz/E if this is 3D or 4D object
            if behavior != "":
                v = None
                if any(last_key.endswith(v) for v in ["X", "Y", "Z", "E"]):
                    v = last_key[-1].lower()
                    v = "energy" if v == "e" else v
                    last_key = last_key[:-1] + ("3D" if v != "energy" else "4D")
                if any(last_key.endswith(v) for v in ["Px", "Py", "Pz"]):
                    v = last_key[-2].lower()
                    last_key = last_key[:-1] + "4D"
                sub_keys[-1] = last_key
                self.mixins[last_key] = (
                    "ThreeVector" if last_key.endswith("3D") else "LorentzVector"
                )
                sub_keys.append(v)

            # sub_key = "_".join(sub_keys)

            keys = [objname] + sub_keys

            self._recursion(
                keys, ak_form, key_form_dict, 0
            )  # add form to end of dictionary

            self._recursion(
                keys[:-1], "obj", key_dict, 0
            )  # add string "obj" to end of dictionary

            # print(ak_form)

        # print(key_form_dict)
        # print(key_form_dict)
        # print(key_dict)
        # print(key_form_dict)

        # print(key_form_dict,'\n')
        # print(key_dict,'\n')

        output = self._recursive_zip(key_form_dict, key_dict, "Events")

        return output

    # @staticmethod
    # def _create_eventindex_form(base_form, key):
    #    form = copy.deepcopy(base_form)
    #    form["content"] = {
    #        "class": "NumpyArray",
    #        "parameters": {},
    #        "form_key": quote(f"{key},!load,!eventindex,!content"),
    #        "itemsize": 8,
    #        "primitive": "int64",
    #    }
    #    return form

    @classmethod
    def behavior(cls):
        """Behaviors necessary to implement this schema"""
        from coffea.nanoevents.methods import pdune

        return pdune.behavior
