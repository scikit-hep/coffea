import urllib.parse
from typing import Any, Dict

from . import BaseSchema


def _build_record_array(
    name: str, name_mapping: Dict[str, str], contents: Dict[str, Any], record_name: str
) -> Dict[str, Any]:
    """Build a record array using the mapping we've got from the contents.

    Args:
        name_mapping (Dict[str, str]): The mapping of user variable to column in the contents
        contents (Dict[str, Any]): The contents of the array we are building into
    """
    items = {
        v_name: contents[col_name]["content"]
        for v_name, col_name in name_mapping.items()
    }
    record = {
        "class": "RecordArray",
        "contents": items.values(),
        "fields": items.keys(),
        "form_key": urllib.parse.quote("!invalid," + name, safe=""),
        "parameters": {"__record__": record_name},
    }
    first = contents[next(iter(name_mapping.values()))]
    return {
        "class": first["class"],
        "offsets": first["offsets"],
        "content": record,
        "form_key": first["form_key"],
        "parameters": {},
    }


class auto_schema(BaseSchema):
    """Build a schema using heuristics to imply a structure"""

    __dask_capable__ = False

    def __init__(self, base_form: Dict[str, Any]):
        """Create an auto schema by parsing the names of the incoming columns

        Notes:
            - There is a recursiveness to this definition, as there is to any data structure,
              that is not matched with this. This should be made much more flexible. Perhaps
              with something like python type-hints so editors can also take advantage of this.
            - Any `_` is inerpreted as going down a level in the structure.

        Args:
            base_form (Dict[str, Any]): The base form of what we are going to generate a new schema (form) for.
        """
        super().__init__(base_form)

        # Get the collection names - anything with a common name before the "_".
        contents = {
            field: content
            for field, content in zip(self._form["fields"], self._form["contents"])
        }
        collections = {k.split("_")[0] for k in contents if "_" in k}

        output = {}
        for c_name in collections:
            mapping = {
                k.split("_")[1]: k for k in contents if k.startswith(f"{c_name}_")
            }

            # Build the new data model from this guy. Look at what is available to see if
            # we can build a 4-vector collection or just a "normal" collection.
            is_4vector_mass = (
                "pt" in mapping
                and "eta" in mapping
                and "phi" in mapping
                and "mass" in mapping
                and "charge" in mapping
            )
            is_4vector_E = (
                "pt" in mapping
                and "eta" in mapping
                and "phi" in mapping
                and "energy" in mapping
                and "charge" in mapping
            )
            record_name = (
                "PtEtaPhiMCandidate"
                if is_4vector_mass
                else "PtEtaPhiECandidate" if is_4vector_E else "NanoCollection"
            )

            record = _build_record_array(
                c_name,
                mapping,
                contents,
                record_name,
            )

            record["parameters"].update({"collection_name": c_name})
            output[c_name] = record

        # Single items in the collection
        single_items = [k for k in contents if "_" not in k]
        for item_name in single_items:
            output[item_name] = contents[item_name]

        self._form["fields"], self._form["contents"] = [k for k in output.keys()], [
            v for v in output.values()
        ]

    @classmethod
    def behavior(cls):
        """Behaviors necessary to implement this schema"""
        from coffea.nanoevents.methods import base, candidate

        behavior = {}
        behavior.update(base.behavior)
        behavior.update(candidate.behavior)
        return behavior
