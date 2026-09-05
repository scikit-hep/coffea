import importlib.resources
import os
from functools import partial

import yaml

root_dir = importlib.resources.files("coffea.nanoevents.assets")

# Zero-padded tags: lexical order is release order.
versions = sorted(
    p.name[len("edm4hep_v") : -len(".yaml")]
    for p in root_dir.iterdir()
    if p.name.startswith("edm4hep_v") and p.name.endswith(".yaml")
)


def _load_edm4hep_version(yamlfile):
    with open(yamlfile) as f:
        loaded = yaml.safe_load(f)
    return loaded


edm4hep_ver = {
    version: partial(
        _load_edm4hep_version,
        yamlfile=os.path.join(root_dir, f"edm4hep_v{version}.yaml"),
    )
    for version in versions
}
