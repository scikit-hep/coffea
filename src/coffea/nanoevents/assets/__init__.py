import os

import yaml

root_dir = os.path.dirname(os.path.abspath(__file__))
path = "/".join([root_dir, "edm4hep.yaml"])
with open(path) as f:
    edm4hep = yaml.safe_load(f)

versions = [
    "00-10-01",
    "00-10-02",
    "00-10-03",
    "00-10-04",
    "00-10-05",
    "00-99-00",
    "00-99-01",
]

edm4hep_ver = {}
for version in versions:
    path = "/".join([root_dir, f"edm4hep_v{version}.yaml"])
    with open(path) as f:
        edm4hep_ver[version] = yaml.safe_load(f)
