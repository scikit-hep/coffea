import os

import yaml

# path = os.path.abspath('src/coffea/nanoevents/assets/edm4hep.yaml')
root_dir = os.path.dirname(os.path.abspath(__file__))
path = "/".join([root_dir, "edm4hep.yaml"])
with open(path) as f:
    edm4hep = yaml.safe_load(f)
