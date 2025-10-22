---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
---

# How to prepare datasets

This guide shows how to assemble input datasets for coffea processors. The goal is to build a `fileset` mapping that the {class}`coffea.processor.Runner` can chunk and distribute efficiently.

## Outline a fileset

```python
from coffea import processor
from coffea.nanoevents import NanoAODSchema

fileset = {
    "DYJets": {
        "treename": "Events",
        "files": [
            "/store/mc/Run3Summer23/.../nanoAOD_*.root",
        ],
        "metadata": {"year": 2022, "is_mc": True},
    },
    "DataMu": {
        "treename": "Events",
        "files": [
            "/eos/cms/store/data/Run2023C/.../nano_data.root",
        ],
        "metadata": {"year": 2023, "is_mc": False},
    },
}

runner = processor.Runner(
    executor=processor.IterativeExecutor(status=True),
    schema=NanoAODSchema,
)
```

- The top-level keys label datasets; they propagate into `events.metadata['dataset']`.
- `treename` is optional if you use the uproot syntax that includes the tree in the filename.
- `metadata` is merged into each chunk’s metadata dictionary and is available inside your processor.

## Mix local and remote files

Uproot accepts local paths, XRootD URLs, and glob patterns. Coffea simply forwards them to uproot.

```python
fileset = {
    "DYJets": {
        "files": [
            "root://cmsxrootd.fnal.gov//store/mc/Run3Summer23/.../nano_1.root",
            "root://cmsxrootd.fnal.gov//store/mc/Run3Summer23/.../nano_2.root",
        ],
        "treename": "Events",
    },
    "LocalTest": {
        "files": ["nano_test.root"],
        "treename": "Events",
    },
}
```

Chunks can span files in different storage systems; coffea relies on uproot to stream the data.

## Discover files programmatically

`coffea.dataset_tools` helps when the list of files lives in Rucio or in JSON manifests.

```python
from coffea.dataset_tools import extract_files_from_rucio

files = extract_files_from_rucio(
    datasets=["/DYJetsToLL_M-50_TuneCP5_13p6TeV/NANOAODSIM"],
    rse="FNAL_DCACHE",
)

fileset = {
    "DYJets": {
        "files": files,
        "treename": "Events",
    }
}
```

You can cache results in JSON and feed them directly to the runner later.

## Store filesets in JSON

```python
import json

with open("fileset.json", "w") as fout:
    json.dump(fileset, fout, indent=2)

# Later
runner(fileset="fileset.json", processor_instance=my_processor)
```

Passing the path to the JSON file lets coffea load the mapping on the fly—a convenient way to share dataset definitions with collaborators.

## Add custom metadata

Metadata travels with each chunk and is available as `events.metadata`.

```python
fileset = {
    "TTJets": {
        "files": ["ttbar.root"],
        "treename": "Events",
        "metadata": {
            "cross_section": 831.76,
            "genfilter": 1.0,
            "year": 2022,
        },
    },
}
```

Inside your processor:

```python
def process(self, events):
    weight = events.metadata.get("cross_section", 1.0)
    ...
```

Use metadata for cross sections, era tags, or analysis-specific flags.

## Keep the fileset small during development

Create reduced filesets for unit tests by slicing the mapping.

```python
mini_fileset = {
    dataset: dict(info, files=info["files"][:1])
    for dataset, info in fileset.items()
}
```

Run your processor with `mini_fileset` locally, then switch back to the full fileset for production.

## Tips & tricks

- Use `NanoEventsFactory.from_root(..., entry_stop=N)` alongside a reduced fileset to validate schemas before launching large jobs.
- Preserve bookkeeping like `n_events` and the sum of generator weights in metadata so weight calculations stay consistent across processors.
- Keep JSON filesets under version control; they document exactly which inputs were analyzed.
- When combining many small files, merge them upstream if possible—large numbers of tiny files increase scheduler overhead during preprocessing.
