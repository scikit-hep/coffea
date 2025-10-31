---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
---

# How to prepare datasets

This guide shows how to assemble input datasets for coffea processors. The goal is to build a `fileset` mapping that the {class}`coffea.processor.Runner` can chunk and distribute efficiently.

## Fileset formats

Coffea supports three fileset formats. The most flexible format (recommended) maps each file to its tree name:

### Format 1: File-to-tree mapping (recommended)

```python
from coffea import processor
from coffea.nanoevents import NanoAODSchema

fileset = {
    "DYJets": {
        "files": {
            "/store/mc/.../nano_dy.root": "Events",
            "/store/mc/.../nano_dy_2.root": "Events",
        },
        "metadata": {"year": 2018, "is_mc": True},
    },
    "DataMu": {
        "files": {
            "/store/data/.../nano_data.root": "Events",
        },
        "metadata": {"year": 2018, "is_mc": False},
    },
}

runner = processor.Runner(
    executor=processor.IterativeExecutor(),
    schema=NanoAODSchema,
)

result = runner(fileset, processor_instance=my_processor)
```

- The top-level keys label datasets; they propagate into `events.metadata['dataset']`
- `files` is a dictionary mapping file paths to tree names (e.g., `"Events"`)
- `metadata` is merged into each chunk's metadata dictionary and is available inside your processor via `events.metadata`
- This format allows different tree names per file

### Format 2: Single tree name per dataset

If all files in a dataset use the same tree name, you can specify it once:

```python
fileset = {
    "DYJets": {
        "treename": "Events",
        "files": [
            "/store/mc/.../nano_dy.root",
            "/store/mc/.../nano_dy_2.root",
        ],
        "metadata": {"year": 2018, "is_mc": True},
    },
}

result = runner(fileset, processor_instance=my_processor)
```

### Format 3: Global tree name

If all files across all datasets use the same tree name, pass it to the runner:

```python
fileset = {
    "DYJets": {
        "files": ["/store/mc/.../nano_dy.root"],
        "metadata": {"year": 2018},
    },
}

result = runner(fileset, processor_instance=my_processor, treename="Events")
```

**Note:** Format 1 (file-to-tree mapping) is most flexible and is recommended for new analyses.

## Mix local and remote files

Uproot accepts local paths, XRootD URLs, and glob patterns. Coffea simply forwards them to uproot.

```python
fileset = {
    "DYJets": {
        "files": {
            "root://cmsxrootd.fnal.gov//store/mc/Run3Summer23/.../nano_1.root": "Events",
            "root://cmsxrootd.fnal.gov//store/mc/Run3Summer23/.../nano_2.root": "Events",
        },
        "metadata": {"is_mc": True},
    },
    "LocalTest": {
        "files": {
            "nano_test.root": "Events",
        },
        "metadata": {"is_mc": False},
    },
}
```

Chunks can span files in different storage systems; coffea relies on uproot to stream the data.

## Discover files programmatically

`coffea.dataset_tools` helps when the list of files lives in Rucio or in JSON manifests.

```python
from coffea.dataset_tools import extract_files_from_rucio

files_list = extract_files_from_rucio(
    datasets=["/DYJetsToLL_M-50_TuneCP5_13p6TeV/NANOAODSIM"],
    rse="FNAL_DCACHE",
)

# Convert list to dict mapping files to tree name (Format 1)
fileset = {
    "DYJets": {
        "files": {f: "Events" for f in files_list},
        "metadata": {"is_mc": True},
    }
}

# Or use Format 2 if all files have same tree
fileset_alt = {
    "DYJets": {
        "treename": "Events",
        "files": files_list,
        "metadata": {"is_mc": True},
    }
}
```

You can cache results in JSON and feed them directly to the runner later.

## Store filesets in JSON

```python
import json

with open("fileset.json", "w") as fout:
    json.dump(fileset, fout, indent=2)

# Later, reload the fileset
with open("fileset.json", "r") as fin:
    loaded_fileset = json.load(fin)

result = runner(loaded_fileset, processor_instance=my_processor)
```

Storing filesets in JSON is a convenient way to share dataset definitions with collaborators.

## Add custom metadata

Metadata travels with each chunk and is available as `events.metadata`.

```python
fileset = {
    "TTJets": {
        "files": {
            "ttbar.root": "Events",
        },
        "metadata": {
            "cross_section": 831.76,
            "year": 2022,
        },
    },
}
```

Inside your processor:

```python
def process(self, events):
    dataset = events.metadata["dataset"]  # "TTJets"
    xsec = events.metadata.get("cross_section", 1.0)
    year = events.metadata.get("year")
    # Use metadata in your analysis
    ...
```

Use metadata for cross sections, era tags, or analysis-specific flags.

## Keep the fileset small during development

Create reduced filesets for unit tests by limiting the number of files.

```python
# Take only first file from each dataset
mini_fileset = {}
for dataset, info in fileset.items():
    first_file = list(info["files"].items())[0]
    mini_fileset[dataset] = {
        "files": {first_file[0]: first_file[1]},
        "metadata": info.get("metadata", {}),
    }
```

Run your processor with `mini_fileset` locally, then switch back to the full fileset for production. You can also use `NanoEventsFactory.from_root(..., entry_stop=N)` to limit the number of events processed.

## Tips & tricks

- Use `NanoEventsFactory.from_root(..., entry_stop=N)` alongside a reduced fileset to validate schemas before launching large jobs.
- Preserve bookkeeping like `n_events` and the sum of generator weights in metadata so weight calculations stay consistent across processors.
- Keep JSON filesets under version control; they document exactly which inputs were analyzed.
- When combining many small files, merge them upstream if possible—large numbers of tiny files increase scheduler overhead during preprocessing.
