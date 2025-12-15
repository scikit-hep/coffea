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

`coffea.dataset_tools.rucio_utils` helps when the list of files lives in Rucio or in JSON manifests.

```python
from coffea.dataset_tools.rucio_utils import get_dataset_files_replicas

outfiles, outsites, sites_counts = get_dataset_files_replicas(
    dataset="/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
)

# chose site from which to extract files
site = max(sites_counts, key=sites_counts.get)
files_by_site = []
for i, (files, sites) in enumerate(zip(outfiles, outsites)):
    iS = sites.index(site)
    files_by_site.append(files[iS])

# Convert list to dict mapping files to tree name (Format 1)
fileset = {
    "DYJets": {
        "files": {f: "Events" for f in files_by_site},
        "metadata": {"is_mc": True},
    }
}

# Or use Format 2 if all files have same tree
fileset_alt = {
    "DYJets": {
        "treename": "Events",
        "files": files_by_site,
        "metadata": {"is_mc": True},
    }
}
```

Output example:

```python
print("site:\n", site)
print("\nfileset:\n", fileset)
print("\nfileset_alt:\n", fileset_alt)
```
```python
site:
 T2_DE_DESY

fileset:
 {'DYJets':
    {'files':
      {'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/0CF0CDED-7582-7A49-84CD-\
        0E5F73DE27B0.root': 'Events',
       'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/0E17EB7B-5779-2741-ABBE-\
        3465CC7C6174.root': 'Events',
       'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/1332F4CB-CEBC-0C4C-BA1E-\
        2D82BD7F7294.root': 'Events',
       'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/1778E98F-966D-204F-9F3F-\
        61DB3DB07616.root': 'Events',
       'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/2060156F-8F7B-5B4C-B2BE-\
        7C7823494E17.root': 'Events',
       'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/3415A617-3335-B341-94D8-\
        809C1B8012A7.root': 'Events',
       'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/42217818-CFFD-8B42-B357-\
        B7D1CE8881D1.root': 'Events',
       'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/53CB5FBD-F338-1F48-A516-\
        3862AC0F87B1.root': 'Events',
       'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/7140ABB2-6BB4-BF4D-B469-\
        9C3C9E01C566.root': 'Events',
       'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/78AB9B61-1B4D-7644-89A7-\
        42945E27D39F.root': 'Events',
       'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/90D780AD-4691-5749-98FF-\
        089926B03C3C.root': 'Events',
       'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/93820618-447B-844C-96FE-4A2488ED5170.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/A05BE01E-9983-0C42-B981-D31FEC2A02D1.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/B0CC5BCA-2D15-CA40-B246-0BFBBC5BD092.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/BF8F5312-B200-B747-9DAF-5ABE838BEA43.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/CD897D02-8EE0-0242-80DE-55A4035A661C.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/130000/5207D412-4107-AA4B-AEB8-06B26220CABB.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/130000/7686FE0F-FCF2-B442-91EC-B2C6328A357D.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/130000/8F22F184-B47D-EB4A-9D9D-07C61386A83C.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/130000/AD708B53-933A-3A46-94FF-18878BB92500.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/130000/B4ECDD8B-3754-214B-941A-8CCD3A1B6BD3.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/130000/C33EB33B-6B49-6B4E-AB95-164DA6CAAD29.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/130000/D9CC0CEB-169D-C545-BA78-3365CB8EA7AD.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/130000/F7EC55C9-049F-5741-972E-6987DAAEDFC7.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/270000/593919B4-087D-7A45-8DC4-3FA23EF86339.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/270000/7C41B936-4CF8-DB45-9843-99BC1ADC2C9D.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/0DF07209-9EE8-4745-9069-F794CA99A000.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/1D8ECE43-77D2-5044-A54F-D7488C4C2DA8.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/1EC72A3E-A224-A741-A2A7-C60D90C56BE4.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/206720BF-5E3C-0F4F-8F4C-6876D1F6C5D8.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/2321D705-4197-3A45-B1F5-74C34CC913B1.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/2CA53FA6-AF6F-B346-A377-4F52020AC9B6.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/48323732-DF4E-1E45-A48E-1777E1C65C65.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/50EBE44A-C5C9-C64E-9BF3-86D4B8378694.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/525CD279-3344-6043-98B9-2EA8A96623E4.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/5DE99625-BE66-254E-80AF-EE2396DF41A1.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/64453944-A238-884D-B867-03F10926EA50.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/675CB76D-11B1-574E-B562-15D746D5341E.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/69341C6D-45FC-5D48-8D12-A0CA319E31FE.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/707AE805-DF14-D746-BEF5-43176841D4AE.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/70A6B363-21C6-4843-A380-279F418E3F99.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/73680BE7-4A24-4F4B-B849-0D1F53A92ED3.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/7549D452-F82C-3F4B-9D5D-550685AAD5FB.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/79CBCFF6-EE37-D34B-AF8B-1AC9A2562EB1.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/7B7D90CB-14EF-B749-B4D7-7C413FE3CCC1.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/7BB46D39-06FF-9542-A394-9BD10ECC3F8E.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/7CDCD4D5-9EE3-714B-8E16-6AD72EFAA35A.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/95537AFE-1978-A847-A49E-5ABDE5F9D8B5.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/98AFA7C8-2183-C844-84B6-6853B7EF82F2.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/A2EE4CB1-B11F-EC47-87D0-BCE91A404A1E.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/B7604511-4086-EF45-A4A8-583B3DD06DB8.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/BE38AA86-BFE0-314E-B104-E7D39B1B63A1.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/D2417B78-FA20-414E-9014-77B15EF86677.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/E090B3F5-8055-A84A-8689-5E9352C7D003.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/F53719F8-34E3-E344-A664-DC01F683A873.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/F630411C-8507-7F42-877F-775BC8EB0812.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/FD9FCF36-AA98-BC42-B78D-CD984051BCB7.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/2E9E5056-0964-7941-8A74-A303A364EEA0.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/40EC0EE5-67F3-4A44-8CC3-7F7E79ED8CF0.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/4CBA4FB4-2008-E34E-9465-6C232A5C651B.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/502F0DB0-2DB0-CB4A-B00B-D21B2D83A6EC.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/62FA4765-0770-CF4B-A2B9-E647DEEB42D7.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/66A48386-9930-254A-A4F7-59727337DE39.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/70047B17-3D2A-1348-B9B2-7F48261BD647.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/90BF253F-0DD5-654D-ABED-1F0E2A6A27CC.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/9601A635-AB30-F343-9F2B-F8D5B1133570.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/B715A9DC-A458-3946-B3F6-34A0A8F44766.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/DB8ED6E4-3FB5-F44F-98BD-9793D0862B10.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/E76A9FA9-712E-1946-9B4E-6CE4E37A2A96.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/E966C645-78B0-0347-B2CB-DBA87B3ED986.root': 'Events', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/F7FF5063-82CD-5043-AF66-00D94D3A2FBD.root': 'Events'
      },
    'metadata': {'is_mc': True}
    }
}

fileset_alt:
 {'DYJets':
   {'treename': 'Events',
    'files': ['root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/0CF0CDED-7582-7A49-84CD-0E5F73DE27B0.root',
              'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/0E17EB7B-5779-2741-ABBE-3465CC7C6174.root',
              'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/1332F4CB-CEBC-0C4C-BA1E-2D82BD7F7294.root',
              'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/1778E98F-966D-204F-9F3F-61DB3DB07616.root',
              'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/2060156F-8F7B-5B4C-B2BE-7C7823494E17.root',
              'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/3415A617-3335-B341-94D8-809C1B8012A7.root',
              'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/42217818-CFFD-8B42-B357-B7D1CE8881D1.root',
              'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/53CB5FBD-F338-1F48-A516-3862AC0F87B1.root',
              'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/7140ABB2-6BB4-BF4D-B469-9C3C9E01C566.root',
              'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/78AB9B61-1B4D-7644-89A7-42945E27D39F.root',
              'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/90D780AD-4691-5749-98FF-089926B03C3C.root',
              'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/93820618-447B-844C-96FE-4A2488ED5170.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/A05BE01E-9983-0C42-B981-D31FEC2A02D1.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/B0CC5BCA-2D15-CA40-B246-0BFBBC5BD092.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/BF8F5312-B200-B747-9DAF-5ABE838BEA43.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/120000/CD897D02-8EE0-0242-80DE-55A4035A661C.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/130000/5207D412-4107-AA4B-AEB8-06B26220CABB.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/130000/7686FE0F-FCF2-B442-91EC-B2C6328A357D.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/130000/8F22F184-B47D-EB4A-9D9D-07C61386A83C.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/130000/AD708B53-933A-3A46-94FF-18878BB92500.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/130000/B4ECDD8B-3754-214B-941A-8CCD3A1B6BD3.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/130000/C33EB33B-6B49-6B4E-AB95-164DA6CAAD29.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/130000/D9CC0CEB-169D-C545-BA78-3365CB8EA7AD.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/130000/F7EC55C9-049F-5741-972E-6987DAAEDFC7.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/270000/593919B4-087D-7A45-8DC4-3FA23EF86339.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/270000/7C41B936-4CF8-DB45-9843-99BC1ADC2C9D.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/0DF07209-9EE8-4745-9069-F794CA99A000.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/1D8ECE43-77D2-5044-A54F-D7488C4C2DA8.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/1EC72A3E-A224-A741-A2A7-C60D90C56BE4.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/206720BF-5E3C-0F4F-8F4C-6876D1F6C5D8.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/2321D705-4197-3A45-B1F5-74C34CC913B1.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/2CA53FA6-AF6F-B346-A377-4F52020AC9B6.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/48323732-DF4E-1E45-A48E-1777E1C65C65.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/50EBE44A-C5C9-C64E-9BF3-86D4B8378694.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/525CD279-3344-6043-98B9-2EA8A96623E4.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/5DE99625-BE66-254E-80AF-EE2396DF41A1.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/64453944-A238-884D-B867-03F10926EA50.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/675CB76D-11B1-574E-B562-15D746D5341E.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/69341C6D-45FC-5D48-8D12-A0CA319E31FE.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/707AE805-DF14-D746-BEF5-43176841D4AE.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/70A6B363-21C6-4843-A380-279F418E3F99.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/73680BE7-4A24-4F4B-B849-0D1F53A92ED3.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/7549D452-F82C-3F4B-9D5D-550685AAD5FB.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/79CBCFF6-EE37-D34B-AF8B-1AC9A2562EB1.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/7B7D90CB-14EF-B749-B4D7-7C413FE3CCC1.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/7BB46D39-06FF-9542-A394-9BD10ECC3F8E.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/7CDCD4D5-9EE3-714B-8E16-6AD72EFAA35A.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/95537AFE-1978-A847-A49E-5ABDE5F9D8B5.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/98AFA7C8-2183-C844-84B6-6853B7EF82F2.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/A2EE4CB1-B11F-EC47-87D0-BCE91A404A1E.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/B7604511-4086-EF45-A4A8-583B3DD06DB8.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/BE38AA86-BFE0-314E-B104-E7D39B1B63A1.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/D2417B78-FA20-414E-9014-77B15EF86677.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/E090B3F5-8055-A84A-8689-5E9352C7D003.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/F53719F8-34E3-E344-A664-DC01F683A873.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/F630411C-8507-7F42-877F-775BC8EB0812.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/FD9FCF36-AA98-BC42-B78D-CD984051BCB7.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/2E9E5056-0964-7941-8A74-A303A364EEA0.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/40EC0EE5-67F3-4A44-8CC3-7F7E79ED8CF0.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/4CBA4FB4-2008-E34E-9465-6C232A5C651B.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/502F0DB0-2DB0-CB4A-B00B-D21B2D83A6EC.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/62FA4765-0770-CF4B-A2B9-E647DEEB42D7.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/66A48386-9930-254A-A4F7-59727337DE39.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/70047B17-3D2A-1348-B9B2-7F48261BD647.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/90BF253F-0DD5-654D-ABED-1F0E2A6A27CC.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/9601A635-AB30-F343-9F2B-F8D5B1133570.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/B715A9DC-A458-3946-B3F6-34A0A8F44766.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/DB8ED6E4-3FB5-F44F-98BD-9793D0862B10.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/E76A9FA9-712E-1946-9B4E-6CE4E37A2A96.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/E966C645-78B0-0347-B2CB-DBA87B3ED986.root', 'root://dcache-cms-xrootd.desy.de:1094//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/F7FF5063-82CD-5043-AF66-00D94D3A2FBD.root'
             ],
      'metadata': {'is_mc': True}
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
