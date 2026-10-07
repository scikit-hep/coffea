# Scale

Scaling does not require modifying the processor, but it does require more thoughtful interaction
with the computing resources you are using.

:::{admonition} Setup at FNAL LPC
:class: note, dropdown

For this example, I am using FNAL LPC; however, the specific configuration
of these jobs and how they interact with `coffea` changes depending on your
cluster, so I am keeping this hidden as extra detail.

### Assumptions
1. You have a valid FNAL LPC Kerberos account (for connecting to the cluster)
2. You have a valid Grid Certificate (for accessing CMS simulation samples)

### Initial Connection
I did some experimentation to double check that I could validate my grid
certificate and maintain access to the simulation samples from within the
coffea image I wanted to use (the same image I used for {doc}`develop`).

```{literalinclude} eg/run
:caption: run
:lineno-match:
```

I double checked that this works by initializing my proxy with
```
voms-proxy-init -voms cms -rfc --valid 168:0
```
and then checking that the proxy is found both outside and inside the container.
```
# outside
$ voms-proxy-info
# inside
$ ./run voms-proxy-info
```
with the only change between the two outputs being the `timeleft`.

### Constructing Fileset
I use `coffea.dataset_tools.dataset_query` to construct a fileset from a
simulation sample that is moderately large (~60GB, 180 files).
In summary (only including the major commands)
```
./run python3 -m coffea.dataset_tools.dataset_query --cli
>>> query
>>> select
>>> replicas
>>> save
```
which produces a JSON file that I can load as the fileset I want
to process.

:::

The specific configuration of the backing service depends on the computing resources you have access to.
As a first pass, in my situation, I start with a smaller dataset (~60GB with 180 files),
so I am just using {class}`~coffea.processor.FuturesExecutor` on one the interactive node I am connected to.

:::{literalinclude} eg/futures.py
:caption: futures.py
:lineno-match:
:::

and the output looks like

```
  Preprocessing 100% ──────────────────────────────────────────────────── 180/180 [ 0:06:30 < 0:00:00 | 0.5   file/s ]
Merging (local)   1% ────────────────────────────────────────────────────   1/180 [ 0:06:30 < -:--:-- | ?   merges/s ]
     Processing   0% ──────────────────────────────────────────────────────── 0/357 [ 0:00:08 < -:--:-- | ?  chunk/s ]
Merging (local)   0% ────────────────────────────────────────────────────────   0/0 [ 0:00:08 < -:--:-- | ? merges/s ]
/usr/local/lib/python3.13/site-packages/coffea/nanoevents/schemas/nanoaod.py:297: RuntimeWarning: Missing cross-reference index for LowPtElectron_electronIdx => Electron
  warnings.warn(
/usr/local/lib/python3.13/site-packages/coffea/nanoevents/schemas/nanoaod.py:297: RuntimeWarning: Missing cross-reference index for LowPtElectron_photonIdx => Photon
  warnings.warn(
/usr/local/lib/python3.13/site-packages/coffea/nanoevents/schemas/nanoaod.py:336: RuntimeWarning: Branch Photon_mass already exists but its values will be replaced with 0.0
  warnings.warn(
/usr/local/lib/python3.13/site-packages/coffea/nanoevents/schemas/nanoaod.py:336: RuntimeWarning: Branch Photon_charge already exists but its values will be replaced with 0.0
     Processing 100% ──────────────────────────────────────────────────── 357/357 [ 0:27:59 < 0:00:00 | 0.3  chunk/s ]
Merging (local)   0% ────────────────────────────────────────────────────   1/357 [ 0:27:59 < -:--:-- | ?   merges/s ]
{'bytesread': 546564363, 'columns': ['Muon_eta-data', 'Muon_charge-data', 'Muon_phi-data', 'nMuon-offsets', 'Muon_mass-data', 'Muon_pt-data'], 'entries': 40192574, 'processtime': 1058.1506989002228, 'chunks': 357}
{'/DYtoLL_NoTau_CP5Plus_13p6TeV_amcatnloFXFX-pythia8/Run3Winter22NanoAOD-Pilot_122X_mcRun3_2021_realistic_v9-v2/NANOAODSIM': {'h_nmuons': Hist(Integer(0, 10, name='nmuons', label='N Muons'), storage=Double()) # Sum: 40192559.0 (40192574.0 with flow), 'h_mass': Hist(Regular(60, 60, 120, name='mass', label='m__ [GeV]'), storage=Double()) # Sum: 10022911.0 (12625272.0 with flow), 'events': 40192574}}
```

And the output ``dimuonmass.coffea`` can be plotted in the same way as before!

The preprocessing step where the fileset is cut up into processing chunks took a significant amount of time (6.5 minutes relative to 28min for the actual processing).
<!-- TODO link to docs on avoiding/caching preprocessing. -->

:::{admonition} I/O Bound
:class: tip

This is where I feel obligated to make an important point:
a vast majority of analyses are limited in time not by the computer doing calculations but by loading data into memory (so-called "I/O Bound").
Specifically, when viewing `htop -u ${USER}` while the above analysis was running, I observed classic I/O-bound-behavior where the sub-processes launched to do the analysis in parallel (`popen_loky_process` in this case) were mostly in the **S**uspended state and the CPU/Mem usage was "pulsing" instead of remaining high.
This is important to keep in mind because your analysis will likely not speed up linearly with the number of parallel processes you use (in some situations, it might even slow down).
<!-- TODO skimming user guide for shrinking data necessary for analyses -->
:::

You can follow the same pattern with  {class}`~coffea.processor.ParslExecutor`, {class}`~coffea.processor.TaskVineExecutor`, or {class}`~coffea.processor.DaskExecutor` (via [``dask_jobqueue``](https://jobqueue.dask.org/en/latest/)) depending on the scheduler available at the computing resources you are using.
If you are working on computing resources that does not already have an executor configuration available, I highly recommend developing one with a _simple_ analysis (e.g. just counting the number of events) in order to make sure all confusions are originating from the executor definition.
