# Develop

This page is focused on getting you started with local (personal computer) development of an analysis.
As such, it will require a few example files to test run over.
Here, we use two example files so that we can demonstrate a common workflow in these analyses.
- [`nano_dimuon.root`](https://github.com/scikit-hep/coffea/raw/refs/heads/master/tests/samples/nano_dimuon.root): 40 events of data selected for muon pairs
- [`nano_dy.root`](https://github.com/scikit-hep/coffea/raw/refs/heads/master/tests/samples/nano_dy.root): 40 events of simulated [Drell-Yan](https://en.wikipedia.org/wiki/Drell%E2%80%93Yan_process)

Our overall goal is to reconstruct the Z mass peak by combining the 4-momenta of two muons with opposite charge
and calculating their mass.

## 0. Walk Before You Run
Coffea is focused on separating the code that does the Physics analysis
from the code that is just running the analysis over some amount of data.
For this example, I've done the same thing.
I have two files  `dimuonmass.py` containing the analysis code and `run-local.py`
which has the infrastructure to run the analysis.

:::{literalinclude} eg/0-dimuonmass.py
:caption: dimuonmass.py
:lineno-match:
:::

:::{literalinclude} eg/run-local.py
:caption: run-local.py
:lineno-match:
:::

Cool, let's see what happens when we run this.

:::{note}
I am technically running this example within the Coffea image for stability
and then using [denv](https://tomeichlersmith.github.io/denv/) to shorten
the contaier-running commands.
```
denv init coffeateam/coffea-dask-almalinux9:2026.9.0-py3.13
```
and then prefixing `python3` with `denv` everywhere below.

Using Coffea in this way should not be necessary, please open an issue
if you can show that your environment and the image are showing different
behavior despite using the same Coffea version.
:::

```bash
$ python3 run-local.py
Preprocessing 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2/2 [ 0:00:01 < 0:00:00 | 1.7 file/s ]
/usr/local/lib/python3.13/site-packages/coffea/nanoevents/schemas/nanoaod.py:297: RuntimeWarning: Missing
cross-reference index for LowPtElectron_electronIdx => Electron
  warnings.warn(
/usr/local/lib/python3.13/site-packages/coffea/nanoevents/schemas/nanoaod.py:297: RuntimeWarning: Missing
cross-reference index for LowPtElectron_genPartIdx => GenPart
  warnings.warn(
/usr/local/lib/python3.13/site-packages/coffea/nanoevents/schemas/nanoaod.py:297: RuntimeWarning: Missing
cross-reference index for LowPtElectron_photonIdx => Photon
  warnings.warn(
/usr/local/lib/python3.13/site-packages/coffea/nanoevents/schemas/nanoaod.py:297: RuntimeWarning: Missing
cross-reference index for FatJet_genJetAK8Idx => GenJetAK8
  warnings.warn(
/usr/local/lib/python3.13/site-packages/coffea/nanoevents/schemas/nanoaod.py:336: RuntimeWarning: Branch Photon_mass
already exists but its values will be replaced with 0.0
  warnings.warn(
/usr/local/lib/python3.13/site-packages/coffea/nanoevents/schemas/nanoaod.py:336: RuntimeWarning: Branch Photon_charge
already exists but its values will be replaced with 0.0
  warnings.warn(
Processing 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2/2 [ 0:00:01 < 0:00:00 | 1.4 chunk/s ]
{'bytesread': 186962, 'columns': [], 'entries': 80, 'processtime': 8.106231689453125e-06, 'chunks': 2}
{}
```
A few observations of note:
- There are a few warnings about missing cross-reference indices, but I am ignoring them because they pertain to branches that I will not be using in this analysis. Outputs later in this example will omit the them.
- The `Preprocessing` and `Processing` bars load live as the processing happens.
- The metrics report no columns being read but still many bytes read. This is because there are some internals that will be read no matter what.
- The final result output is empty because we didn't do anything!

In conclusion, we can successfully process events and our `run-local.py` script is able to run over the files without issue.

## 1. Count Muons
The main ingredient in our reconstruction of the mass peak is a pair of muons,
so let's start simple and just count the number of muons in the event.
We use {any}`ak.num` to count the number of `Muon` objects in each of the events.
There are many other helper functions available from `awkward` and `numpy` that
are tools in your toolbox when designing your analysis.

:::{literalinclude} eg/1-dimuonmass.py
:caption: dimuonmass.py
:lineno-match:
:::

which produces
```bash
# omitting pre-processing and its warnings
Processing 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2/2 [ 0:00:01 < 0:00:00 | 1.1 chunk/s ]
{'bytesread': 187230, 'columns': ['nMuon-offsets'], 'entries': 80, 'processtime': 0.022721052169799805, 'chunks': 2}
{'DYJets': {'h_nmuons': Hist(Integer(0, 10, name='nmuons'), storage=Double()) # Sum: 40.0, 'events': 40}, 'Data': {'h_nmuons': Hist(Integer(0, 10, name='nmuons'), storage=Double()) # Sum: 40.0, 'events': 40}}
```
Notice that the number of `bytesread` and `columns` have increased according to the fact that we have
now accessed some data from the input files.

You could use {any}`hist.axis.StrCategory` instead of putting `h_nmuons` inside the sub-dictionary.
The goal is to keep the separate datasets distinct and these two methods are equivalent.

:::{admonition} Using `StrCategory`
:class: tip, dropdown

Instead of keeping a 1D histogram in separate dictionaries indexed by `dataset`,
this entails keeping 2D histograms where one of the axes is the `dataset`.

The last few lines of the `process` method would look like
```python
h_nmuons = hist.Hist.new.StrCategory([],growth=True,name="dataset").Integer(0, 10, name="nmuons", label="N Muons").Double()
h_nmuons.fill(dataset = dataset, nmuons = ak.num(events.Muon))
return { "h_nmuons": h_nmuons }
```
:::

:::{admonition} Plotting Code
:class: note, dropdown

I ran this with `python3 plot.py h_nmuons`.
```{literalinclude} eg/plot.py
:caption: plot.py
:lineno-match:
```
:::

:::{figure} eg/h_nmuons.png
:alt: Image of Muon Count Histograms Separated by Dataset
:figwidth: 500px
:align: center

The filled muon count histograms plotted separately by dataset.
We should definitely be prepared for events that don't have exactly two muons!
:::

## 2. Calculate Mass
As mentioned before, the {any}`awkward` package provides many functions that do basic operations like
counting (e.g. {any}`ak.num`), arithmetic (e.g. {any}`ak.sum`), and combinatorics (e.g. {any}`ak.combinations`).
In order to recostruct the Z mass peak, we need two muons from the same event with opposite charge.
With {any}`awkward`, we do this by first getting all pairs of muons using {any}`ak.combinations` and
then selecting only those pairs that have opposite charges.

In the final step, we use the attached {any}`ak.behavior`[^1] which gives the 4-momenta special functions
(like the ``+`` operator calculating the 4-vector-sum and the ``mass`` function calculating the 4-momentum mass).

[^1]: Specifically, the attached behavior is {any}`vector._methods.MomentumProtocolLorentz` with some special additions via {any}`coffea.nanoevents.methods.vector.LorentzVector`.

:::{literalinclude} eg/2-dimuonmass.py
:caption: dimuonmass.py
:lineno-match:
:::

But I now get a confusing error!

```
Traceback (most recent call last):
  File "/usr/local/lib/python3.13/site-packages/coffea/processor/executor.py", line 1726, in _work_function
    out = processor_instance.process(events)
  File "/home/tom/code/cms/scikit-hep/coffea/docs/source/getting_started/eg/dimuonmass.py", line 22, in process
    h_mass.fill(mass=mass)
    ~~~~~~~~~~~^^^^^^^^^^^
  File "/usr/local/lib/python3.13/site-packages/hist/basehist.py", line 343, in fill
    return super().fill(*args, *data, weight=weight, sample=sample, threads=threads)
           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/site-packages/boost_histogram/histogram.py", line 968, in fill
    args_ars = _fill_cast(args)
  File "/usr/local/lib/python3.13/site-packages/boost_histogram/histogram.py", line 148, in _fill_cast
    return tuple(_fill_cast(a, inner=True) for a in value)
  File "/usr/local/lib/python3.13/site-packages/boost_histogram/histogram.py", line 148, in <genexpr>
    return tuple(_fill_cast(a, inner=True) for a in value)
                 ~~~~~~~~~~^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/site-packages/boost_histogram/histogram.py", line 151, in _fill_cast
    return np.asarray(value)
           ~~~~~~~~~~^^^^^^^
  File "/usr/local/lib/python3.13/site-packages/awkward/highlevel.py", line 1563, in __array__
    with ak._errors.OperationErrorContext(
         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        "numpy.asarray", (self,), {"dtype": dtype, "copy": copy}
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/usr/local/lib/python3.13/site-packages/awkward/_errors.py", line 79, in __exit__
    raise self.decorate_exception(exception_type, exception_value)
  File "/usr/local/lib/python3.13/site-packages/awkward/highlevel.py", line 1568, in __array__
    return convert_to_array(self._layout, dtype=dtype, copy=copy)
  File "/usr/local/lib/python3.13/site-packages/awkward/_connect/numpy.py", line 526, in convert_to_array
    out = ak.operations.to_numpy(layout, allow_missing=False)
  File "/usr/local/lib/python3.13/site-packages/awkward/_dispatch.py", line 66, in dispatch
    next(gen_or_result)
    ~~~~^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/site-packages/awkward/operations/ak_to_numpy.py", line 48, in to_numpy
    return _impl(array, allow_missing)
  File "/usr/local/lib/python3.13/site-packages/awkward/operations/ak_to_numpy.py", line 60, in _impl
    return numpy_layout.to_backend_array(allow_missing=allow_missing)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/site-packages/awkward/contents/content.py", line 1131, in to_backend_array
    return self._to_backend_array(allow_missing, backend)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/site-packages/awkward/contents/listoffsetarray.py", line 1900, in _to_backend_array
    return self.to_RegularArray()._to_backend_array(allow_missing, backend)
           ~~~~~~~~~~~~~~~~~~~~^^
  File "/usr/local/lib/python3.13/site-packages/awkward/contents/listoffsetarray.py", line 296, in to_RegularArray
    self._backend.maybe_kernel_error(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        self._backend[
        ^^^^^^^^^^^^^^
    ...<7 lines>...
        )
        ^
    )
    ^
  File "/usr/local/lib/python3.13/site-packages/awkward/_backends/backend.py", line 62, in maybe_kernel_error
    raise ValueError(self.format_kernel_error(error))
ValueError: cannot convert to RegularArray because subarray lengths are not regular (in compiled code: https://github.
com/scikit-hep/awkward/blob/awkward-cpp-56/awkward-cpp/src/cpu-kernels/awkward_ListOffsetArray_toRegularArray.cpp#L22)

This error occurred while calling

    numpy.asarray(
        <Array [[], [], [], [], ..., [], [], []] type='40 * var * float32[p...'>
        dtype = None
        copy = None
    )

The above exception was the direct cause of the following exception:

Traceback (most recent call last):                                                                   13:00:13 [30/310]
  File "/home/tom/code/cms/scikit-hep/coffea/docs/source/getting_started/eg/run-local.py", line 33, in <module>
    result, metrics = runner(fileset, processor_instance=DiMuonMassCalculator())
                      ~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/site-packages/coffea/processor/executor.py", line 1843, in __call__
    wrapped_out = self.run(
        fileset=fileset,
    ...<4 lines>...
        trace=trace,
    )
  File "/usr/local/lib/python3.13/site-packages/coffea/processor/executor.py", line 2016, in run
    return self._run(
           ~~~~~~~~~^
        fileset,
        ^^^^^^^^
    ...<4 lines>...
        trace=trace,
        ^^^^^^^^^^^^
    )
    ^
  File "/usr/local/lib/python3.13/site-packages/coffea/processor/executor.py", line 2085, in _run
    out = self._run_impl(
        fileset,
    ...<4 lines>...
        trace=trace,
    )
  File "/usr/local/lib/python3.13/site-packages/coffea/processor/executor.py", line 2209, in _run_impl
    wrapped_out, e = executor(chunks, closure, None)
                     ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/site-packages/coffea/processor/executor.py", line 554, in __call__
    accumulate(
    ~~~~~~~~~~^
        progress.track(
        ^^^^^^^^^^^^^^^
    ...<4 lines>...
        accumulator,
        ^^^^^^^^^^^^
    ),
    ^
  File "/usr/local/lib/python3.13/site-packages/coffea/processor/accumulator.py", line 97, in accumulate
    accum = next(gen)
  File "/usr/local/lib/python3.13/site-packages/coffea/processor/accumulator.py", line 94, in <genexpr>
    gen = (x for x in items if x is not None)
                      ^^^^^
  File "/usr/local/lib/python3.13/site-packages/rich/progress.py", line 1232, in track
    for value in sequence:
                 ^^^^^^^^
  File "/usr/local/lib/python3.13/site-packages/coffea/processor/executor.py", line 1321, in automatic_retries
    raise e
  File "/usr/local/lib/python3.13/site-packages/coffea/processor/executor.py", line 1308, in automatic_retries
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.13/site-packages/coffea/processor/executor.py", line 1730, in _work_function
    raise Exception(
        f"Failed processing file: {item!r}. The error was: {e!r}."
    ) from e
Exception: Failed processing file: WorkItem(dataset='DYJets', filename='nano_dy.root', treename='Events', entrystart=0, entrystop=40, fileuuid=b'\xa9I\x01$6H\x11\xea\x89\xe9\xf5\xb5\\\x90\xbe\xef', usermeta={'is_mc': True}, preload=None). The error was: ValueError('cannot convert to RegularArray because subarray lengths are not regular (in compiled code: https://github.com/scikit-hep/awkward/blob/awkward-cpp-56/awkward-cpp/src/cpu-kernels/awkward_ListOffsetArray_toRegularArray.cpp#L22)').
```
This is a long and ugly error message, but I think it is important to not ignore it.
There are a few key features that I want to make sure you observe:
1. The first few lines show where the error originated from. The line that I wrote and is included in the error is `h_mass.fill(mass=mass)`, so I know that it has something to do with filling the histogram of masses.
2. The actual error is "cannot convert to RegularArray because subarray lengths are not regular". This error message is repeated in a few places, but is also included in the final exception report at the bottom along with the specific ``WorkItem`` where it originated (you could use this to test code on the specific file that caused the exception if you wanted).

The histogram package is being intentionally cautious.
It is requiring a {any}`ak.contents.RegularArray` so that the filling of the histogram is simply defined.
Put another way: if the array being filled into the histogram was ragged (different lengths per event), then how should the histogram be filled? All of the values? Just the first one in each event? This open question can and should be answered by you.
In this case, I want to include all pairs of opposite-sign muons _even if there is more than one pair per event_. I can include all pairs by intentionally flattening the array into a one-dimensionally array so that it is ready to fill (see {any}`ak.flatten` for details).

:::{literalinclude} eg/3-dimuonmass.py
:caption: dimuonmass.py
:lineno-match:
:::

which now runs

```
Processing 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2/2 [ 0:00:01 < 0:00:00 | 1.5 chunk/s ]
{'bytesread': 190147, 'columns': ['Muon_phi-data', 'Muon_pt-data', 'Muon_mass-data', 'nMuon-offsets', 'Muon_charge-data', 'Muon_eta-data'], 'entries': 80, 'processtime': 0.22384929656982422, 'chunks': 2}
{'DYJets': {'h_nmuons': Hist(Integer(0, 10, name='nmuons', label='N Muons'), storage=Double()) # Sum: 40.0, 'h_mass': Hist(Regular(60, 60, 120, name='mass', label='mμμ [GeV]'), storage=Double()) # Sum: 5.0 (6.0 with flow), 'events': 40}, 'Data': {'h_nmuons': Hist(Integer(0, 10, name='nmuons', label='N Muons'), storage=Double()) # Sum: 40.0, 'h_mass': Hist(Regular(60, 60, 120, name='mass', label='mμμ [GeV]'), storage=Double()) # Sum: 6.0 (38.0 with flow), 'events': 40}}
```

although the produced histogram is pretty lightly-filled.

:::{figure} eg/h_mass.png
:alt: Image of DiMuon mass within the Z-peak region.
:figwidth: 500px
:align: center

The fill dimoun mass histogram plotted separately by dataset.
:::

## Further Developments
In order to make this analysis more well-prepared for real data at scale, there are further
developments that would be required.

- Select muons based on their "tight ID" so that we are very confident they are muons
- Use {any}`correctionlib` to calculate weights for the histograms based off the muon's kinematic properties
