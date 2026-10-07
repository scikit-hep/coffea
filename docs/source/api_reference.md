# API Reference Guide

Coffea: a column object framework for effective analysis.

When executing

```python
import coffea
```

a subset of the full coffea package is imported into the python environment.
Some packages must be imported explicitly, so as to avoid importing unnecessary
and/or heavy dependencies.  Below lists the packages available in the `coffea` namespace.
Under that, we list documentation for some of the coffea packages that need to be
imported explicitly.


## In coffea Namespace

```{eval-rst}
.. autosummary::
    :toctree: modules
    :caption: Available by Default
    :template: automodapi_templ.rst
    :recursive:

    coffea.analysis_tools
    coffea.btag_tools
    coffea.dataset_tools
    coffea.jetmet_tools
    coffea.lookup_tools
    coffea.lumi_tools
    coffea.ml_tools
    coffea.nanoevents
    coffea.nanoevents.methods.base
    coffea.nanoevents.methods.candidate
    coffea.nanoevents.methods.nanoaod
    coffea.nanoevents.methods.vector
    coffea.processor
    coffea.util
```

## Not in coffea Namespace

Here is documentation for some of the packages that are not automatically
imported on a call to `import coffea`.

This page contains documentation for parts of the `coffea.dataset_tools`
package that are not included in the `coffea` namespace. That is, they
must be explicitly imported.

```{eval-rst}
.. autosummary::
    :toctree: modules
    :caption: Need to Manually Import
    :template: automodapi_templ.rst
    :recursive:

    coffea.dataset_tools.dataset_query
    coffea.dataset_tools.rucio_utils
    coffea.processor.dask
```
