API Reference Guide
*******************
Coffea: a column object framework for effective analysis.

When executing

    >>> import coffea

a subset of the full coffea package is imported into the python environment.
Some packages must be imported explicitly, so as to avoid importing unnecessary
and/or heavy dependencies.  Below lists the packages available in the ``coffea`` namespace.
Under that, we list documentation for some of the coffea packages that need to be
imported explicitly.

In coffea Namespace
-----------------------

.. autosummary::
    :toctree: modules
    :template: automodapi_templ.rst

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

Not in coffea Namespace
---------------------------
Here is documentation for some of the packages that are not automatically
imported on a call to ``import coffea``.

* :ref:`dataset-tools`.
