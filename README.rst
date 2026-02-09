coffea - Columnar Object Framework For Effective Analysis
=========================================================

.. image:: https://zenodo.org/badge/159673139.svg
   :target: https://zenodo.org/badge/latestdoi/159673139

.. image:: https://github.com/CoffeaTeam/coffea/workflows/CI%2FCD/badge.svg
    :target: https://github.com/CoffeaTeam/coffea/actions?query=workflow%3ACI%2FCD+event%3Aschedule+branch%3Amaster

.. image:: https://codecov.io/gh/CoffeaTeam/coffea/branch/master/graph/badge.svg?event=schedule
    :target: https://codecov.io/gh/CoffeaTeam/coffea

.. image:: https://badge.fury.io/py/coffea.svg
    :target: https://badge.fury.io/py/coffea

.. image:: https://img.shields.io/pypi/dm/coffea.svg
    :target: https://img.shields.io/pypi/dm/coffea

.. image:: https://img.shields.io/conda/vn/conda-forge/coffea.svg
    :target: https://anaconda.org/conda-forge/coffea

.. image:: https://badges.gitter.im/CoffeaTeam/coffea.svg
    :target: https://gitter.im/coffea-hep

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/CoffeaTeam/coffea/master?filepath=binder/

.. inclusion-marker-1-do-not-remove

Basic tools and wrappers for enabling not-too-alien syntax when running columnar Collider HEP analysis.

.. inclusion-marker-1-5-do-not-remove

coffea is a prototype package for pulling together all the typical needs
of a high-energy collider physics (HEP) experiment analysis using the scientific
python ecosystem. It makes use of `uproot <https://github.com/scikit-hep/uproot4>`_
and `awkward-array <https://github.com/scikit-hep/awkward-1.0>`_ to provide an
array-based syntax for manipulating HEP event data in an efficient and numpythonic
way. There are sub-packages that implement histogramming, plotting, and look-up
table functionalities that are needed to convey scientific insight, apply transformations
to data, and correct for discrepancies in Monte Carlo simulations compared to data.

coffea also supplies facilities for horizontally scaling an analysis in order to reduce
time-to-insight in a way that is largely independent of the resource the analysis
is being executed on. By making use of modern *big-data* technologies like
`Apache Spark <https://spark.apache.org/>`_,  `parsl <https://github.com/Parsl/parsl>`_,
`Dask <https://dask.org>`_ , and `Work Queue <http://ccl.cse.nd.edu/software/workqueue>`_,
it is possible with coffea to scale a HEP analysis from a testing
on a laptop to: a large multi-core server, computing clusters, and super-computers without
the need to alter or otherwise adapt the analysis code itself.

coffea is a HEP community project collaborating with `iris-hep <http://iris-hep.org/>`_
and is currently a prototype. We welcome input to improve its quality as we progress towards
a sensible refactorization into the scientific python ecosystem and a first release. Please
feel free to contribute at our `github repo <https://github.com/CoffeaTeam/coffea>`_!

.. inclusion-marker-2-do-not-remove

Installation
============

Install coffea like any other Python package:

.. code-block:: bash

    pip install coffea

or similar (use ``sudo``, ``--user``, ``virtualenv``, or pip-in-conda if you wish).
For more details, see the `Installing coffea <https://coffea-hep.readthedocs.io/en/backports-v0.7.x/installation.html>`_ section of the documentation.

Strict dependencies
===================

- `Python <http://docs.python-guide.org/en/latest/starting/installation/>`__ (3.6+)

The following are installed automatically when you install coffea with pip:

- `numpy <https://scipy.org/install.html>`__ (1.15+);
- `uproot <https://github.com/scikit-hep/uproot4>`__ for interacting with ROOT files and handling their data transparently;
- `awkward-array <https://github.com/scikit-hep/awkward-1.0>`__ to manipulate complex-structured columnar data, such as jagged arrays;
- `numba <https://numba.pydata.org/>`__ just-in-time compilation of python functions;
- `scipy <https://scipy.org/scipylib/index.html>`__ for many statistical functions;
- `matplotlib <https://matplotlib.org/>`__ as a plotting backend;
- and other utility packages, as enumerated in ``setup.py``.

Running the test suite
======================

The pytest suite ships with the minimal ROOT and parquet fixtures required to
exercise coffea's features, so it does not need network access or
experiment-specific packages.  After installing the development dependencies
simply run:

.. code-block:: bash

    pytest tests

The tests detect accidental imports of disallowed optional packages and fail
fast so that a clean environment continues to work out of the box.

.. inclusion-marker-3-do-not-remove

Configuring correctionlib-based JEC inputs
=========================================

``coffea.jetmet_tools.JECStack`` can be pointed at any correctionlib JSON or
an in-memory :class:`correctionlib.schemav2.CorrectionSet` without relying on
CVMFS defaults.  When running in environments without CVMFS, provide a
``resolver`` callable or string template that returns the path to your JSON,
or pass the already-loaded correction set directly:

.. code-block:: python

    from coffea.jetmet_tools import JECStack
    import correctionlib.schemav2 as cs

    # Use a path template
    stack = JECStack(
        use_clib=True,
        jec_tag="Summer22_V1",
        jec_levels=["L1", "L2L3"],
        jet_algo="AK4PFchs",
        resolver="/site/local/corrections/{jec_tag}_{jet_algo}.json",
    )

    # Or provide a fully-materialized correction set
    stack = JECStack(
        use_clib=True,
        jec_tag="Local",
        jec_levels=["L1"],
        jet_algo="AK4PF",
        correction_set=cs.CorrectionSet.from_file("./local.json"),
    )

The resolved path and correction set are then consumed automatically by
``CorrectedJetsFactory`` during evaluation.

When running in environments where the official ``JME-JSONs`` repository is
available (for example via CVMFS) you can ask ``JECStack`` to auto-discover the
correctionlib payload by providing the ``year`` and a list of search directories
or by setting the ``COFFEA_JME_JSONS`` environment variable:

.. code-block:: python

    stack = JECStack(
        use_clib=True,
        jec_tag="Summer22Run3_V1_MC",
        jec_levels=["L1", "L2L3"],
        jet_algo="AK8PFPuppi",
        year=2022,
        json_search_dirs=["/cvmfs/cms.cern.ch/rsync/cms-jet/JME-JSONs"],
    )

The stack sets ``json_path`` automatically based on ``{jec_tag}_{jet_algo}``
within the provided directory (preferring year-specific subdirectories), so
downstream factories do not need to construct algorithm-specific file names.

Requesting specific correction levels
=====================================

``CorrectedJetsFactory`` now exposes lightweight accessors for multiplying the
corrections up to a named level without materializing a column per step.  The
``correction_factors`` method returns an awkward array matching the jet
structure and accepts ``target_level`` strings that align with the
``jec_levels`` provided to ``JECStack`` (``"L1"``, ``"L2Relative"``, etc.) or
the fully qualified correction names:

.. code-block:: python

    jet_factory = CorrectedJetsFactory(name_map, jec_stack)
    jec_cache = cachetools.Cache(np.inf)

    # Retrieve the correction factors up to the L1 step
    l1_factors = jet_factory.correction_factors(
        jets,
        lazy_cache=jec_cache,
        target_level="L1",
    )

    # Build jets using only corrections up to L2
    partially_corrected = jet_factory.build(
        jets,
        lazy_cache=jec_cache,
        target_level="L2",
    )

If ``target_level`` is omitted the factory multiplies all available JEC levels
as before.

Documentation
=============
All documentation is hosted at https://coffea-hep.readthedocs.io/en/backports-v0.7.x/
