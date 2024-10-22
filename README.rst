.. image:: docs/source/logo/coffea_logo.svg
    :align: center
    :width: 250px
    :alt: logo


coffea - Columnar Object Framework For Effective Analysis
=========================================================

.. image:: https://zenodo.org/badge/159673139.svg
   :target: https://zenodo.org/badge/latestdoi/159673139

.. image:: https://github.com/CoffeaTeam/coffea/actions/workflows/ci.yml/badge.svg
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
    :target: https://matrix.to/#/#coffea-hep_community:gitter.im

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
For more details, see the `Installing coffea <https://coffeateam.github.io/coffea/installation.html>`_ section of the documentation.

Strict dependencies
===================

- `Python <http://docs.python-guide.org/en/latest/starting/installation/>`__ (3.8+)

The following are installed automatically when you install coffea with pip:

- `numpy <https://scipy.org/install.html>`__ (1.22+);
- `uproot <https://github.com/scikit-hep/uproot5>`__ for interacting with ROOT files and handling their data transparently;
- `awkward-array <https://github.com/scikit-hep/awkward>`__ to manipulate complex-structured columnar data, such as jagged arrays;
- `numba <https://numba.pydata.org/>`__ just-in-time compilation of python functions;
- `scipy <https://scipy.org/scipylib/index.html>`__ for many statistical functions;
- `matplotlib <https://matplotlib.org/>`__ as a plotting backend;
- and other utility packages, as enumerated in ``pyproject.toml``.

.. inclusion-marker-3-do-not-remove

Documentation
=============
All documentation is hosted at https://coffeateam.github.io/coffea/

Citation
========
If you would like to cite this code in your work, you can use the zenodo DOI indicated in ``CITATION.cff``, or the `latest DOI <https://zenodo.org/badge/latestdoi/159673139>`__. You may also cite the proceedings:

- "N. Smith et al 2020 EPJ Web Conf. 245 06012"
- "L. Gray et al 2023 J. Phys.: Conf. Ser. 2438 012033"
