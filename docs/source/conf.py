# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

import importlib
import inspect
import subprocess
import sys
from functools import reduce

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from pathlib import Path

import coffea

print("sys.path:", sys.path)
print("coffea version:", coffea.__version__)

# -- Project information -----------------------------------------------------

project = "coffea"
copyright = "2025, Fermi National Accelerator Laboratory"
author = "L. Gray, N. Smith, I. Krommydas et al. (The Coffea Team)"

version = coffea.__version__.rsplit(".", 1)[0]
release = coffea.__version__
githash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("ascii")

language = "en"

# -- General configuration ---------------------------------------------------

source_suffix = [".rst", ".md"]

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    # "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.graphviz",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "sphinx_copybutton",
    # 'IPython.sphinxext.ipython_console_highlighting',
]

# sphinx-copybutton configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_here_doc_delimiter = "EOF"

numpydoc_show_class_members = False
nbsphinx_execute = "never"
autosummary_generate = True
autosummary_imported_members = True

COFFEA_ROOT = Path(coffea.__file__).parent


def linkcode_resolve(domain, info: dict):
    if domain != "py":
        return None
    if not info.get("module", "").startswith("coffea"):
        return None
    mod = importlib.import_module(info["module"])
    try:
        obj = reduce(getattr, [mod] + info["fullname"].split("."))
    except AttributeError:
        return None
    try:
        path = inspect.getsourcefile(obj)
        if path is None:
            return None
        relpath = Path(path).relative_to(COFFEA_ROOT)
        _, lineno = inspect.getsourcelines(obj)
    except TypeError:
        # skip property or other type that inspect doesn't like
        return None
    url = f"http://github.com/scikit-hep/coffea/blob/{githash}/src/coffea/{relpath}#L{lineno}"
    return url


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "awkward": ("https://awkward-array.org/doc/main/", None),
    "dask-awkward": ("https://dask-awkward.readthedocs.io/en/stable/", None),
}

napoleon_preprocess_types = True

napoleon_type_aliases = {
    "awkward.Array": ":class:`awkward.Array <ak.Array>`",
    "awkward.Record": ":class:`awkward.Record <ak.Record>`",
    "awkward.highlevel.Array": ":class:`awkward.Array <ak.Array>`",
    "awkward.highlevel.Record": ":class:`awkward.Record <ak.Record>`",
    "dask_awkward.Array": ":class:`dask_awkward.Array <dask_awkward.Array>`",
    "dask_awkward.Record": ":class:`dask_awkward.Record <dask_awkward.Record>`",
    "dask_awkward.Scalar": ":class:`dask_awkward.Scalar <dask_awkward.Scalar>`",
}

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

default_role = "any"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
pygments_style = "sphinx"
html_theme = "pydata_sphinx_theme"
todo_include_todos = False
htmlhelp_basename = "coffeadoc"
html_logo = "logo/coffea_favicon.png"
html_favicon = "logo/coffea_favicon.png"

# -- MyST configuration -------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

# exclude for now
nb_execution_excludepatterns = ["notebooks/*"]

nb_execution_mode = "cache"
nb_execution_raise_on_error = True
nb_execution_show_tb = True


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "coffea.tex", "Coffea Documentation", "The Coffea Team", "manual"),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#
# latex_use_parts = False

# If true, show page references after internal links.
#
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
#
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
#
# latex_appendices = []

# It false, will not define \strong, \code,     itleref, \crossref ... but only
# \sphinxstrong, ..., \sphinxtitleref, ... To help avoid clash with user added
# packages.
#
# latex_keep_old_macro_names = True

# If false, no module index is generated.
#
# latex_domain_indices = True


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "coffea", "Coffea Documentation", [author], 1)]

# If true, show URL addresses after external links.
#
# man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "Coffea",
        "Coffea Documentation",
        author,
        "Coffea",
        "Efficient columnar HEP analysis in python.",
        "Miscellaneous",
    ),
]

# Documents to append as an appendix to all manuals.
#
# texinfo_appendices = []

# If false, no module index is generated.
#
# texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#
# texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
#
# texinfo_no_detailmenu = False
