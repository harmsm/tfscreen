import os
import sys

# -- Project information -----------------------------------------------------

project = "tfscreen"
copyright = "2026, Michael Harms"
author = "Michael Harms"

# -- Safe version retrieval --------------------------------------------------
# Get version without importing the package, which might have missing dependencies
# in the build environment.
sys.path.insert(0, os.path.abspath("../../src"))
version_path = os.path.join(os.path.abspath("../../src"), "tfscreen", "__version__.py")
version_ns = {}
with open(version_path, "r") as f:
    exec(f.read(), version_ns)
version = version_ns["__version__"]
release = version

# -- General configuration ---------------------------------------------------

# Mock heavy dependencies so documentation can be built without them
autodoc_mock_imports = [
    "numpy",
    "pandas",
    "matplotlib",
    "jax",
    "jaxlib",
    "flax",
    "optax",
    "numpyro",
    "numba",
    "scipy",
    "tqdm",
    "statsmodels",
    "yaml",
    "corner",
    "patsy",
    "pybktree",
    "pyfastx",
    "dill",
    "h5py",
]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# -- Napoleon configuration --------------------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None
