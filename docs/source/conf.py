# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "llmprivacy"
author = "Jeffrey Wang, Jason Wang Marvin Li, Seth Neel"
copyright = f"2024 {author}. Docs written by Jason Wang."

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "autoapi.extension",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_title = "Pandora's White-Box"
html_logo = "assets/pandora.png"

# -- Options for extensions ---------------------------------------------------
autoapi_dirs = ["../../src"]
autoapi_keep_files = True
autodoc_typehints = "description"

autoapi_options = [
    'members',
    'inherited-members',
    'undoc-members',
    'private-members',
    'show-inheritance',
]

autoapi_ignore = ["*/routines/*.py"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("http://matplotlib.org/stable", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://pytorch.org/docs/master/", None),
    "transformers": ("https://huggingface.co/docs/transformers/master/en/", None),
    "datasets": ("https://huggingface.co/docs/datasets/master/en/", None),
    "accelerate": ("https://huggingface.co/docs/accelerate/master/en/", None),
}