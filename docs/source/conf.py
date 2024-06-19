# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'llmprivacy'
author = 'Jason Wang, Jeffrey Wang, Marvin Li, Seth Neel'
copyright = f'2024 {author}'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "autoapi.extension",
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_title = "Pandora's White-Box"
html_logo = "assets/pandora.png"

# -- Options for extensions ---------------------------------------------------
autoapi_dirs = ['../../src']
autoapi_keep_files = True
autodoc_typehints = 'description'