# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'llmprivacy'
copyright = '2024, Jason Wang, Jeffrey Wang, Marvin Li, Seth Neel'
author = 'Jason Wang, Jeffrey Wang, Marvin Li, Seth Neel'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["myst_parser","sphinx.ext.napoleon","sphinx.ext.autodoc","sphinx.ext.autosummary","sphinx.ext.todo","sphinx.ext.viewcode"]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_title = "Pandora's White-Box"
html_logo = "assets/pandy.png"