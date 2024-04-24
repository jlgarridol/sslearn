# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'Semi-Supervised Learning Library'
copyright = '2024, J.L. Garrido-Labrador'
author = 'José Luis Garrido-Labrador'

release = '1.0.4.1'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']


templates_path = ['_templates']

# -- Options for EPUB output
