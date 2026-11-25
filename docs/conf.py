"""Sphinx configuration of the user documentation.

    pip install ./python[docs]
    sphinx-build -b html docs docs/_build/html      # the site
    sphinx-build -b doctest docs docs/_build/doctest   # run the examples
"""

import datetime
import re

import cortical_bone_mapping_rs as cbm

project = "cortical-bone-mapping-rs"
author = "Nathan J. Neeteson"
copyright = f"2025–{datetime.date.today().year}, Nathan J. Neeteson"
release = cbm.__version__
version = ".".join(release.split(".")[:2])

REPO = "https://github.com/njneeteson/cortical-bone-mapping-rs"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build", "scripts", "results", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["colon_fence", "dollarmath", "deflist"]
myst_heading_anchors = 3

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
# the option classes have dozens of fields: one per line, with its default
python_maximum_signature_line_length = 88
autodoc_default_options = {"members": True, "undoc-members": False}
autosummary_generate = False
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

doctest_global_setup = """
import numpy as np
import cortical_bone_mapping_rs as cbm
"""

html_theme = "furo"
html_title = "cortical-bone-mapping-rs"
html_static_path = []
html_theme_options = {
    "source_repository": REPO,
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [],
}
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# The method and validation pages are also read on GitHub, where they link to
# the Rust sources, scripts and result files by relative path. Outside the
# documentation's source tree those links would break, so they point to the
# repository instead.
_REPO_LINK = re.compile(r"\]\((\.\./[^)#\s]+|(?:scripts|results)/[^)#\s]+)")


def _repo_links(app, docname, source):
    def to_repo(m):
        path = m.group(1)
        path = path[3:] if path.startswith("../") else "docs/" + path
        return f"]({REPO}/blob/main/{path}"

    source[0] = _REPO_LINK.sub(to_repo, source[0])


def setup(app):
    app.connect("source-read", _repo_links)
