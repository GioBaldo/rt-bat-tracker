import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "RealBat"
copyright = "2026, Giovanni Baldini"
author = "Giovanni Baldini"
release = "1.0.0"

extensions = [
    "sphinx.ext.autodoc",
]

toc_object_entries = False

autodoc_mock_imports = [
    "keyboard",
    "pyroomacoustics",
    "natsort",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]