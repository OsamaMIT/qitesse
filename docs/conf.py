from importlib import metadata
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

project = "qitesse"
author = "Osama Elmahdy"
copyright = "2026, Osama Elmahdy"

try:
    release = metadata.version("qitesse")
except metadata.PackageNotFoundError:
    release = "0.0.0"

version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autosummary_imported_members = True
autodoc_member_order = "bysource"
autoclass_content = "both"
add_module_names = False
html_theme = "alabaster"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

try:
    import qitesse as _qitesse
except ImportError as exc:
    raise RuntimeError(
        "The qitesse extension must be installed before building the docs. "
        "Run `maturin develop --release` locally, or let Read the Docs install the project via `.readthedocs.yaml`."
    ) from exc

required_symbols = [
    "Gate",
    "Circuit",
    "CircuitSpec",
    "CompiledCircuit",
    "ExecutionContext",
    "Observable",
    "Parameter",
    "set_num_threads",
]

missing = [name for name in required_symbols if not hasattr(_qitesse, name)]
if missing and os.environ.get("QITESSE_DOCS_ALLOW_STALE_IMPORT") != "1":
    raise RuntimeError(
        "The imported qitesse package does not expose the current API required for docs generation. "
        f"Missing symbols: {', '.join(missing)}. "
        "Reinstall the local extension with `maturin develop --release` before building docs."
    )
