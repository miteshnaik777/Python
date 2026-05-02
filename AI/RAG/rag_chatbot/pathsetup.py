"""
pathsetup.py
─────────────
Registers namespace-package aliases so that numbered directories
(01_ingestion, 02_preprocessing, …) are importable under clean names.

    import ingestion        →  01_ingestion/
    import preprocessing    →  02_preprocessing/
    import embedding        →  03_embedding/
    import retrieval        →  04_retrieval/
    import generation       →  05_generation/
    import pipeline         →  06_pipeline/

This module is imported by:
  • config.py          (so every source file benefits via `from config import cfg`)
  • conftest.py        (so pytest picks it up automatically)

It is safe to import multiple times (guarded by sys.modules check).
"""

import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

_ALIASES = {
    "ingestion":    "01_ingestion",
    "preprocessing":"02_preprocessing",
    "embedding":    "03_embedding",
    "retrieval":    "04_retrieval",
    "generation":   "05_generation",
    "pipeline":     "06_pipeline",
}


def register():
    """Create namespace-package stubs in sys.modules for each alias."""
    for name, dirname in _ALIASES.items():
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(_ROOT / dirname)]
            pkg.__package__ = name
            pkg.__spec__ = None
            sys.modules[name] = pkg


register()
