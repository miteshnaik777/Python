"""
conftest.py
────────────
Pytest configuration — loaded automatically before any test file.
Registers module aliases so numbered directories are importable.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Register ingestion / preprocessing / embedding / retrieval / generation / pipeline
import pathsetup  # noqa: F401
