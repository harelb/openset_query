"""Shared pytest setup.

Tests marked ``gpu`` load real models (SigLIP2 / OpenCLIP weights, SAM3) and run
only with ``OPENSET_QUERY_MODEL_TESTS=1``. Tests that need Neo4j skip unless
``NEO4J_URI`` points at a throwaway database (they write to it). Tests that need
``hydra_python.image_store`` or ``heracles`` skip when those are not importable.
"""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("OPENSET_QUERY_MODEL_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="loads model weights (set OPENSET_QUERY_MODEL_TESTS=1)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
