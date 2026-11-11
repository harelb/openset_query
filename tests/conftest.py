"""Shared pytest setup.

Tests marked ``gpu`` load real models (SigLIP2 / OpenCLIP weights, SAM3) and run
only with ``OPENSET_QUERY_MODEL_TESTS=1``. Tests that need Neo4j skip unless
``NEO4J_URI`` points at a throwaway database (they write to it). Tests marked
``adt4`` need ``heracles`` and ``spark_dsg`` from the ADT4 workspace and skip
when either is not importable.
"""

import importlib.util
import os

import pytest


def _importable(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def pytest_collection_modifyitems(config, items):
    run_models = os.environ.get("OPENSET_QUERY_MODEL_TESTS") == "1"
    have_adt4 = _importable("heracles") and _importable("spark_dsg")
    no_models = pytest.mark.skip(
        reason="loads model weights (set OPENSET_QUERY_MODEL_TESTS=1)"
    )
    no_adt4 = pytest.mark.skip(reason="needs heracles + spark_dsg (ADT4 workspace)")
    for item in items:
        if "gpu" in item.keywords and not run_models:
            item.add_marker(no_models)
        if "adt4" in item.keywords and not have_adt4:
            item.add_marker(no_adt4)
