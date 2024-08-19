from __future__ import annotations

import inspect

import pytest

from ibis.backends import (
    BaseBackend,
    CanCreateCatalog,
    CanCreateDatabase,
    CanListCatalog,
    CanListDatabase,
)
from ibis.backends.sql import SQLBackend
from ibis.backends.tests.signature.typecheck import compatible

SKIP_METHODS = ["do_connect", "from_connection"]


def _scrape_methods(modules):
    params = []
    for module in modules:
        methods = list(filter(lambda x: not x.startswith("_"), dir(module)))
        for method in methods:
            # Only test methods that are callable (so we can grab the signature)
            # and skip any methods that we don't want to check
            if method not in SKIP_METHODS and callable(getattr(module, method)):
                params.append((module, method))

    return params


params = _scrape_methods(
    [
        BaseBackend,
        CanCreateCatalog,
        CanCreateDatabase,
        CanListCatalog,
        CanListDatabase,
    ]
)


@pytest.mark.parametrize("base_cls, method", params)
def test_signatures(base_cls, method, backend_cls):
    if not hasattr(backend_cls, method):
        pytest.skip(f"Method {method} not present in {backend_cls}, skipping...")

    base_sig = inspect.signature(getattr(base_cls, method))
    backend_sig = inspect.signature(getattr(backend_cls, method))

    # Usage is compatible(implementation_signature, defined_interface_signature, ...)
    assert compatible(backend_sig, base_sig, check_annotations=False)


sql_backend_params = _scrape_methods([SQLBackend])


@pytest.mark.parametrize("base_cls, method", sql_backend_params)
def test_signatures_sql_backends(base_cls, method, backend_sql_cls):
    if not hasattr(backend_sql_cls, method):
        pytest.skip(f"Method {method} not present in {backend_sql_cls}, skipping...")

    base_sig = inspect.signature(getattr(base_cls, method))
    backend_sig = inspect.signature(getattr(backend_sql_cls, method))

    # Usage is compatible(implementation_signature, defined_interface_signature, ...)
    assert compatible(backend_sig, base_sig, check_annotations=False)
