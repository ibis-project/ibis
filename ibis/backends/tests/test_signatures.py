from __future__ import annotations

import inspect

import pytest

from ibis.backends import (
    BaseBackend,
    CanCreateCatalog,
    CanCreateDatabase,
    CanListCatalog,
    CanListDatabase,
    _FileIOHandler,
)
from ibis.backends.sql import SQLBackend
from ibis.backends.tests.signature.typecheck import compatible

params = []

for module in [
    BaseBackend,
    CanCreateCatalog,
    CanCreateDatabase,
    CanListCatalog,
    CanListDatabase,
    _FileIOHandler,
]:
    methods = list(filter(lambda x: not x.startswith("_"), dir(module)))
    for method in methods:
        params.append((module, method))


@pytest.mark.parametrize("base_cls, method", params)
def test_signatures(base_cls, method, backend_cls):
    if not hasattr(backend_cls, method):
        pytest.skip(f"Method {method} not present in {backend_cls}, skipping...")

    if not callable(base_method := getattr(base_cls, method)):
        pytest.skip(
            f"Method {method} in {base_cls} isn't callable, can't grab signature"
        )

    base_sig = inspect.signature(base_method)
    backend_sig = inspect.signature(getattr(backend_cls, method))

    # Usage is compatible(implementation_signature, defined_interface_signature, ...)
    assert compatible(backend_sig, base_sig, check_annotations=False)


sql_backend_params = []

for module in [SQLBackend]:
    methods = list(filter(lambda x: not x.startswith("_"), dir(module)))
    for method in methods:
        sql_backend_params.append((module, method))


@pytest.mark.parametrize("base_cls, method", sql_backend_params)
def test_signatures_sql_backends(base_cls, method, backend_sql_cls):
    if not hasattr(backend_sql_cls, method):
        pytest.skip(f"Method {method} not present in {backend_sql_cls}, skipping...")

    if not callable(base_method := getattr(base_cls, method)):
        pytest.skip(
            f"Method {method} in {base_cls} isn't callable, can't grab signature"
        )

    base_sig = inspect.signature(getattr(base_method))
    backend_sig = inspect.signature(getattr(backend_sql_cls, method))

    # Usage is compatible(implementation_signature, defined_interface_signature, ...)
    assert compatible(backend_sig, base_sig, check_annotations=False)
