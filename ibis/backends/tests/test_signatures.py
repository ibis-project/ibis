from __future__ import annotations

import inspect

import pytest

from ibis.backends import _FileIOHandler
from ibis.backends.tests.signature.typecheck import compatible

params = []

for module in [_FileIOHandler]:
    methods = list(filter(lambda x: not x.startswith("_"), dir(module)))
    for method in methods:
        params.append((_FileIOHandler, method))


@pytest.mark.parametrize("base_cls, method", params)
def test_signatures(base_cls, method, backend_cls):
    if not hasattr(backend_cls, method):
        pytest.skip(f"Method {method} not present in {backend_cls}, skipping...")

    base_sig = inspect.signature(getattr(base_cls, method))
    backend_sig = inspect.signature(getattr(backend_cls, method))

    # Usage is compatible(implementation_signature, defined_interface_signature, ...)
    assert compatible(backend_sig, base_sig, check_annotations=False)
