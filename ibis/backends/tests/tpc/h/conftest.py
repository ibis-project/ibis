from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def customer(backend):
    return backend.h("customer")


@pytest.fixture(scope="session")
def lineitem(backend):
    return backend.h("lineitem")


@pytest.fixture(scope="session")
def nation(backend):
    return backend.h("nation")


@pytest.fixture(scope="session")
def orders(backend):
    return backend.h("orders")


@pytest.fixture(scope="session")
def part(backend):
    return backend.h("part")


@pytest.fixture(scope="session")
def partsupp(backend):
    return backend.h("partsupp")


@pytest.fixture(scope="session")
def region(backend):
    return backend.h("region")


@pytest.fixture(scope="session")
def supplier(backend):
    return backend.h("supplier")
