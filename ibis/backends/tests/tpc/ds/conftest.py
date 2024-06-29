from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def call_center(backend):
    return backend.ds("call_center")


@pytest.fixture(scope="session")
def catalog_page(backend):
    return backend.ds("catalog_page")


@pytest.fixture(scope="session")
def catalog_returns(backend):
    return backend.ds("catalog_returns")


@pytest.fixture(scope="session")
def catalog_sales(backend):
    return backend.ds("catalog_sales")


@pytest.fixture(scope="session")
def customer(backend):
    return backend.ds("customer")


@pytest.fixture(scope="session")
def customer_address(backend):
    return backend.ds("customer_address")


@pytest.fixture(scope="session")
def customer_demographics(backend):
    return backend.ds("customer_demographics")


@pytest.fixture(scope="session")
def date_dim(backend):
    return backend.ds("date_dim")


@pytest.fixture(scope="session")
def household_demographics(backend):
    return backend.ds("household_demographics")


@pytest.fixture(scope="session")
def income_band(backend):
    return backend.ds("income_band")


@pytest.fixture(scope="session")
def inventory(backend):
    return backend.ds("inventory")


@pytest.fixture(scope="session")
def item(backend):
    return backend.ds("item")


@pytest.fixture(scope="session")
def promotion(backend):
    return backend.ds("promotion")


@pytest.fixture(scope="session")
def reason(backend):
    return backend.ds("reason")


@pytest.fixture(scope="session")
def ship_mode(backend):
    return backend.ds("ship_mode")


@pytest.fixture(scope="session")
def store(backend):
    return backend.ds("store")


@pytest.fixture(scope="session")
def store_returns(backend):
    return backend.ds("store_returns")


@pytest.fixture(scope="session")
def store_sales(backend):
    return backend.ds("store_sales")


@pytest.fixture(scope="session")
def time_dim(backend):
    return backend.ds("time_dim")


@pytest.fixture(scope="session")
def warehouse(backend):
    return backend.ds("warehouse")


@pytest.fixture(scope="session")
def web_page(backend):
    return backend.ds("web_page")


@pytest.fixture(scope="session")
def web_returns(backend):
    return backend.ds("web_returns")


@pytest.fixture(scope="session")
def web_sales(backend):
    return backend.ds("web_sales")


@pytest.fixture(scope="session")
def web_site(backend):
    return backend.ds("web_site")
