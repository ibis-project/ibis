from __future__ import annotations

import datetime
import functools
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import sqlglot as sg
from dateutil.relativedelta import relativedelta

import ibis
from ibis.formats.pandas import PandasData

if TYPE_CHECKING:
    from collections.abc import Callable

    import ibis.expr.types as ir


def pytest_pyfunc_call(pyfuncitem):
    """Inject `backend` and `snapshot` fixtures to all TPC-DS test functions.

    Defining this hook here limits its scope to the TPC-DS tests.
    """
    testfunction = pyfuncitem.obj
    funcargs = pyfuncitem.funcargs
    testargs = {arg: funcargs[arg] for arg in pyfuncitem._fixtureinfo.argnames}
    result = testfunction(
        **testargs, backend=funcargs["backend"], snapshot=funcargs["snapshot"]
    )
    assert (
        result is None
    ), "test function should not return anything, did you mean to use assert?"
    return True


def tpc_test(suite_name):
    def inner(test: Callable[..., ir.Table]):
        """Decorator for TPC tests.

        Automates the process of loading the SQL query from the file system and
        asserting that the result of the ibis expression is equal to the expected
        result of executing the raw SQL.
        """

        name = f"tpc{suite_name}"

        @getattr(pytest.mark, name)
        @pytest.mark.usefixtures("backend", "snapshot")
        @pytest.mark.xdist_group(name)
        @functools.wraps(test)
        def wrapper(*args, backend, snapshot, **kwargs):
            backend_name = backend.name()
            if not getattr(backend, f"supports_{name}"):
                pytest.skip(
                    f"{backend_name} backend doesn't support testing {name} queries yet"
                )
            query_name_match = re.match(r"^test_(\d\d)$", test.__name__)
            assert query_name_match is not None

            query_number = query_name_match.group(1)
            sql_path_name = f"{query_number}.sql"

            path = Path(__file__).parent.joinpath(
                "queries", "duckdb", suite_name, sql_path_name
            )
            raw_sql = path.read_text()

            sql = sg.parse_one(raw_sql, read="duckdb")

            transform_method = getattr(
                backend, f"_transform_{name}_sql", lambda sql: sql
            )
            sql = transform_method(sql)

            raw_sql = sql.sql(dialect="duckdb", pretty=True)

            expected_expr = backend.connection.sql(raw_sql, dialect="duckdb")

            result_expr = test(*args, **kwargs)

            ibis_sql = ibis.to_sql(result_expr, dialect=backend_name)

            assert result_expr._find_backend(use_default=False) is backend.connection
            result = backend.connection.execute(result_expr)
            assert not result.empty

            expected = expected_expr.execute()
            assert list(map(str.lower, expected.columns)) == result.columns.tolist()
            expected.columns = result.columns

            expected = PandasData.convert_table(expected, result_expr.schema())
            assert not expected.empty

            assert len(expected) == len(result)
            assert result.columns.tolist() == expected.columns.tolist()
            for column in result.columns:
                left = result.loc[:, column]
                right = expected.loc[:, column]
                assert (
                    pytest.approx(
                        left.values.tolist(),
                        nan_ok=True,
                        abs=backend.tpc_absolute_tolerance,
                    )
                    == right.values.tolist()
                )

            # only write sql if the execution passes
            snapshot.assert_match(ibis_sql, sql_path_name)

        return wrapper

    return inner


def add_date(datestr: str, dy: int = 0, dm: int = 0, dd: int = 0) -> ir.DateScalar:
    dt = datetime.date.fromisoformat(datestr)
    dt += relativedelta(years=dy, months=dm, days=dd)
    return ibis.date(dt.isoformat())


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
def ds_customer(backend):
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
