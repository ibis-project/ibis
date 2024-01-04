from __future__ import annotations

import datetime
import functools
import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import pytest
import sqlglot as sg
from dateutil.relativedelta import relativedelta

import ibis
from ibis.formats.pandas import PandasData

if TYPE_CHECKING:
    import ibis.expr.types as ir


def pytest_pyfunc_call(pyfuncitem):
    """Inject `backend` and `snapshot` fixtures to all TPC-H test functions.

    Defining this hook here limits its scope to the TPC-H tests.
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


def tpch_test(test: Callable[..., ir.Table]):
    """Decorator for TPCH tests.

    Automates the process of loading the SQL query from the file system and
    asserting that the result of the ibis expression is equal to the expected
    result of executing the raw SQL.
    """

    @pytest.mark.tpch
    @pytest.mark.usefixtures("backend", "snapshot")
    @pytest.mark.xdist_group("tpch")
    @functools.wraps(test)
    def wrapper(*args, backend, snapshot, **kwargs):
        backend_name = backend.name()
        if not backend.supports_tpch:
            pytest.skip(
                f"{backend_name} backend doesn't support testing TPC-H queries yet"
            )
        query_name_match = re.match(r"^test_tpc_(h\d\d)$", test.__name__)
        assert query_name_match is not None

        query_number = query_name_match.group(1)
        sql_path_name = f"{query_number}.sql"

        path = Path(__file__).parent.joinpath("queries", "duckdb", sql_path_name)
        raw_sql = path.read_text()

        sql = sg.parse_one(raw_sql, read="duckdb")

        sql = backend._transform_tpch_sql(sql)

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
                pytest.approx(left.values.tolist(), nan_ok=True)
                == right.values.tolist()
            )

        # only write sql if the execution passes
        snapshot.assert_match(ibis_sql, sql_path_name)

    return wrapper


def add_date(datestr: str, dy: int = 0, dm: int = 0, dd: int = 0) -> ir.DateScalar:
    dt = datetime.date.fromisoformat(datestr)
    dt += relativedelta(years=dy, months=dm, days=dd)
    return ibis.date(dt.isoformat())


@pytest.fixture(scope="session")
def customer(backend):
    return backend.customer


@pytest.fixture(scope="session")
def lineitem(backend):
    return backend.lineitem


@pytest.fixture(scope="session")
def nation(backend):
    return backend.nation


@pytest.fixture(scope="session")
def orders(backend):
    return backend.orders


@pytest.fixture(scope="session")
def part(backend):
    return backend.part


@pytest.fixture(scope="session")
def partsupp(backend):
    return backend.partsupp


@pytest.fixture(scope="session")
def region(backend):
    return backend.region


@pytest.fixture(scope="session")
def supplier(backend):
    return backend.supplier
