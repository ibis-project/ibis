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

if TYPE_CHECKING:
    import ibis.expr.types as ir


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

        expected_expr = backend.connection.sql(
            # in theory this should allow us to use one dialect for every backend
            raw_sql,
            dialect="duckdb",
        )

        result_expr = test(*args, **kwargs)

        ibis_sql = ibis.to_sql(result_expr, dialect=backend_name)

        assert result_expr._find_backend(use_default=False) is backend.connection
        result = backend.connection.execute(result_expr)
        assert not result.empty

        expected = expected_expr.cast(result_expr.schema()).execute()
        assert not expected.empty

        assert list(map(str.lower, expected.columns)) == result.columns.tolist()
        expected.columns = result.columns

        assert len(expected) == len(result)
        backend.assert_frame_equal(result, expected, check_dtype=False)

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
