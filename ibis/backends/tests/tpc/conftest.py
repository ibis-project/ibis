from __future__ import annotations

import datetime
import functools
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytest
import sqlglot as sg
from dateutil.relativedelta import relativedelta

import ibis
from ibis.formats.pandas import PandasData

if TYPE_CHECKING:
    from collections.abc import Callable

    import ibis.expr.types as ir


def pytest_pyfunc_call(pyfuncitem):
    """Inject `backend` and fixtures to all TPC-DS test functions.

    Defining this hook here limits its scope to the TPC-DS tests.
    """
    testfunction = pyfuncitem.obj
    funcargs = pyfuncitem.funcargs
    testargs = {arg: funcargs[arg] for arg in pyfuncitem._fixtureinfo.argnames}
    result = testfunction(**testargs, backend=funcargs["backend"])
    assert (
        result is None
    ), "test function should not return anything, did you mean to use assert?"
    return True


def tpc_test(suite_name: Literal["h", "ds"], *, result_is_empty=False):
    """Decorator for TPC tests.

    Parameters
    ----------
    suite_name
        The name of the TPC suite. Only `'h'` and ~'ds'~ are supported right now.
    result_is_empty
        If the expected result is an empty table.

    Automates the process of loading the SQL query from the file system and
    asserting that the result of the ibis expression is equal to the expected
    result of executing the raw SQL.
    """

    def inner(test: Callable[..., ir.Table]):
        name = f"tpc{suite_name}"

        @getattr(pytest.mark, name)
        @pytest.mark.usefixtures("backend")
        @pytest.mark.xdist_group(name)
        @functools.wraps(test)
        def wrapper(*args, backend, **kwargs):
            backend_name = backend.name()
            if not getattr(backend, f"supports_{name}"):
                pytest.skip(
                    f"{backend_name} backend doesn't support testing {name} queries yet"
                )
            query_name_match = re.match(r"^test_(\d\d)$", test.__name__)
            assert query_name_match is not None

            query_number = query_name_match.group(1)
            sql_path_name = f"{query_number}.sql"

            base = Path(__file__).parent / "queries"

            path = base / backend_name / suite_name / sql_path_name

            if path.exists():
                dialect = backend_name
            else:
                dialect = "duckdb"
                path = base / "duckdb" / suite_name / sql_path_name

            raw_sql = path.read_text()

            sql = sg.parse_one(raw_sql, read=dialect)

            sql = backend._transform_tpc_sql(
                sql, suite=suite_name, leaves=backend.list_tpc_tables(suite_name)
            )

            raw_sql = sql.sql(dialect=dialect, pretty=True)

            expected_expr = backend.connection.sql(raw_sql, dialect=dialect)

            result_expr = test(*args, **kwargs)

            assert result_expr._find_backend(use_default=False) is backend.connection
            result = backend.connection.to_pandas(result_expr)
            assert (result_is_empty and result.empty) or not result.empty

            expected = expected_expr.to_pandas()

            assert len(expected.columns) == len(result.columns)
            assert all(r in e.lower() for r, e in zip(result.columns, expected.columns))

            expected.columns = result.columns

            expected = PandasData.convert_table(expected, result_expr.schema())
            assert (result_is_empty and expected.empty) or not expected.empty

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

        return wrapper

    return inner


def add_date(datestr: str, dy: int = 0, dm: int = 0, dd: int = 0) -> ir.DateScalar:
    dt = datetime.date.fromisoformat(datestr)
    dt += relativedelta(years=dy, months=dm, days=dd)
    return ibis.date(dt.isoformat())
