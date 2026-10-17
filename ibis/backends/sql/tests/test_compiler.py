from __future__ import annotations

import pytest
import sqlglot as sg

import ibis
from ibis import _
from ibis.backends import _get_backend_names
from ibis.backends.sql.dialects import Trino


def test_window_with_row_number_compiles():
    # GH #8058: the add_order_by_to_empty_ranking_window_functions rule was
    # matching on `RankBase` subclasses with a pattern expecting an `arg`
    # attribute, which is not present on `RowNumber`
    expr = (
        ibis.memtable({"a": range(30)})
        .mutate(id=ibis.row_number())
        .sample(fraction=0.25, seed=0)
        .mutate(is_test=_.id.isin(_.id))
        .filter(~_.is_test)
    )
    assert ibis.to_sql(expr)


def test_transpile_join():
    (result,) = sg.transpile(
        "SELECT * FROM t1 JOIN t2 ON x = y", read="duckdb", write=Trino
    )
    assert "CROSS JOIN" not in result


@pytest.mark.parametrize("dialect", sorted(_get_backend_names(exclude=("polars",))))
def test_to_sql_knows_every_sql_backend(dialect):
    # `ibis.to_sql` looks the dialect up on `ibis.backends.sql.compilers`, so
    # every SQL backend's compiler module has to be imported there
    expr = ibis.table({"a": "int64"}, name="t")
    assert ibis.to_sql(expr, dialect=dialect)
