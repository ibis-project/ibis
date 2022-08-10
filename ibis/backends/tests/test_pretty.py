import io

import pytest
from pytest import mark, param

import ibis
import ibis.common.exceptions as exc
from ibis import _

sa = pytest.importorskip("sqlalchemy")
pytest.importorskip("sqlglot")


@mark.never(
    ["dask", "pandas"],
    reason="Dask and Pandas are not SQL backends",
    raises=(NotImplementedError, AssertionError),
)
@mark.notimpl(
    ["datafusion", "pyspark"],
    reason="Not clear how to extract SQL from the backend",
    raises=(exc.OperationNotDefinedError, NotImplementedError, AssertionError),
)
def test_table(con):
    expr = con.tables.functional_alltypes.select(c=_.int_col + 1)
    buf = io.StringIO()
    ibis.show_sql(expr, file=buf)
    assert buf.getvalue()


simple_literal = param(
    ibis.literal(1),
    id="simple_literal",
)
array_literal = param(
    ibis.array([1]),
    marks=[
        mark.never(
            ["mysql", "sqlite"],
            raises=sa.exc.CompileError,
            reason="arrays not supported in the backend",
        ),
        mark.notyet(
            ["impala"],
            raises=NotImplementedError,
            reason="Impala hasn't implemented array literals",
        ),
        mark.notimpl(
            ["postgres"],
            reason="array literals are not yet implemented",
            raises=NotImplementedError,
        ),
    ],
    id="array_literal",
)
no_structs = mark.never(
    ["impala", "mysql", "sqlite"],
    raises=(NotImplementedError, sa.exc.CompileError),
    reason="structs not supported in the backend",
)
no_struct_literals = mark.notimpl(
    ["postgres"],
    reason="struct literals are not yet implemented",
)
not_sql = mark.never(
    ["pandas", "dask"],
    raises=(exc.IbisError, NotImplementedError, AssertionError),
    reason="Not a SQL backend",
)
no_sql_extraction = mark.notimpl(
    ["datafusion", "pyspark"],
    reason="Not clear how to extract SQL from the backend",
)


@mark.parametrize(
    "expr",
    [
        simple_literal,
        array_literal,
        param(
            ibis.struct(dict(a=1)),
            marks=[no_structs, no_struct_literals],
            id="struct_literal",
        ),
    ],
)
@not_sql
@no_sql_extraction
def test_literal(backend, expr):
    buf = io.StringIO()
    ibis.show_sql(expr, dialect=backend.name(), file=buf)
    assert buf.getvalue()
