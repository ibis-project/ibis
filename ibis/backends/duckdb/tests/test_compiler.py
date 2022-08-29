import pytest

import ibis
from ibis.backends.duckdb.compiler import DuckDBSQLCompiler

compiler = DuckDBSQLCompiler()

QUERY = """SELECT array_length(CAST(list_value{} AS ARRAY)) AS n"""


@pytest.mark.parametrize(
    "array",
    [
        range(10),
        range(5),
        [-1, 0, 1],
    ],
)
def test_compiler_array_length(array):
    t = ibis.array(array).length().name("n")
    result = str(
        compiler.to_sql(t).compile(compile_kwargs={"literal_binds": True})
    )
    assert result == QUERY.format(tuple(array))
