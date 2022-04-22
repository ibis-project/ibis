import numpy as np
import pytest
import toolz

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir

pytestmark = pytest.mark.never(["sqlite", "mysql"], reason="No array support")


@pytest.mark.notimpl(["impala", "postgres", "clickhouse", "datafusion"])
def test_array_column(backend, alltypes, df):
    expr = ibis.array([alltypes['double_col'], alltypes['double_col']])
    assert isinstance(expr, ir.ArrayColumn)

    result = expr.execute()
    expected = df.apply(
        lambda row: np.array(
            [row['double_col'], row['double_col']], dtype=object
        ),
        axis=1,
    )
    backend.assert_series_equal(result, expected, check_names=False)


@pytest.mark.notimpl(["impala"])
def test_array_scalar(con):
    expr = ibis.array([1.0, 2.0, 3.0])
    assert isinstance(expr, ir.ArrayScalar)

    result = con.execute(expr)
    expected = np.array([1.0, 2.0, 3.0])

    # This does not check whether `result` is an np.array or a list,
    # because it varies across backends and backend configurations
    assert np.array_equal(result, expected)


# Issues #2370
@pytest.mark.notimpl(["impala", "datafusion"])
def test_array_concat(con):
    left = ibis.literal([1, 2, 3])
    right = ibis.literal([2, 1])
    expr = left + right
    result = con.execute(expr)
    expected = np.array([1, 2, 3, 2, 1])

    # This does not check whether `result` is an np.array or a list,
    # because it varies across backends and backend configurations
    assert np.array_equal(result, expected)


@pytest.mark.notimpl(["impala", "datafusion"])
def test_array_length(con):
    expr = ibis.literal([1, 2, 3]).length()
    assert con.execute(expr) == 3


@pytest.mark.notimpl(["impala"])
def test_list_literal(con):
    arr = [1, 2, 3]
    expr = ibis.literal(arr)
    result = con.execute(expr)

    # This does not check whether `result` is an np.array or a list,
    # because it varies across backends and backend configurations
    assert np.array_equal(result, arr)


@pytest.mark.notimpl(["impala"])
def test_np_array_literal(con):
    arr = np.array([1, 2, 3])
    expr = ibis.literal(arr)
    result = con.execute(expr)

    # This does not check whether `result` is an np.array or a list,
    # because it varies across backends and backend configurations
    assert np.array_equal(result, arr)


builtin_array = toolz.compose(
    # the type parser needs additional work for this to work
    pytest.mark.broken(
        "clickhouse",
        reason="nullable types can't yet be parsed",
    ),
    # these will almost certainly never be supported
    pytest.mark.never(
        ["mysql", "sqlite"],
        reason="array types are unsupported",
    ),
    # someone just needs to implement these
    pytest.mark.notimpl(["datafusion", "dask"]),
    # unclear if thi will ever be supported
    pytest.mark.notyet(
        ["impala"],
        reason="impala doesn't support array types",
    ),
)


@builtin_array
def test_array_discovery(con):
    t = con.table("array_types")
    expected = ibis.schema(
        dict(
            x=dt.Array(dt.int64),
            y=dt.Array(dt.string),
            z=dt.Array(dt.float64),
            grouper=dt.string,
            scalar_column=dt.float64,
        )
    )
    assert t.schema() == expected
