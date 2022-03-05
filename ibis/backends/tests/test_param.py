import pytest

import ibis
import ibis.expr.datatypes as dt


@pytest.mark.parametrize(
    ('column', 'raw_value'),
    [
        ('double_col', 0.0),
        ('double_col', 10.1),
        ('float_col', 1.1),
        ('float_col', 2.2),
    ],
)
@pytest.mark.notimpl(["datafusion"])
def test_floating_scalar_parameter(backend, alltypes, df, column, raw_value):
    value = ibis.param(dt.double)
    expr = alltypes[column] + value
    expected = df[column] + raw_value
    result = expr.execute(params={value: raw_value})
    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize(
    ('start_string', 'end_string'),
    [('2009-03-01', '2010-07-03'), ('2014-12-01', '2017-01-05')],
)
@pytest.mark.notimpl(["datafusion", "pyspark"])
def test_date_scalar_parameter(backend, alltypes, start_string, end_string):
    start, end = ibis.param(dt.date), ibis.param(dt.date)

    col = alltypes.timestamp_col.date()
    expr = col.between(start, end)
    expected_expr = col.between(start_string, end_string)

    result = expr.execute(params={start: start_string, end: end_string})
    expected = expected_expr.execute()

    backend.assert_series_equal(result, expected)


@pytest.mark.notimpl(["datafusion", "pyspark"])
def test_timestamp_accepts_date_literals(backend, alltypes):
    date_string = '2009-03-01'
    param = ibis.param(dt.timestamp)
    expr = alltypes.mutate(param=param)
    params = {param: date_string}
    assert expr.compile(params=params) is not None


@pytest.mark.notimpl(
    [
        "dask",
        "datafusion",
        "impala",
        "pandas",
        "pyspark",
    ]
)
@pytest.mark.never(
    ["mysql", "sqlite"],
    reason="mysql and sqlite will never implement array types",
)
def test_scalar_param_array(backend, con):
    value = [1, 2, 3]
    param = ibis.param(dt.Array(dt.int64))
    result = con.execute(param.length(), params={param: value})
    assert result == len(value)


@pytest.mark.notimpl(
    [
        "clickhouse",
        "datafusion",
        "impala",
        "postgres",
        "pyspark",
    ]
)
@pytest.mark.never(
    ["mysql", "sqlite"],
    reason="mysql and sqlite will never implement struct types",
)
def test_scalar_param_struct(backend, con):
    value = dict(a=1, b="abc", c=3.0)
    param = ibis.param("struct<a: int64, b: string, c: float64>")
    result = con.execute(param["a"], params={param: value})
    assert result == value["a"]


@pytest.mark.notimpl(
    [
        "clickhouse",
        "datafusion",
        # TODO: duckdb maps are tricky because they are multimaps
        "duckdb",
        "impala",
        "pyspark",
    ]
)
@pytest.mark.never(
    ["mysql", "sqlite"],
    reason="mysql and sqlite will never implement map types",
)
@pytest.mark.notyet(["postgres"])
def test_scalar_param_map(backend, con):
    value = {'a': 'ghi', 'b': 'def', 'c': 'abc'}
    param = ibis.param(dt.Map(dt.string, dt.string))
    result = con.execute(param['b'], params={param: value})
    assert result == value['b']
