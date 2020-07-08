import pandas as pd
import pytest

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir

pymapd = pytest.importorskip('pymapd')
rbc = pytest.importorskip('rbc')

pytestmark = pytest.mark.omniscidb


@pytest.fixture
def add_number(con):
    def my_add_number(left, right):
        return left + right

    for dtype in [
        dt.float32,
        dt.float64,
        dt.int8,
        dt.int16,
        dt.int32,
        dt.int64,
    ]:
        con.udf.elementwise(
            input_type=[dtype, dtype], output_type=dtype, infer_literal=True
        )(my_add_number)
    return my_add_number


@pytest.mark.parametrize(
    'left_arg, right_arg, expected_fn',
    [
        pytest.param(
            lambda t: t.float_col,
            lambda t: t.float_col,
            lambda df: df.float_col + df.float_col,
            id='elementwise_float_col_plus_float_col',
        ),
        pytest.param(
            lambda t: t.double_col,
            lambda t: t.double_col,
            lambda df: df.double_col + df.double_col,
            id='elementwise_double_col_plus_double_col',
        ),
        pytest.param(
            lambda t: t.int_col,
            lambda t: t.int_col,
            lambda df: df.int_col + df.int_col,
            id='elementwise_int_col_plus_int_col',
        ),
        pytest.param(
            lambda t: t.bigint_col,
            lambda t: t.bigint_col,
            lambda df: df.bigint_col + df.bigint_col,
            id='elementwise_bigint_col_plus_bigint_col',
        ),
        pytest.param(
            lambda t: t.int_col,
            lambda t: 1,
            lambda df: df.int_col + 1,
            id='elementwise_int_col_plus_literal_int',
        ),
        pytest.param(
            lambda t: t.float_col,
            lambda t: 1.0,
            lambda df: df.float_col + 1.0,
            id='elementwise_float_col_plus_literal_float',
        ),
        pytest.param(
            lambda t: t.bigint_col,
            lambda t: ibis.literal(1, type='int64'),
            lambda df: df.bigint_col + 1,
            id='elementwise_bigint_col_plus_literal_bigint',
        ),
    ],
)
def test_elementwise_udf_number(
    con, alltypes, add_number, left_arg, right_arg, expected_fn
):
    expr = add_number(left_arg(alltypes), right_arg(alltypes))

    assert isinstance(expr, ir.ColumnExpr)
    assert isinstance(expr, ir.NumericColumn)

    result = expr.execute()

    df = alltypes.execute()
    expected = expected_fn(df)
    pd.testing.assert_series_equal(
        result, expected, check_dtype=False, check_names=False
    )
