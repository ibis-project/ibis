import pandas.testing as tm
import pyarrow.compute as pc

import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis.udf.vectorized import elementwise, reduction


@elementwise(input_type=['string'], output_type='int64')
def my_string_length(arr, **kwargs):
    # arr is a pyarrow.StringArray
    return pc.cast(pc.multiply(pc.utf8_length(arr), 2), target_type='int64')


@elementwise(input_type=[dt.int64, dt.int64], output_type=dt.int64)
def my_add(arr1, arr2, **kwargs):
    return pc.add(arr1, arr2)


@reduction(input_type=[dt.float64], output_type=dt.float64)
def my_mean(arr):
    return pc.mean(arr)


def test_udf(alltypes):
    data_string_col = alltypes.date_string_col.execute()
    expected = data_string_col.str.len() * 2

    expr = my_string_length(alltypes.date_string_col)
    assert isinstance(expr, ir.ColumnExpr)

    result = expr.execute()
    tm.assert_series_equal(result, expected, check_names=False)


def test_multiple_argument_udf(alltypes):
    expr = my_add(alltypes.smallint_col, alltypes.int_col)
    result = expr.execute()

    df = alltypes[['smallint_col', 'int_col']].execute()
    expected = (df.smallint_col + df.int_col).astype('int64')

    tm.assert_series_equal(result, expected.rename('tmp'))
