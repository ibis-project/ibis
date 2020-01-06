import pytest

import ibis.expr.datatypes as dt
from ibis.tests.backends import Pandas, PySpark
from ibis.udf.vectorized import elementwise


@elementwise(input_type=[dt.double], output_type=dt.double)
def add_one(s):
    return s + 1


@pytest.mark.only_on_backends([Pandas, PySpark])
@pytest.mark.xfail_unsupported
def test_elementwise_udf(backend, alltypes, df):
    result = add_one(alltypes['double_col']).execute()
    expected = add_one.func(df['double_col'])
    backend.assert_series_equal(result, expected, check_names=False)
