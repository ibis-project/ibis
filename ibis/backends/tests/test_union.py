import pandas as pd
import pytest

from ibis.backends.bigquery.tests.conftest import BigQueryTest
from ibis.backends.impala.tests.conftest import ImpalaTest
from ibis.backends.pandas.tests.conftest import PandasTest
from ibis.backends.postgres.tests.conftest import PostgresTest
from ibis.backends.pyspark.tests.conftest import PySparkTest


@pytest.mark.parametrize('distinct', [False, True])
@pytest.mark.only_on_backends([BigQueryTest, ImpalaTest, PandasTest,
                               PostgresTest, PySparkTest])
@pytest.mark.xfail_unsupported
def test_union(backend, alltypes, df, distinct):
    result = alltypes.union(alltypes, distinct=distinct).execute()
    expected = df if distinct else pd.concat([df, df], axis=0)

    # Result is not in original order on PySpark backend when distinct=True
    result = result.sort_values(['id'])
    expected = expected.sort_values(['id'])

    backend.assert_frame_equal(result, expected)
