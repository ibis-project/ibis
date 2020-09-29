import pandas as pd
import pandas.util.testing as tm
import pytest

pytest.importorskip('pyspark')
pytestmark = pytest.mark.pyspark


def test_table_with_timecontext(client):
    table = client.table('time_indexed_table')
    context = (pd.Timestamp('20170102'), pd.Timestamp('20170103'))
    result = table.compile(timecontext=context).toPandas()
    expected = table.compile().toPandas()
    expected = expected[expected.time.between(*context)]
    tm.assert_frame_equal(result, expected)
