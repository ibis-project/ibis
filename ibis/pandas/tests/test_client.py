import pytest

pytest.importorskip('multipledispatch')

import pandas as pd  # noqa: E402
import pandas.util.testing as tm  # noqa: E402

from ibis.pandas.api import connect  # noqa: E402

pytestmark = pytest.mark.pandas


@pytest.fixture
def df():
    return pd.DataFrame({
        'a': [1, 2, 3],
        'b': list('abc'),
        'c': [4.0, 5.0, 6.0],
        'd': pd.date_range('now', periods=3).values
    })


@pytest.fixture
def dictionary(df):
    return dict(df=df)


def test_table(dictionary, df):
    con = connect(dictionary)
    data = con.table('df')
    result = data.execute()
    tm.assert_frame_equal(df, result)
