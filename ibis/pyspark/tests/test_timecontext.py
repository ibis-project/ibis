import pandas as pd
import pandas.util.testing as tm
import pytest

from ibis.pyspark.timecontext import combine_time_context

pytest.importorskip('pyspark')
pytestmark = pytest.mark.pyspark


def test_table_with_timecontext(client):
    table = client.table('time_indexed_table')
    context = (pd.Timestamp('20170102'), pd.Timestamp('20170103'))
    result = table.execute(timecontext=context)
    expected = table.execute()
    expected = expected[expected.time.between(*context)]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ('contexts', 'expected'),
    [
        (
            [
                (pd.Timestamp('20200102'), pd.Timestamp('20200103')),
                (pd.Timestamp('20200101'), pd.Timestamp('20200106')),
            ],
            (pd.Timestamp('20200101'), pd.Timestamp('20200106')),
        ),  # superset
        (
            [
                (pd.Timestamp('20200101'), pd.Timestamp('20200103')),
                (pd.Timestamp('20200102'), pd.Timestamp('20200106')),
            ],
            (pd.Timestamp('20200101'), pd.Timestamp('20200106')),
        ),  # overlap
        (
            [
                (pd.Timestamp('20200101'), pd.Timestamp('20200103')),
                (pd.Timestamp('20200202'), pd.Timestamp('20200206')),
            ],
            (pd.Timestamp('20200101'), pd.Timestamp('20200206')),
        ),  # non-overlap
        (
            [(pd.Timestamp('20200101'), pd.Timestamp('20200103')), None],
            (pd.Timestamp('20200101'), pd.Timestamp('20200103')),
        ),  # None in input
        ([None], None),  # None for all
        (
            [
                (pd.Timestamp('20200102'), pd.Timestamp('20200103')),
                (pd.Timestamp('20200101'), pd.Timestamp('20200106')),
                (pd.Timestamp('20200109'), pd.Timestamp('20200110')),
            ],
            (pd.Timestamp('20200101'), pd.Timestamp('20200110')),
        ),  # complex
    ],
)
def test_combine_time_context(contexts, expected):
    assert combine_time_context(contexts) == expected
