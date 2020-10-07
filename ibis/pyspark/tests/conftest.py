from datetime import datetime, timezone

import numpy as np
import pytest

import ibis


@pytest.fixture(scope='session')
def client():
    import pyspark.sql.functions as F
    from pyspark.sql import SparkSession

    session = SparkSession.builder.getOrCreate()
    client = ibis.pyspark.connect(session)

    df = client._session.range(0, 10)
    df = df.withColumn("str_col", F.lit('value'))
    df.createTempView('basic_table')

    df_nans = client._session.createDataFrame(
        [
            [np.NaN, 'Alfred', None],
            [27.0, 'Batman', 'motocycle'],
            [3.0, None, 'joker'],
        ],
        ['age', 'user', 'toy'],
    )
    df_nans.createTempView('nan_table')

    df_dates = client._session.createDataFrame(
        [['2018-01-02'], ['2018-01-03'], ['2018-01-04']], ['date_str']
    )
    df_dates.createTempView('date_table')

    df_arrays = client._session.createDataFrame(
        [
            ['k1', [1, 2, 3], ['a']],
            ['k2', [4, 5], ['test1', 'test2', 'test3']],
            ['k3', [6], ['w', 'x', 'y', 'z']],
            ['k1', [], ['cat', 'dog']],
            ['k1', [7, 8], []],
        ],
        ['key', 'array_int', 'array_str'],
    )
    df_arrays.createTempView('array_table')

    df_time_indexed = client._session.createDataFrame(
        [
            [datetime(2017, 1, 2, 5, tzinfo=timezone.utc), 1, 1.0],
            [datetime(2017, 1, 2, 5, tzinfo=timezone.utc), 2, 2.0],
            [datetime(2017, 1, 2, 6, tzinfo=timezone.utc), 1, 3.0],
            [datetime(2017, 1, 2, 6, tzinfo=timezone.utc), 2, 4.0],
            [datetime(2017, 1, 2, 7, tzinfo=timezone.utc), 1, 5.0],
            [datetime(2017, 1, 2, 7, tzinfo=timezone.utc), 2, 6.0],
            [datetime(2017, 1, 4, 8, tzinfo=timezone.utc), 1, 7.0],
            [datetime(2017, 1, 4, 8, tzinfo=timezone.utc), 2, 8.0],
        ],
        ['time', 'key', 'value'],
    )

    df_time_indexed.createTempView('time_indexed_table')

    return client


class IbisWindow:
    # Test util class to generate different types of ibis windows
    def __init__(self, windows):
        self.windows = windows

    def get_windows(self):
        # Return a list of Ibis windows
        return [
            ibis.window(
                preceding=w[0],
                following=w[1],
                order_by='time',
                group_by='key',
            )
            for w in self.windows
        ]


@pytest.fixture
def ibis_windows(request):
    return IbisWindow(request.param).get_windows()
