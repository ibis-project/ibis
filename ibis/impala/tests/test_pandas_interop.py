import pytest

import numpy as np

from pandas.util.testing import assert_frame_equal
import pandas as pd

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch

pytestmark = pytest.mark.impala

pytest.importorskip('hdfs')
pytest.importorskip('sqlalchemy')
pytest.importorskip('impala.dbapi')

from ibis.impala.pandas_interop import DataFrameWriter  # noqa: E402


@pytest.fixture
def exhaustive_df():
    return pd.DataFrame(
        {
            'bigint_col': np.array(
                [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype='i8'
            ),
            'bool_col': np.array(
                [
                    True,
                    False,
                    True,
                    False,
                    True,
                    None,
                    True,
                    False,
                    True,
                    False,
                ],
                dtype=np.bool_,
            ),
            'date_string_col': [
                '11/01/10',
                None,
                '11/01/10',
                '11/01/10',
                '11/01/10',
                '11/01/10',
                '11/01/10',
                '11/01/10',
                '11/01/10',
                '11/01/10',
            ],
            'double_col': np.array(
                [
                    0.0,
                    10.1,
                    np.nan,
                    30.299_999_999_999_997,
                    40.399_999_999_999_999,
                    50.5,
                    60.599_999_999_999_994,
                    70.700_000_000_000_003,
                    80.799_999_999_999_997,
                    90.899_999_999_999_991,
                ],
                dtype=np.float64,
            ),
            'float_col': np.array(
                [
                    np.nan,
                    1.100_000_023_841_857_9,
                    2.200_000_047_683_715_8,
                    3.299_999_952_316_284_2,
                    4.400_000_095_367_431_6,
                    5.5,
                    6.599_999_904_632_568_4,
                    7.699_999_809_265_136_7,
                    8.800_000_190_734_863_3,
                    9.899_999_618_530_273_4,
                ],
                dtype='f4',
            ),
            'int_col': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='i4'),
            'month': [11, 11, 11, 11, 2, 11, 11, 11, 11, 11],
            'smallint_col': np.array(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='i2'
            ),
            'string_col': [
                '0',
                '1',
                None,
                'double , whammy',
                '4',
                '5',
                '6',
                '7',
                '8',
                '9',
            ],
            'timestamp_col': [
                pd.Timestamp('2010-11-01 00:00:00'),
                None,
                pd.Timestamp('2010-11-01 00:02:00.100000'),
                pd.Timestamp('2010-11-01 00:03:00.300000'),
                pd.Timestamp('2010-11-01 00:04:00.600000'),
                pd.Timestamp('2010-11-01 00:05:00.100000'),
                pd.Timestamp('2010-11-01 00:06:00.150000'),
                pd.Timestamp('2010-11-01 00:07:00.210000'),
                pd.Timestamp('2010-11-01 00:08:00.280000'),
                pd.Timestamp('2010-11-01 00:09:00.360000'),
            ],
            'tinyint_col': np.array(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='i1'
            ),
            'year': [
                2010,
                2010,
                2010,
                2010,
                2010,
                2009,
                2009,
                2009,
                2009,
                2009,
            ],
        }
    )


def test_alltypes_roundtrip(con, alltypes_df):
    _check_roundtrip(con, alltypes_df)


def test_writer_cleanup_deletes_hdfs_dir(con, hdfs, alltypes_df):
    writer = DataFrameWriter(con, alltypes_df)

    path = writer.write_temp_csv()
    assert hdfs.exists(path)

    writer.cleanup()
    assert not hdfs.exists(path)

    # noop
    writer.cleanup()
    assert not hdfs.exists(path)


def test_create_table_from_dataframe(con, alltypes_df, temp_table_db):
    tmp_db, tname = temp_table_db
    con.create_table(tname, alltypes_df, database=tmp_db)

    table = con.table(tname, database=tmp_db)
    df = table.execute()
    assert_frame_equal(df, alltypes_df)


def test_insert(con, temp_table_db, exhaustive_df):
    tmp_db, table_name = temp_table_db
    schema = sch.infer(exhaustive_df)

    con.create_table(table_name, database=tmp_db, schema=schema)

    con.insert(table_name, exhaustive_df.iloc[:4], database=tmp_db)
    con.insert(table_name, exhaustive_df.iloc[4:], database=tmp_db)

    table = con.table(table_name, database=tmp_db)

    result = (
        table.execute().sort_values(by='tinyint_col').reset_index(drop=True)
    )
    assert_frame_equal(result, exhaustive_df)


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_insert_partition():
    assert False


def test_round_trip_exhaustive(con, exhaustive_df):
    _check_roundtrip(con, exhaustive_df)


def _check_roundtrip(con, df):
    writer = DataFrameWriter(con, df)
    path = writer.write_temp_csv()

    table = writer.delimited_table(path)
    df2 = table.execute()
    assert_frame_equal(df2, df)


def test_timestamp_with_timezone():
    df = pd.DataFrame(
        {'A': pd.date_range('20130101', periods=3, tz='US/Eastern')}
    )
    schema = sch.infer(df)
    expected = ibis.schema([('A', "timestamp('US/Eastern')")])
    assert schema.equals(expected)
    assert schema.types[0].equals(dt.Timestamp('US/Eastern'))
