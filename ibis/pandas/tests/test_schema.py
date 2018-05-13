import pytest
import numpy as np
import pandas as pd

import pandas.util.testing as tm

import ibis
from ibis.expr import datatypes as dt
from ibis.expr import schema as sch


@pytest.mark.parametrize(('column', 'expected_dtype'), [
    ([True, False, False], dt.boolean),
    (np.int8([-3, 9, 17]), dt.int8),
    (np.uint8([3, 0, 16]), dt.uint8),
    (np.int16([-5, 0, 12]), dt.int16),
    (np.uint16([5569, 1, 33]), dt.uint16),
    (np.int32([-12, 3, 25000]), dt.int32),
    (np.uint32([100, 0, 6]), dt.uint32),
    (np.uint64([666, 2, 3]), dt.uint64),
    (np.int64([102, 67228734, -0]), dt.int64),
    (np.float32([45e-3, -0.4, 99.]), dt.float),
    (np.float64([-3e43, 43., 10000000.]), dt.double),
    (['foo', 'bar', 'hello'], dt.string),
    ([pd.Timestamp('2010-11-01 00:01:00'),
      pd.Timestamp('2010-11-01 00:02:00.1000'),
      pd.Timestamp('2010-11-01 00:03:00.300000')], dt.timestamp),
    (pd.date_range('20130101', periods=3, tz='US/Eastern'),
     dt.timestamp('US/Eastern')),
    ([pd.Timedelta('1 days'),
      pd.Timedelta('-1 days 2 min 3us'),
      pd.Timedelta('-2 days +23:57:59.999997')], dt.Interval('ns')),
    (pd.Series(['a', 'b', 'c', 'a']).astype('category'), dt.Category())
])
def test_infer_simple_dataframe(column, expected_dtype):
    df = pd.DataFrame({'col': column})
    assert sch.infer(df) == ibis.schema([('col', expected_dtype)])


def test_infer_exhaustive_dataframe():
    df = pd.DataFrame({
        'bigint_col': np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                               dtype='i8'),
        'bool_col': np.array([True, False, True, False, True, None,
                              True, False, True, False], dtype=np.bool_),
        'bool_obj_col': np.array([True, False, np.nan, False, True, np.nan,
                                  True, np.nan, True, False],
                                 dtype=np.object_),
        'date_string_col': ['11/01/10', None, '11/01/10', '11/01/10',
                            '11/01/10', '11/01/10', '11/01/10', '11/01/10',
                            '11/01/10', '11/01/10'],
        'double_col': np.array([0.0, 10.1, np.nan, 30.299999999999997,
                                40.399999999999999, 50.5, 60.599999999999994,
                                70.700000000000003, 80.799999999999997,
                                90.899999999999991], dtype=np.float64),
        'float_col': np.array([np.nan, 1.1000000238418579, 2.2000000476837158,
                               3.2999999523162842, 4.4000000953674316, 5.5,
                               6.5999999046325684, 7.6999998092651367,
                               8.8000001907348633,
                               9.8999996185302734], dtype='f4'),
        'int_col': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='i4'),
        'month': [11, 11, 11, 11, 2, 11, 11, 11, 11, 11],
        'smallint_col': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='i2'),
        'string_col': ['0', '1', None, 'double , whammy', '4', '5',
                       '6', '7', '8', '9'],
        'timestamp_col': [pd.Timestamp('2010-11-01 00:00:00'),
                          None,
                          pd.Timestamp('2010-11-01 00:02:00.100000'),
                          pd.Timestamp('2010-11-01 00:03:00.300000'),
                          pd.Timestamp('2010-11-01 00:04:00.600000'),
                          pd.Timestamp('2010-11-01 00:05:00.100000'),
                          pd.Timestamp('2010-11-01 00:06:00.150000'),
                          pd.Timestamp('2010-11-01 00:07:00.210000'),
                          pd.Timestamp('2010-11-01 00:08:00.280000'),
                          pd.Timestamp('2010-11-01 00:09:00.360000')],
        'tinyint_col': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='i1'),
        'year': [2010, 2010, 2010, 2010, 2010, 2009, 2009, 2009, 2009, 2009]
    })

    expected = [
        ('bigint_col', dt.int64),
        ('bool_col', dt.boolean),
        ('bool_obj_col', dt.boolean),
        ('date_string_col', dt.string),
        ('double_col', dt.double),
        ('float_col', dt.float),
        ('int_col', dt.int32),
        ('month', dt.int64),
        ('smallint_col', dt.int16),
        ('string_col', dt.string),
        ('timestamp_col', dt.timestamp),
        ('tinyint_col', dt.int8),
        ('year', dt.int64)
    ]

    assert sch.infer(df) == ibis.schema(expected)


def test_apply_to_schema_with_timezone():
    data = {
        'time': pd.date_range('2018-01-01', '2018-01-02', freq='H')
    }
    df = pd.DataFrame(data)
    expected = df.assign(time=df.time.astype('datetime64[ns, EST]'))
    desired_schema = ibis.schema([('time', 'timestamp("EST")')])
    result = desired_schema.apply_to(df.copy())
    tm.assert_frame_equal(expected, result)


# TODO(kszucs): test_Schema_to_pandas
