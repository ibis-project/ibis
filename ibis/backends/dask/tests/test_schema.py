import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.dataframe.utils import tm

import ibis
from ibis.expr import datatypes as dt
from ibis.expr import schema as sch


def test_infer_exhaustive_dataframe(npartitions):
    df = dd.from_pandas(
        pd.DataFrame(
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
                'bool_obj_col': np.array(
                    [
                        True,
                        False,
                        np.nan,
                        False,
                        True,
                        np.nan,
                        True,
                        np.nan,
                        True,
                        False,
                    ],
                    dtype=np.object_,
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
                        30.299999999999997,
                        40.399999999999999,
                        50.5,
                        60.599999999999994,
                        70.700000000000003,
                        80.799999999999997,
                        90.899999999999991,
                    ],
                    dtype=np.float64,
                ),
                'float_col': np.array(
                    [
                        np.nan,
                        1.1000000238418579,
                        2.2000000476837158,
                        3.2999999523162842,
                        4.4000000953674316,
                        5.5,
                        6.5999999046325684,
                        7.6999998092651367,
                        8.8000001907348633,
                        9.8999996185302734,
                    ],
                    dtype='f4',
                ),
                'int_col': np.array(
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='i4'
                ),
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
        ),
        npartitions=npartitions,
    )

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
        ('year', dt.int64),
    ]

    assert sch.infer(df) == ibis.schema(expected)


def test_apply_to_schema_with_timezone(npartitions):
    data = {'time': pd.date_range('2018-01-01', '2018-01-02', freq='H')}
    df = dd.from_pandas(pd.DataFrame(data), npartitions=npartitions)
    expected = df.assign(time=df.time.astype('datetime64[ns, EST]'))
    desired_schema = ibis.schema([('time', 'timestamp("EST")')])
    result = desired_schema.apply_to(df.copy())
    tm.assert_frame_equal(result.compute(), expected.compute())
