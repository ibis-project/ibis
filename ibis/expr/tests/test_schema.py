
import pytest
import numpy as np
import pandas as pd

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
    ([pd.Timedelta('1 days'),
      pd.Timedelta('-1 days 2 min 3us'),
      pd.Timedelta('-2 days +23:57:59.999997')], dt.Interval('ns')),
    # (pd.Categorical(['a', 'b', 'c', 'a']), 'category')
])
def test_pandas_dtypes(column, expected_dtype):
    df = pd.DataFrame({'col': column})
    assert sch.infer(df) == ibis.schema([('col', expected_dtype)])


def test_whole_schema():
    customers = ibis.table(
        [
            ('cid', 'int64'),
            ('mktsegment', 'string'),
            ('address', ('struct<city: string, street: string, '
                         'street_number: int32, zip: int16>')),
            ('phone_numbers', 'array<string>'),
            (
                'orders', """array<struct<
                                oid: int64,
                                status: string,
                                totalprice: decimal(12, 2),
                                order_date: string,
                                items: array<struct<
                                    iid: int64,
                                    name: string,
                                    price: decimal(12, 2),
                                    discount_perc: decimal(12, 2),
                                    shipdate: string
                                >>
                            >>"""
            ),
            ('web_visits', ('map<string, struct<user_agent: string, '
                            'client_ip: string, visit_date: string, '
                            'duration_ms: int32>>')),
            ('support_calls', ('array<struct<agent_id: int64, '
                               'call_date: string, duration_ms: int64, '
                               'issue_resolved: boolean, '
                               'agent_comment: string>>'))
        ],
        name='customers',
    )
    expected = ibis.Schema.from_tuples(
        [
            ('cid', dt.int64),
            ('mktsegment', dt.string),
            (
                'address',
                dt.Struct.from_tuples([
                    ('city', dt.string),
                    ('street', dt.string),
                    ('street_number', dt.int32),
                    ('zip', dt.int16)
                ]),
            ),
            ('phone_numbers', dt.Array(dt.string)),
            (
                'orders', dt.Array(dt.Struct.from_tuples([
                            ('oid', dt.int64),
                            ('status', dt.string),
                            ('totalprice', dt.Decimal(12, 2)),
                            ('order_date', dt.string),
                            (
                                'items',
                                dt.Array(dt.Struct.from_tuples([
                                    ('iid', dt.int64),
                                    ('name', dt.string),
                                    ('price', dt.Decimal(12, 2)),
                                    ('discount_perc', dt.Decimal(12, 2)),
                                    ('shipdate', dt.string),
                                ]))
                            )
                        ]))
            ),
            (
                'web_visits',
                dt.Map(
                    dt.string,
                    dt.Struct.from_tuples([
                        ('user_agent', dt.string),
                        ('client_ip', dt.string),
                        ('visit_date', dt.string),
                        ('duration_ms', dt.int32),
                    ])
                )
            ),
            (
                'support_calls',
                dt.Array(dt.Struct.from_tuples([
                    ('agent_id', dt.int64),
                    ('call_date', dt.string),
                    ('duration_ms', dt.int64),
                    ('issue_resolved', dt.boolean),
                    ('agent_comment', dt.string)
                ]))
            ),
        ],
    )
    assert customers.schema() == expected
