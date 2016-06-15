import pytest
import ibis
from ibis.expr import datatypes as dt


def test_array():
    assert dt.validate_type('ARRAY<DOUBLE>') == dt.Array(dt.double)


def test_nested_array():
    assert dt.validate_type(
        'array<array<string>>'
    ) == dt.Array(dt.Array(dt.string))


def test_map():
    assert dt.validate_type(
        'map<string, double>'
    ) == dt.Map(dt.string, dt.double)


def test_nested_map():
    assert dt.validate_type(
        'map<int64, array<map<string, int8>>>'
    ) == dt.Map(dt.int64, dt.Array(dt.Map(dt.string, dt.int8)))


def test_map_does_not_allow_non_primitive_keys():
    with pytest.raises(SyntaxError):
        dt.validate_type('map<array<string>, double>')


def test_token_error():
    with pytest.raises(SyntaxError):
        dt.validate_type('array<string>>')


def test_empty_complex_type():
    with pytest.raises(SyntaxError):
        dt.validate_type('map<>')


def test_struct():
    orders = """array<struct<
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
    expected = dt.Array(dt.Struct.from_tuples([
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

    assert dt.validate_type(orders) == expected


def test_decimal():
    with pytest.raises(SyntaxError):
        dt.validate_type('decimal(')


def test_whole_schema():
    customers = ibis.table(
        [
            ('cid', 'int64'),
            ('mktsegment', 'string'),
            ('address', 'struct<city: string, street: string, street_number: int32, zip: int16>'),
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
            ('web_visits', 'map<string, struct<user_agent: string, client_ip: string, visit_date: string, duration_ms: int32>>'),
            ('support_calls', 'array<struct<agent_id: int64, call_date: string, duration_ms: int64, issue_resolved: boolean, agent_comment: string>>')
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
