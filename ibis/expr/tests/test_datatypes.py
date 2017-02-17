import pytest
import ibis
from ibis import IbisError
from ibis.expr import datatypes as dt
from ibis.expr.rules import highest_precedence_type
import ibis.expr.api as api
import ibis.expr.types as types
import ibis.expr.rules as rules


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


def test_decimal_failure():
    with pytest.raises(SyntaxError):
        dt.validate_type('decimal(')


@pytest.mark.parametrize(
    'spec',
    ['varchar', 'varchar(10)', 'char', 'char(10)']
)
def test_char_varchar(spec):
    assert dt.validate_type(spec) == dt.string


@pytest.mark.parametrize(
    'spec',
    ['varchar(', 'varchar)', 'varchar()', 'char(', 'char)', 'char()']
)
def test_char_varchar_invalid(spec):
    with pytest.raises(SyntaxError):
        dt.validate_type(spec)


@pytest.mark.parametrize('spec', dt._primitive_types.keys())
def test_primitive(spec):
    assert dt.validate_type(spec) == dt._primitive_types[spec]


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


def test_precedence_with_no_arguments():
    with pytest.raises(ValueError) as e:
        highest_precedence_type([])
    assert str(e.value) == 'Must pass at least one expression'


def test_rule_instance_of():
    class MyOperation(types.Node):

        input_type = [rules.instance_of(types.IntegerValue)]

    MyOperation([api.literal(5)])

    with pytest.raises(IbisError):
        MyOperation([api.literal('string')])


def test_literal_mixed_type_fails():
    data = [1, 'a']
    with pytest.raises(TypeError):
        ibis.literal(data)
