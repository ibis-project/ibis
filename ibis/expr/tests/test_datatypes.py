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


def test_array_with_string_value_type():
    assert dt.Array('int32') == dt.Array(dt.int32)
    assert dt.Array(dt.Array('array<map<string, double>>')) == (
        dt.Array(dt.Array(dt.Array(dt.Map(dt.string, dt.double))))
    )


def test_map():
    assert dt.validate_type(
        'map<string, double>'
    ) == dt.Map(dt.string, dt.double)


def test_nested_map():
    assert dt.validate_type(
        'map<int64, array<map<string, int8>>>'
    ) == dt.Map(dt.int64, dt.Array(dt.Map(dt.string, dt.int8)))


def test_map_with_string_value_type():
    assert dt.Map('int32', 'double') == dt.Map(dt.int32, dt.double)
    assert dt.Map('int32', 'array<double>') == \
        dt.Map(dt.int32, dt.Array(dt.double))


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


def test_struct_with_string_types():
    result = dt.Struct.from_tuples(
        [
            ('a', 'map<double, string>'),
            ('b', 'array<map<string, array<int32>>>'),
            ('c', 'array<string>'),
            ('d', 'int8'),
        ]
    )

    assert result == dt.Struct.from_tuples(
        [
            ('a', dt.Map(dt.double, dt.string)),
            ('b', dt.Array(dt.Map(dt.string, dt.Array(dt.int32)))),
            ('c', dt.Array(dt.string)),
            ('d', dt.int8),
        ]
    )


@pytest.mark.parametrize(
    'case',
    [
        'decimal(',
        'decimal()',
        'decimal(3)',
        'decimal(,)',
        'decimal(3,)',
        'decimal(3,',
    ]
)
def test_decimal_failure(case):
    with pytest.raises(SyntaxError):
        dt.validate_type(case)


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


def test_array_type_not_equals():
    left = dt.Array(dt.string)
    right = dt.Array(dt.int32)

    assert not left.equals(right)
    assert left != right
    assert not (left == right)


def test_array_type_equals():
    left = dt.Array(dt.string)
    right = dt.Array(dt.string)

    assert left.equals(right)
    assert left == right
    assert not (left != right)


def test_timestamp_with_timezone_parser_single_quote():
    t = dt.validate_type("timestamp('US/Eastern')")
    assert isinstance(t, dt.Timestamp)
    assert t.timezone == 'US/Eastern'


def test_timestamp_with_timezone_parser_double_quote():
    t = dt.validate_type("timestamp('US/Eastern')")
    assert isinstance(t, dt.Timestamp)
    assert t.timezone == 'US/Eastern'


def test_timestamp_with_timezone_parser_invalid_timezone():
    ts = dt.validate_type("timestamp('US/Ea')")
    assert str(ts) == "timestamp('US/Ea')"


@pytest.mark.parametrize('unit', [
    'Y', 'M', 'w',  'd',  # date units
    'h',  'm',  's',  'ms', 'us', 'ns'  # time units
])
def test_interval(unit):
    definition = "interval('{}')".format(unit)
    dt.Interval(dt.int32, unit) == dt.validate_type(definition)

    definition = "interval<uint16>('{}')".format(unit)
    dt.Interval(dt.uint16, unit) == dt.validate_type(definition)

    definition = "interval<int64>('{}')".format(unit)
    dt.Interval(dt.int64, unit) == dt.validate_type(definition)


def test_interval_invalid_type():
    with pytest.raises(TypeError):
        dt.Interval(dt.float32, 'm')

    with pytest.raises(TypeError):
        dt.validate_type("interval<float>('s')")


@pytest.mark.parametrize('unit', [
    'H', 'unsupported'
])
def test_interval_unvalid_unit(unit):
    definition = "interval('{}')".format(unit)

    with pytest.raises(ValueError):
        dt.validate_type(definition)

    with pytest.raises(ValueError):
        dt.Interval(dt.int32, unit)


@pytest.mark.parametrize('case', [
    "timestamp(US/Ea)",
    "timestamp('US/Eastern\")",
    'timestamp("US/Eastern\')',
    "interval(Y)",
    "interval('Y\")",
    'interval("Y\')',
])
def test_string_argument_parsing_failure_mode(case):
    with pytest.raises(SyntaxError):
        dt.validate_type(case)


def test_timestamp_with_invalid_timezone():
    ts = dt.Timestamp('Foo/Bar&234')
    assert str(ts) == "timestamp('Foo/Bar&234')"


def test_timestamp_with_timezone_repr():
    ts = dt.Timestamp('UTC')
    assert repr(ts) == "Timestamp(timezone='UTC', nullable=True)"


def test_timestamp_with_timezone_str():
    ts = dt.Timestamp('UTC')
    assert str(ts) == "timestamp('UTC')"


def test_time():
    ts = dt.time
    assert str(ts) == "time"


def test_time_valid():
    assert dt.validate_type('time').equals(dt.time)
