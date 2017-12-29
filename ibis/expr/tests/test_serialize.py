import ibis
from ibis.expr.serialize import serialize, deserialize


def assert_round_trip(expr):
    metadata = 'foo'
    s = serialize(expr, metadata=metadata)
    result = deserialize(s)
    assert result['expr'].equals(expr)
    assert result['metadata'] == metadata


def test_table(table):
    assert_round_trip(table)


def test_projection(table):
    assert_round_trip(table[table.columns])
    assert_round_trip(table[table.columns[0:2]])


def test_predicates(table):
    t = table[table.columns]
    t = t[table.a > 1]
    assert_round_trip(t)


def test_filter_self_join():
    # GH #667
    purchases = ibis.table([('region', 'string'),
                            ('kind', 'string'),
                            ('user', 'int64'),
                            ('amount', 'double')], 'purchases')

    metric = purchases.amount.sum().name('total')
    agged = (purchases.group_by(['region', 'kind'])
             .aggregate(metric))

    left = agged[agged.kind == 'foo']
    right = agged[agged.kind == 'bar']

    cond = left.region == right.region
    joined = left.join(right, cond)
    assert_round_trip(joined)


def test_compound(table):

    t = table.a + table.b / 2
    assert_round_trip(t)


def test_literal():
    assert_round_trip(ibis.literal(5))
