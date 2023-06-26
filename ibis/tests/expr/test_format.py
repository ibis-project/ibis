from __future__ import annotations

import re

import pytest

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.format
import ibis.expr.operations as ops
import ibis.legacy.udf.vectorized as udf
from ibis.expr.operations.relations import Projection


def test_format_table_column(table):
    # GH #507
    result = repr(table.f)
    assert 'float64' in result


def test_format_projection(table):
    # This should produce a ref to the projection
    proj = table[['c', 'a', 'f']]
    repr(proj['a'])


def test_table_type_output():
    foo = ibis.table(
        [
            ('job', 'string'),
            ('dept_id', 'string'),
            ('year', 'int32'),
            ('y', 'double'),
        ],
        name='foo',
    )

    expr = foo.dept_id == foo.view().dept_id
    result = repr(expr)

    assert 'SelfReference[r0]' in result
    assert 'UnboundTable: foo' in result


def test_aggregate_arg_names(table):
    # Not sure how to test this *well*

    t = table

    by_exprs = [t.g.name('key1'), t.f.round().name('key2')]
    metrics = [t.c.sum().name('c'), t.d.mean().name('d')]

    expr = t.group_by(by_exprs).aggregate(metrics)
    result = repr(expr)
    assert 'metrics' in result
    assert 'by' in result


def test_format_multiple_join_with_projection():
    # Star schema with fact table
    table = ibis.table(
        [
            ('c', 'int32'),
            ('f', 'double'),
            ('foo_id', 'string'),
            ('bar_id', 'string'),
        ],
        'one',
    )

    table2 = ibis.table([('foo_id', 'string'), ('value1', 'double')], 'two')

    table3 = ibis.table([('bar_id', 'string'), ('value2', 'double')], 'three')

    filtered = table[table['f'] > 0]

    pred1 = filtered['foo_id'] == table2['foo_id']
    pred2 = filtered['bar_id'] == table3['bar_id']

    j1 = filtered.left_join(table2, [pred1])
    j2 = j1.inner_join(table3, [pred2])

    # Project out the desired fields
    view = j2[[filtered, table2['value1'], table3['value2']]]

    # it works!
    repr(view)


def test_memoize_database_table(con):
    table = con.table('test1')
    table2 = con.table('test2')

    filter_pred = table['f'] > 0
    table3 = table[filter_pred]
    join_pred = table3['g'] == table2['key']

    joined = table2.inner_join(table3, [join_pred])

    met1 = (table3['f'] - table2['value']).mean().name('foo')
    result = joined.aggregate(
        [met1, table3['f'].sum().name('bar')], by=[table3['g'], table2['key']]
    )

    formatted = repr(result)
    assert formatted.count('test1') == 1
    assert formatted.count('test2') == 1


def test_memoize_filtered_table():
    airlines = ibis.table(
        [('dest', 'string'), ('origin', 'string'), ('arrdelay', 'int32')],
        'airlines',
    )

    dests = ['ORD', 'JFK', 'SFO']
    t = airlines[airlines.dest.isin(dests)]
    delay_filter = t.dest.topk(10, by=t.arrdelay.mean())

    result = repr(delay_filter)
    assert result.count('Selection') == 1


def test_memoize_insert_sort_key(con):
    table = con.table('airlines')

    t = table['arrdelay', 'dest']
    expr = t.group_by('dest').mutate(
        dest_avg=t.arrdelay.mean(), dev=t.arrdelay - t.arrdelay.mean()
    )

    worst = expr[expr.dev.notnull()].order_by(ibis.desc('dev')).limit(10)

    result = repr(worst)

    assert result.count('airlines') == 1


def test_named_value_expr_show_name(table):
    expr = table.f * 2
    expr2 = expr.name('baz')

    # it works!
    repr(expr)

    result2 = repr(expr2)

    assert 'baz' in result2


def test_memoize_filtered_tables_in_join():
    # related: GH #667
    purchases = ibis.table(
        [
            ('region', 'string'),
            ('kind', 'string'),
            ('user', 'int64'),
            ('amount', 'double'),
        ],
        'purchases',
    )

    metric = purchases.amount.sum().name('total')
    agged = purchases.group_by(['region', 'kind']).aggregate(metric)

    left = agged[agged.kind == 'foo']
    right = agged[agged.kind == 'bar']

    cond = left.region == right.region
    joined = left.join(right, cond)[left, right.total.name('right_total')]

    result = repr(joined)

    # one for each aggregation
    # joins are shown without the word `predicates` above them
    # since joins only have predicates as arguments
    assert result.count('predicates') == 2


def test_argument_repr_shows_name():
    t = ibis.table([('fakecolname1', 'int64')], name='fakename2')
    expr = t.fakecolname1.nullif(2)
    result = repr(expr)
    assert 'fakecolname1' in result
    assert 'fakename2' in result


def test_scalar_parameter_formatting():
    value = ibis.param('array<date>')
    assert re.match(r"^param_\d+: \$\(array<date>\)$", str(value)) is not None

    value = ibis.param('int64').name('my_param')
    assert str(value) == 'my_param: $(int64)'


def test_same_column_multiple_aliases():
    table = ibis.table([('col', 'int64')], name='t')
    expr = table[table.col.name('fakealias1'), table.col.name('fakealias2')]
    result = repr(expr)

    assert "UnboundTable: t" in result
    assert "col int64" in result
    assert "fakealias1: r0.col" in result
    assert "fakealias2: r0.col" in result


def test_scalar_parameter_repr():
    value = ibis.param(dt.timestamp).name('value')
    assert repr(value) == "value: $(timestamp)"


def test_repr_exact():
    # NB: This is the only exact repr test. Do
    # not add new exact repr tests. New repr tests
    # should only check for the presence of substrings.
    table = ibis.table(
        [("col", "int64"), ("col2", "string"), ("col3", "double")],
        name="t",
    ).mutate(col4=lambda t: t.col2.length())
    result = repr(table)
    expected = """\
r0 := UnboundTable: t
  col  int64
  col2 string
  col3 float64

Selection[r0]
  selections:
    r0
    col4: StringLength(r0.col2)"""
    assert result == expected


def test_complex_repr():
    t = (
        ibis.table(dict(a="int64"), name="t")
        .filter([lambda t: t.a < 42, lambda t: t.a >= 42])
        .mutate(x=lambda t: t.a + 42)
        .group_by("x")
        .aggregate(y=lambda t: t.a.sum())
        .limit(10)
    )
    repr(t)


def test_value_exprs_repr():
    t = ibis.table(dict(a="int64", b="string"), name="t")
    assert "r0.a" in repr(t.a)
    assert "Sum(r0.a)" in repr(t.a.sum())


def test_show_types(monkeypatch):
    monkeypatch.setattr(ibis.options.repr, "show_types", True)

    t = ibis.table(dict(a="int64", b="string"), name="t")
    expr = t.a / 1.0
    assert "# int64" in repr(t.a)
    assert "# float64" in repr(expr)
    assert "# float64" in repr(expr.sum())


@pytest.mark.parametrize("cls", set(ops.TableNode.__subclasses__()) - {Projection})
def test_tables_have_format_rules(cls):
    assert cls in ibis.expr.format.fmt_table_op.registry


@pytest.mark.parametrize("cls", [ops.PhysicalTable, ops.TableNode])
def test_tables_have_format_value_rules(cls):
    assert cls in ibis.expr.format.fmt_value.registry


@pytest.mark.parametrize(
    "f",
    [
        lambda t1, _: t1.count(),
        lambda t1, t2: t1.join(t2, t1.a == t2.a).count(),
        lambda t1, t2: ibis.union(t1, t2).count(),
    ],
)
def test_table_value_expr(f):
    t1 = ibis.table([("a", "int"), ("b", "float")], name="t1")
    t2 = ibis.table([("a", "int"), ("b", "float")], name="t2")
    expr = f(t1, t2)
    repr(expr)  # smoketest


def test_window_no_group_by():
    t = ibis.table(dict(a="int64", b="string"), name="t")
    expr = t.a.mean().over(ibis.window(preceding=0))
    result = repr(expr)
    assert "group_by=[]" not in result


def test_window_group_by():
    t = ibis.table(dict(a="int64", b="string"), name="t")
    expr = t.a.mean().over(ibis.window(group_by=t.b))

    result = repr(expr)
    assert "start=0" not in result
    assert "group_by=[r0.b]" in result


def test_fillna():
    t = ibis.table(dict(a="int64", b="string"), name="t")

    expr = t.fillna({"a": 3})
    repr(expr)

    expr = t[["a"]].fillna(3)
    repr(expr)

    expr = t[["b"]].fillna("foo")
    repr(expr)


def test_asof_join():
    left = ibis.table([("time1", 'int32'), ('value', 'double')])
    right = ibis.table([("time2", 'int32'), ('value2', 'double')])
    joined = left.asof_join(right, [("time1", "time2")]).inner_join(
        right, left.value == right.value2
    )
    rep = repr(joined)
    assert rep.count("InnerJoin") == 1
    assert rep.count("AsOfJoin") == 1


def test_two_inner_joins():
    left = ibis.table([("time1", 'int32'), ('value', 'double'), ('a', 'string')])
    right = ibis.table([("time2", 'int32'), ('value2', 'double'), ('b', 'string')])
    joined = left.inner_join(right, left.a == right.b).inner_join(
        right, left.value == right.value2
    )
    rep = repr(joined)
    assert rep.count("InnerJoin") == 2


def test_destruct_selection():
    table = ibis.table([('col', 'int64')], name='t')

    @udf.reduction(
        input_type=['int64'],
        output_type=dt.Struct({'sum': 'int64', 'mean': 'float64'}),
    )
    def multi_output_udf(v):
        return v.sum(), v.mean()

    expr = table.aggregate(multi_output_udf(table['col']).destructure())
    result = repr(expr)

    assert "sum:  StructField(ReductionVectorizedUDF" in result
    assert "mean: StructField(ReductionVectorizedUDF" in result


@pytest.mark.parametrize(
    "literal, typ, output",
    [(42, None, '42'), ('42', None, "'42'"), (42, "double", '42.0')],
)
def test_format_literal(literal, typ, output):
    assert repr(ibis.literal(literal, type=typ)) == output


def test_format_dummy_table():
    t = ops.DummyTable([ibis.array([1], type="array<int8>").name("foo")]).to_expr()
    result = repr(t)
    assert "DummyTable" in result
    assert "foo array<int8>" in result
