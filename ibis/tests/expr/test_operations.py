from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pytest

import ibis
import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.common.patterns import ValidationError

t = ibis.table([('a', 'int64')], name='t')

true = ir.literal(True)
false = ir.literal(False)
two = ir.literal(2)
three = ir.literal(3)


@pytest.fixture(scope='module')
def operations(request):
    true = ir.literal(True)
    false = ir.literal(False)
    two = ir.literal(2)
    three = ir.literal(3)
    return [
        ops.Cast(three, to='int64'),
        ops.TypeOf(arg=2),
        ops.Negate(4),
        ops.Negate(4.0),
        ops.NullIfZero(0),
        ops.NullIfZero(1),
        ops.IsNull(ir.null()),
        ops.NotNull(ir.null()),
        ops.ZeroIfNull(ir.null()),
        ops.IfNull(1, ops.NullIfZero(0).to_expr()),
        ops.NullIf(ir.null(), ops.NullIfZero(0).to_expr()),
        ops.IsNan(np.nan),
        ops.IsInf(np.inf),
        ops.Ceil(4.5),
        ops.Floor(4.5),
        ops.Round(3.43456),
        ops.Round(3.43456, 2),
        ops.Round(3.43456, digits=1),
        ops.Clip(123, lower=30),
        ops.Clip(123, lower=30, upper=100),
        ops.BaseConvert('EEE', from_base=16, to_base=10),
        ops.Logarithm(100),
        ops.Log(100),
        ops.Log(100, base=2),
        ops.Ln(100),
        ops.Log2(100),
        ops.Log10(100),
        ops.Uppercase('asd'),
        ops.Lowercase('asd'),
        ops.Reverse('asd'),
        ops.Strip('asd'),
        ops.LStrip('asd'),
        ops.RStrip('asd'),
        ops.Capitalize('asd'),
        ops.Substring('asd', start=1),
        ops.Substring('asd', 1),
        ops.Substring('asd', 1, length=2),
        ops.StrRight('asd', nchars=2),
        ops.Repeat('asd', times=4),
        ops.StringFind('asd', 'sd', start=1),
        ops.Translate('asd', from_str='bd', to_str='ce'),
        ops.LPad('asd', length=2, pad='ss'),
        ops.RPad('asd', length=2, pad='ss'),
        ops.StringJoin(',', ['asd', 'bsdf']),
        ops.FuzzySearch('asd', pattern='n'),
        ops.StringSQLLike('asd', pattern='as', escape='asd'),
        ops.RegexExtract('asd', pattern='as', index=1),
        ops.RegexReplace('asd', 'as', 'a'),
        ops.StringReplace('asd', 'as', 'a'),
        ops.StringSplit('asd', 's'),
        ops.StringConcat(('s', 'e')),
        ops.StartsWith('asd', 'as'),
        ops.EndsWith('asd', 'xyz'),
        ops.Not(false),
        ops.And(false, true),
        ops.Or(false, true),
        ops.GreaterEqual(three, two),
        ops.Sum(t.a),
        t.a.op(),
    ]


class Expr:
    def __init__(self, op):
        self.op = op


class Base(ops.Node):
    def to_expr(self):
        return Expr(self)


class Name(Base):
    name: str


class NamedValue(Base):
    value: int
    name: Name


class Values(Base):
    lst: Tuple[ops.Node, ...]


one = NamedValue(value=1, name=Name("one"))
two = NamedValue(value=2, name=Name("two"))
three = NamedValue(value=3, name=Name("three"))
values = Values((one, two, three))


def test_node_base():
    assert one.__args__ == (1, Name("one"))
    assert values.__args__ == ((one, two, three),)

    calls = []
    returns = {
        Name("one"): "Name_one",
        Name("two"): "Name_two",
        Name("three"): "Name_three",
        NamedValue(1, Name("one")): "NamedValue_1_one",
        NamedValue(2, Name("two")): "NamedValue_2_two",
        NamedValue(3, Name("three")): "NamedValue_3_three",
        values: "final",
    }

    def record(node, _, *args, **kwargs):
        calls.append((node, args, kwargs))
        return returns[node]

    results = values.map(record)

    assert results == returns
    assert calls == [
        (Name("one"), (), {"name": "one"}),
        (Name("two"), (), {"name": "two"}),
        (Name("three"), (), {"name": "three"}),
        (one, (), {"value": 1, "name": "Name_one"}),
        (two, (), {"value": 2, "name": "Name_two"}),
        (three, (), {"value": 3, "name": "Name_three"}),
        (
            values,
            (),
            {"lst": ("NamedValue_1_one", "NamedValue_2_two", "NamedValue_3_three")},
        ),
    ]


def test_node_subtitution():
    class Aliased(Base):
        arg: ops.Node
        name: str

    ketto = Aliased(one, "ketto")
    subs = {Name("one"): Name("zero"), two: ketto}

    new_values = values.replace(subs)
    expected = Values((NamedValue(value=1, name=Name("zero")), ketto, three))

    assert expected == new_values


def test_value_annotations():
    class Op1(ops.Value):
        arg: ops.Value

        output_dtype = dt.int64
        output_shape = ds.scalar

    class Op2(ops.Value):
        arg: ops.Value[dt.Any, ds.Any]

        output_dtype = dt.int64
        output_shape = ds.scalar

    class Op3(ops.Value):
        arg: ops.Value[dt.Integer, ds.Any]

        output_dtype = dt.int64
        output_shape = ds.scalar

    class Op4(ops.Value):
        arg: ops.Value[dt.Integer, ds.Scalar]

        output_dtype = dt.int64
        output_shape = ds.scalar

    assert Op1(1).arg.dtype == dt.int8
    assert Op2(1).arg.dtype == dt.int8
    assert Op3(1).arg.dtype == dt.int8
    assert Op4(1).arg.dtype == dt.int8


def test_operation():
    class Logarithm(ir.Expr):
        pass

    class Log(ops.Node):
        arg: ops.Value[dt.Float64, ds.Any]
        base: Optional[ops.Value[dt.Float64, ds.Any]] = None

        def to_expr(self):
            return Logarithm(self)

    Log(1, base=2)
    Log(1, base=2)
    Log(arg=10)

    assert isinstance(Log(arg=100).to_expr(), Logarithm)


def test_operation_nodes_are_slotted(operations):
    for op in operations:
        assert hasattr(op, "__slots__")
        assert not hasattr(op, "__dict__")


def test_instance_of_operation():
    class MyOperation(ops.Node):
        arg: ir.IntegerValue

        def to_expr(self):
            return ir.IntegerScalar(self)

    MyOperation(ir.literal(5))

    with pytest.raises(ValidationError):
        MyOperation(ir.literal('string'))


def test_array_input():
    class MyOp(ops.Value):
        value: ops.Value[dt.Array[dt.Float64], ds.Any]
        output_dtype = rlz.dtype_like('value')
        output_shape = rlz.shape_like('value')

    raw_value = [1.0, 2.0, 3.0]
    op = MyOp(raw_value)

    expected = ibis.literal(raw_value)
    assert op.value == expected.op()


def test_custom_table_expr():
    class MyTable(ir.Table):
        pass

    class SpecialTable(ops.DatabaseTable):
        def to_expr(self):
            return MyTable(self)

    con = ibis.pandas.connect({})
    node = SpecialTable('foo', ibis.schema([('a', 'int64')]), con)
    expr = node.to_expr()
    assert isinstance(expr, MyTable)


@pytest.fixture(scope='session')
def dummy_op():
    class DummyOp(ops.Value):
        arg: ops.Value

    return DummyOp


def test_too_many_args_not_allowed(dummy_op):
    with pytest.raises(TypeError):
        dummy_op(1, 2)


def test_too_few_args_not_allowed(dummy_op):
    with pytest.raises(TypeError):
        dummy_op()


def test_operation_class_aliases():
    assert ops.ValueOp is ops.Value
    assert ops.UnaryOp is ops.Unary
    assert ops.BinaryOp is ops.Binary
    assert ops.WindowOp is ops.Window
    assert ops.AnalyticOp is ops.Analytic


def test_expression_class_aliases():
    assert ir.TableExpr is ir.Table
    assert ir.ValueExpr is ir.Value
    assert ir.ScalarExpr is ir.Scalar
    assert ir.ColumnExpr is ir.Column
    assert ir.AnyValue is ir.Value
    assert ir.AnyScalar is ir.Scalar
    assert ir.AnyColumn is ir.Column
