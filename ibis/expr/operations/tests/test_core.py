from __future__ import annotations

from typing import Optional

import pytest

import ibis
import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.types as ir
from ibis.common.annotations import ValidationError
from ibis.common.patterns import EqualTo

t = ibis.table([("a", "int64")], name="t")

true = ir.literal(True)
false = ir.literal(False)
two = ir.literal(2)
three = ir.literal(3)


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
    lst: tuple[ops.Node, ...]


one = NamedValue(value=1, name=Name("one"))
two = NamedValue(value=2, name=Name("two"))
three = NamedValue(value=3, name=Name("three"))
values = Values((one, two, three))


def test_node_base():
    assert hasattr(one, "__slots__")
    assert not hasattr(one, "__dict__")
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

    first_rule = EqualTo(Name("one")) >> Name("zero")
    second_rule = EqualTo(two) >> ketto

    new_values = values.replace(first_rule | second_rule)
    expected = Values((NamedValue(value=1, name=Name("zero")), ketto, three))

    assert expected == new_values


def test_value_annotations():
    class Op1(ops.Value):
        arg: ops.Value

        dtype = dt.int64
        shape = ds.scalar

    class Op2(ops.Value):
        arg: ops.Value[dt.Any, ds.Any]

        dtype = dt.int64
        shape = ds.scalar

    class Op3(ops.Value):
        arg: ops.Value[dt.Integer, ds.Any]

        dtype = dt.int64
        shape = ds.scalar

    class Op4(ops.Value):
        arg: ops.Value[dt.Integer, ds.Scalar]

        dtype = dt.int64
        shape = ds.scalar

    assert Op1(1).arg.dtype == dt.int8
    assert Op2(1).arg.dtype == dt.int8
    assert Op3(1).arg.dtype == dt.int8
    assert Op4(1).arg.dtype == dt.int8


def test_operation_definition():
    class Logarithm(ir.Expr):
        pass

    class Log(ops.Node):
        arg: ops.Value[dt.Float64, ds.Any]
        base: Optional[ops.Value[dt.Float64, ds.Any]] = None

        def to_expr(self):
            return Logarithm(self)

    assert Log(1, base=2).arg == ops.Literal(1, dtype=dt.float64)
    assert Log(1, base=2).base == ops.Literal(2, dtype=dt.float64)
    assert Log(arg=10).arg == ops.Literal(10, dtype=dt.float64)
    assert Log(arg=10).base is None

    assert isinstance(Log(arg=100).to_expr(), Logarithm)


def test_instance_of_operation():
    class MyOperation(ops.Node):
        arg: ir.IntegerValue

        def to_expr(self):
            return ir.IntegerScalar(self)

    MyOperation(ir.literal(5))

    with pytest.raises(ValidationError):
        MyOperation(ir.literal("string"))


def test_array_input():
    class MyOp(ops.Value):
        value: ops.Value[dt.Array[dt.Float64], ds.Any]
        dtype = rlz.dtype_like("value")
        shape = rlz.shape_like("value")

    raw_value = [1.0, 2.0, 3.0]
    op = MyOp(raw_value)

    expected = ibis.literal(raw_value)
    assert op.value == expected.op()


def test_custom_table_expr():
    class MyTable(ir.Table):
        pass

    class SpecialTable(ops.UnboundTable):
        def to_expr(self):
            return MyTable(self)

    node = SpecialTable(name="foo", schema=ibis.schema([("a", "int64")]))
    expr = node.to_expr()
    assert isinstance(expr, MyTable)


def test_too_many_or_too_few_args_not_allowed():
    class DummyOp(ops.Value):
        arg: ops.Value

    with pytest.raises(ValidationError):
        DummyOp(1, 2)

    with pytest.raises(ValidationError):
        DummyOp()


def test_getitem_on_column_is_error():
    t = ibis.table(dict(a="int"))

    with pytest.raises(TypeError, match="#ibis-for-pandas-users"):
        t.a[0]

    with pytest.raises(TypeError, match="#ibis-for-pandas-users"):
        t.a[:1]


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
