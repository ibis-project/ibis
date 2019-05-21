import pytest
import numpy as np

import ibis
import ibis.expr.types as ir
import ibis.expr.rules as rlz
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops

from ibis.common import IbisTypeError
from ibis.expr.signature import Argument as Arg


def test_operation():
    class Log(ops.Node):
        arg = Arg(rlz.double())
        base = Arg(rlz.double(), default=None)

    Log(1, base=2)
    Log(1, base=2)
    Log(arg=10)


def test_ops_smoke():
    expr = ir.literal(3)
    ops.UnaryOp(expr)
    ops.Cast(expr, to='int64')
    ops.TypeOf(arg=2)
    ops.Negate(4)
    ops.Negate(4.)
    ops.NullIfZero(0)
    ops.NullIfZero(1)
    ops.IsNull(ir.null())
    ops.NotNull(ir.null())
    ops.ZeroIfNull(ir.null())
    ops.IfNull(1, ops.NullIfZero(0).to_expr())
    ops.NullIf(ir.null(), ops.NullIfZero(0).to_expr())
    ops.IsNan(np.nan)
    ops.IsInf(np.inf)
    ops.Ceil(4.5)
    ops.Floor(4.5)
    ops.Round(3.43456)
    ops.Round(3.43456, 2)
    ops.Round(3.43456, digits=1)
    ops.Clip(123, lower=30)
    ops.Clip(123, lower=30, upper=100)
    ops.BaseConvert('EEE', from_base=16, to_base=10)
    ops.Logarithm(100)
    ops.Log(100)
    ops.Log(100, base=2)
    ops.Ln(100)
    ops.Log2(100)
    ops.Log10(100)
    ops.Uppercase('asd')
    ops.Lowercase('asd')
    ops.Reverse('asd')
    ops.Strip('asd')
    ops.LStrip('asd')
    ops.RStrip('asd')
    ops.Capitalize('asd')
    ops.Substring('asd', start=1)
    ops.Substring('asd', 1)
    ops.Substring('asd', 1, length=2)
    ops.StrRight('asd', nchars=2)
    ops.Repeat('asd', times=4)
    ops.StringFind('asd', 'sd', start=1)
    ops.Translate('asd', from_str='bd', to_str='ce')
    ops.LPad('asd', length=2, pad='ss')
    ops.RPad('asd', length=2, pad='ss')
    ops.StringJoin(',', ['asd', 'bsdf'])
    ops.FuzzySearch('asd', pattern='n')
    ops.StringSQLLike('asd', pattern='as', escape='asd')
    ops.RegexExtract('asd', pattern='as', index=1)
    ops.RegexReplace('asd', 'as', 'a')
    ops.StringReplace('asd', 'as', 'a')
    ops.StringSplit('asd', 's')
    ops.StringConcat(['s', 'e'])


def test_instance_of_operation():
    class MyOperation(ops.Node):
        arg = Arg(ir.IntegerValue)

    MyOperation(ir.literal(5))

    with pytest.raises(IbisTypeError):
        MyOperation(ir.literal('string'))


def test_array_input():
    class MyOp(ops.ValueOp):
        value = Arg(rlz.value(dt.Array(dt.double)))
        output_type = rlz.typeof('value')

    raw_value = [1.0, 2.0, 3.0]
    op = MyOp(raw_value)
    result = op.value
    expected = ibis.literal(raw_value)
    assert result.equals(expected)


def test_custom_table_expr():
    class MyTableExpr(ir.TableExpr):
        pass

    class SpecialTable(ops.DatabaseTable):
        def output_type(self):
            return MyTableExpr

    con = ibis.pandas.connect({})
    node = SpecialTable('foo', ibis.schema([('a', 'int64')]), con)
    expr = node.to_expr()
    assert isinstance(expr, MyTableExpr)
