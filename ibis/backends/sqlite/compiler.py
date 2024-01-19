from __future__ import annotations

from functools import singledispatchmethod

import sqlglot as sg
import sqlglot.expressions as sge
from public import public

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sqlglot.compiler import SQLGlotCompiler
from ibis.backends.base.sqlglot.datatypes import SQLiteType


@public
class SQLiteCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = "sqlite"
    type_mapper = SQLiteType

    NAN = sge.NULL
    POS_INF = sge.Literal.number("1e999")
    NEG_INF = sge.Literal.number("-1e999")

    def _aggregate(self, funcname: str, *args, where):
        expr = self.f[funcname](*args)
        if where is not None:
            return sge.Filter(this=expr, expression=sge.Where(this=where))
        return expr

    @singledispatchmethod
    def visit_node(self, op, **kw):
        return super().visit_node(op, **kw)

    @visit_node.register(ops.Levenshtein)
    @visit_node.register(ops.RegexSplit)
    @visit_node.register(ops.StringSplit)
    @visit_node.register(ops.ArrayStringJoin)
    @visit_node.register(ops.IsNan)
    @visit_node.register(ops.IsInf)
    @visit_node.register(ops.Covariance)
    @visit_node.register(ops.Correlation)
    @visit_node.register(ops.Quantile)
    @visit_node.register(ops.MultiQuantile)
    @visit_node.register(ops.Median)
    @visit_node.register(ops.ApproxMedian)
    @visit_node.register(ops.ArrayCollect)
    @visit_node.register(ops.ArrayContains)
    @visit_node.register(ops.ArrayFlatten)
    @visit_node.register(ops.ArrayLength)
    @visit_node.register(ops.ArraySort)
    @visit_node.register(ops.ArrayStringJoin)
    @visit_node.register(ops.CountDistinctStar)
    def visit_Undefined(self, op, **kwargs):
        return super().visit_Undefined(op, **kwargs)

    @visit_node.register(ops.StartsWith)
    def visit_StartsWith(self, op, *, arg, start):
        return arg.like(self.f.concat(start, "%"))

    @visit_node.register(ops.EndsWith)
    def visit_EndsWith(self, op, *, arg, end):
        return arg.like(self.f.concat("%", end))

    @visit_node.register(ops.StrRight)
    def visit_StrRight(self, op, *, arg, nchars):
        return self.f.substr(arg, -nchars, nchars)

    @visit_node.register(ops.StringFind)
    def visit_StringFind(self, op, *, arg, substr, start, end):
        if op.end is not None:
            raise NotImplementedError("`end` not yet implemented")

        if op.start is not None:
            arg = self.f.substr(arg, start + 1)
            pos = self.f.instr(arg, substr)
            return sg.case().when(pos > 0, pos + start).else_(0)

        return self.f.instr(arg, substr)

    @visit_node.register(ops.StringJoin)
    def visit_StringJoin(self, op, *, arg, sep):
        args = [arg[0]]
        for item in arg[1:]:
            args.extend([sep, item])
        return self.f.concat(*args)

    @visit_node.register(ops.StringContains)
    def visit_Contains(self, op, *, haystack, needle):
        return self.f.instr(haystack, needle) >= 1

    @visit_node.register(ops.ExtractQuery)
    def visit_ExtractQuery(self, op, *, arg, key):
        if op.key is None:
            return self.f._ibis_extract_full_query(arg)
        return self.f._ibis_extract_query(arg, key)

    @visit_node.register(ops.Greatest)
    def visit_Greatest(self, op, *, arg):
        return self.f.max(*arg)

    @visit_node.register(ops.Least)
    def visit_Least(self, op, *, arg):
        return self.f.min(*arg)

    @visit_node.register(ops.Clip)
    def visit_Clip(self, op, *, arg, lower, upper):
        if upper is not None:
            arg = self.if_(arg.is_(sge.NULL), arg, self.f.min(upper, arg))

        if lower is not None:
            arg = self.if_(arg.is_(sge.NULL), arg, self.f.max(lower, arg))

        return arg

    @visit_node.register(ops.RandomScalar)
    def visit_RandomScalar(self, op):
        return self.f.random() / sge.Literal.number(float(-1 << 64))

    @visit_node.register(ops.Cot)
    def visit_Cot(self, op, *, arg):
        return 1 / self.f.tan(arg)

    @visit_node.register(ops.Arbitrary)
    def visit_Arbitrary(self, op, *, arg, how, where):
        if op.how == "heavy":
            raise com.OperationNotDefinedError(
                "how='heavy' not implemented for the SQLite backend"
            )

        return self._aggregate(f"_ibis_sqlite_arbitrary_{how}", arg, where=where)

    @visit_node.register(ops.ArgMin)
    def visit_ArgMin(self, *args, **kwargs):
        return self._visit_arg_reduction("min", *args, **kwargs)

    @visit_node.register(ops.ArgMax)
    def visit_ArgMax(self, *args, **kwargs):
        return self._visit_arg_reduction("max", *args, **kwargs)

    def _visit_arg_reduction(self, func, op, *, arg, key, where):
        cond = arg.is_(sg.not_(sge.NULL))

        if op.where is not None:
            cond = sg.and_(cond, where)

        agg = self._aggregate(func, key, where=cond)
        return self.f.anon.json_extract(self.f.json_array(arg, agg), "$[0]")

    @visit_node.register(ops.Variance)
    def visit_Variance(self, op, *, arg, how, where):
        return self._aggregate(f"_ibis_sqlite_var_{op.how}", arg, where=where)

    @visit_node.register(ops.StandardDev)
    def visit_StandardDev(self, op, *, arg, how, where):
        var = self._aggregate(f"_ibis_sqlite_var_{op.how}", arg, where=where)
        return self.f.sqrt(var)

    @visit_node.register(ops.ApproxCountDistinct)
    def visit_ApproxCountDistinct(self, op, *, arg, where):
        return self.agg.count(sge.Distinct(expressions=[arg]), where=where)

    @visit_node.register(ops.CountDistinct)
    def visit_CountDistinct(self, op, *, arg, where):
        return self.agg.count(sge.Distinct(expressions=[arg]), where=where)

    def visit_NonNullLiteral(self, op, *, value, dtype):
        if dtype.is_decimal():
            value = float(value)
            dtype = dt.double(nullable=dtype.nullable)
        elif dtype.is_uuid():
            value = str(value)
            dtype = dt.string(nullable=dtype.nullable)
        return super().visit_NonNullLiteral(op, value=value, dtype=dtype)


_SIMPLE_OPS = {
    ops.RegexReplace: "_ibis_sqlite_regex_replace",
    ops.RegexExtract: "_ibis_sqlite_regex_extract",
    ops.RegexSearch: "_ibis_sqlite_regex_search",
    ops.Translate: "_ibis_sqlite_translate",
    ops.Capitalize: "_ibis_sqlite_capitalize",
    ops.Reverse: "_ibis_sqlite_reverse",
    ops.RPad: "_ibis_sqlite_rpad",
    ops.LPad: "_ibis_sqlite_lpad",
    ops.Repeat: "_ibis_sqlite_repeat",
    ops.StringAscii: "_ibis_sqlite_string_ascii",
    ops.ExtractAuthority: "_ibis_extract_authority",
    ops.ExtractFragment: "_ibis_extract_fragment",
    ops.ExtractHost: "_ibis_extract_host",
    ops.ExtractPath: "_ibis_extract_path",
    ops.ExtractProtocol: "_ibis_extract_protocol",
    ops.ExtractUserInfo: "_ibis_extract_user_info",
    ops.BitwiseXor: "_ibis_sqlite_xor",
    ops.BitwiseNot: "_ibis_sqlite_inv",
    ops.Modulus: "mod",
    ops.TypeOf: "typeof",
    ops.BitOr: "_ibis_sqlite_bit_or",
    ops.BitAnd: "_ibis_sqlite_bit_and",
    ops.BitXor: "_ibis_sqlite_bit_xor",
    ops.First: "_ibis_sqlite_arbitrary_first",
    ops.Last: "_ibis_sqlite_arbitrary_last",
    ops.Mode: "_ibis_sqlite_mode",
}


for _op, _name in _SIMPLE_OPS.items():
    assert isinstance(type(_op), type), type(_op)
    if issubclass(_op, ops.Reduction):

        @SQLiteCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, where, **kw):
            return self.agg[_name](*kw.values(), where=where)

    else:

        @SQLiteCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, **kw):
            return self.f[_name](*kw.values())

    setattr(SQLiteCompiler, f"visit_{_op.__name__}", _fmt)


del _op, _name, _fmt
