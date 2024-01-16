from __future__ import annotations

from functools import singledispatchmethod

import sqlglot as sg
import sqlglot.expressions as sge
import toolz
from public import public
from sqlglot.dialects import Oracle
from sqlglot.dialects.dialect import create_with_partitions_sql, rename_func

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.base.sqlglot.compiler import NULL, STAR, SQLGlotCompiler
from ibis.backends.base.sqlglot.datatypes import OracleType
from ibis.backends.base.sqlglot.rewrites import replace_log2, replace_log10
from ibis.common.patterns import replace
from ibis.expr.analysis import p, x, y
from ibis.expr.rewrites import rewrite_sample


def _create_sql(self, expression: sge.Create) -> str:
    # TODO: should we use CREATE PRIVATE instead?  That will set an implicit lower bound of Oracle 18c
    properties = expression.args.get("properties")
    temporary = any(
        isinstance(prop, sge.TemporaryProperty)
        for prop in (properties.expressions if properties else [])
    )

    kind = expression.args["kind"]
    if (obj := kind.upper()) in ("TABLE", "VIEW") and temporary:
        if expression.expression:
            return f"CREATE GLOBAL TEMPORARY {obj} {self.sql(expression, 'this')} AS {self.sql(expression, 'expression')}"
        else:
            # TODO: why does autocommit not work here?  need to specify the ON COMMIT part...
            return f"CREATE GLOBAL TEMPORARY {obj} {self.sql(expression, 'this')} ON COMMIT PRESERVE ROWS"

    return create_with_partitions_sql(self, expression)


Oracle.Generator.TRANSFORMS |= {
    sge.LogicalOr: rename_func("max"),
    sge.LogicalAnd: rename_func("min"),
    sge.VariancePop: rename_func("var_pop"),
    sge.Variance: rename_func("var_samp"),
    sge.Stddev: rename_func("stddev_pop"),
    sge.StddevPop: rename_func("stddev_pop"),
    sge.StddevSamp: rename_func("stddev_samp"),
    sge.ApproxDistinct: rename_func("approx_count_distinct"),
    sge.Create: _create_sql,
}


@replace(p.WindowFunction(p.First(x, y)))
def rewrite_first(_, x, y):
    if y is not None:
        raise com.UnsupportedOperationError(
            "`first` aggregate over window does not support `where`"
        )
    return _.copy(func=ops.FirstValue(x))


@replace(p.WindowFunction(p.Last(x, y)))
def rewrite_last(_, x, y):
    if y is not None:
        raise com.UnsupportedOperationError(
            "`last` aggregate over window does not support `where`"
        )
    return _.copy(func=ops.LastValue(x))


@replace(p.WindowFunction(frame=x @ p.WindowFrame(order_by=())))
def rewrite_empty_order_by_window(_, x):
    return _.copy(frame=x.copy(order_by=(ibis.NA,)))


@replace(p.WindowFunction(p.RowNumber | p.NTile, x))
def exclude_unsupported_window_frame_from_row_number(_, x):
    return ops.Subtract(_.copy(frame=x.copy(start=None, end=None)), 1)


@replace(
    p.WindowFunction(
        p.Lag | p.Lead | p.PercentRank | p.CumeDist | p.Any | p.All,
        x @ p.WindowFrame(start=None),
    )
)
def exclude_unsupported_window_frame_from_ops(_, x):
    return _.copy(frame=x.copy(start=None, end=None))


@public
class OracleCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = "oracle"
    quoted = True
    type_mapper = OracleType
    rewrites = (
        exclude_unsupported_window_frame_from_row_number,
        exclude_unsupported_window_frame_from_ops,
        rewrite_first,
        rewrite_last,
        rewrite_empty_order_by_window,
        rewrite_sample,
        replace_log2,
        replace_log10,
        *SQLGlotCompiler.rewrites,
    )

    NAN = sge.Literal.number("binary_double_nan")
    """Backend's NaN literal."""

    POS_INF = sge.Literal.number("binary_double_infinity")
    """Backend's positive infinity literal."""

    NEG_INF = sge.Literal.number("-binary_double_infinity")
    """Backend's negative infinity literal."""

    def _aggregate(self, funcname: str, *args, where):
        func = self.f[funcname]
        if where is not None:
            args = tuple(self.if_(where, arg) for arg in args)
        return func(*args)

    @staticmethod
    def _generate_groups(groups):
        return groups

    @singledispatchmethod
    def visit_node(self, op, **kwargs):
        return super().visit_node(op, **kwargs)

    @visit_node.register(ops.Date)
    def visit_Date(self, op, *, arg):
        return sg.cast(arg, to="date")

    @visit_node.register(ops.IsNan)
    def visit_IsNan(self, op, *, arg):
        return arg.eq(self.NAN)

    @visit_node.register(ops.Log)
    def visit_Log(self, op, *, arg, base):
        return self.f.log(base, arg, dialect=self.dialect)

    @visit_node.register(ops.IsInf)
    def visit_IsInf(self, op, *, arg):
        return arg.isin(self.POS_INF, self.NEG_INF)

    @visit_node.register(ops.RandomScalar)
    def visit_RandomScalar(self, op):
        # Not using FuncGen here because of dotted function call
        return sg.func("dbms_random.value")

    @visit_node.register(ops.Pi)
    def visit_Pi(self, op):
        return self.f.acos(-1)

    @visit_node.register(ops.Cot)
    def visit_Cot(self, op, *, arg):
        return 1 / self.f.tan(arg)

    @visit_node.register(ops.Degrees)
    def visit_Degrees(self, op, *, arg):
        return 180 * arg / self.visit_node(ops.Pi())

    @visit_node.register(ops.Radians)
    def visit_Radians(self, op, *, arg):
        return self.visit_node(ops.Pi()) * arg / 180

    @visit_node.register(ops.Modulus)
    def visit_Modulus(self, op, *, left, right):
        return self.f.mod(left, right)

    @visit_node.register(ops.Levenshtein)
    def visit_Levenshtein(self, op, *, left, right):
        # Not using FuncGen here because of dotted function call
        return sg.func("utl_match.edit_distance", left, right)

    @visit_node.register(ops.StartsWith)
    def visit_StartsWith(self, op, *, arg, start):
        return self.f.substr(arg, 0, self.f.length(start)).eq(start)

    @visit_node.register(ops.EndsWith)
    def visit_EndsWith(self, op, *, arg, end):
        return self.f.substr(arg, -1 * self.f.length(end), self.f.length(end)).eq(end)

    @visit_node.register(ops.StringFind)
    def visit_StringFind(self, op, *, arg, substr, start, end):
        if end is not None:
            raise NotImplementedError("`end` is not implemented")

        sub_string = substr

        if start is not None:
            arg = self.f.substr(arg, start + 1)
            pos = self.f.instr(arg, sub_string)
            # TODO(gil): why, oh why, does this need an extra +1 on the end?
            return sg.case().when(pos > 0, pos - 1 + start).else_(-1) + 1

        return self.f.instr(arg, sub_string)

    @visit_node.register(ops.StrRight)
    def visit_StrRight(self, op, *, arg, nchars):
        return self.f.substr(arg, -nchars)

    @visit_node.register(ops.RegexExtract)
    def visit_RegexExtract(self, op, *, arg, pattern, index):
        # TODO: this is frustratingly close to working but breaks on group extraction
        return self.if_(
            index.eq(0),
            self.f.regexp_substr(arg, pattern),
            self.f.regexp_substr(arg, pattern, 1, index, "cn"),
        )

    @visit_node.register(ops.StringContains)
    def visit_StringContains(self, op, *, haystack, needle):
        return self.f.instr(haystack, needle) > 0

    @visit_node.register(ops.StringJoin)
    def visit_StringJoin(self, op, *, arg, sep):
        return self.f.concat(*toolz.interpose(sep, arg))

    ## Aggregate stuff

    @visit_node.register(ops.Correlation)
    def visit_Correlation(self, op, *, left, right, where, how):
        if how == "sample":
            raise ValueError(
                "Oracle only implements population correlation coefficient"
            )
        return self.agg.corr(left, right, where=where)

    @visit_node.register(ops.Covariance)
    def visit_Covariance(self, op, *, left, right, where, how):
        if how == "sample":
            return self.agg.covar_samp(left, right, where=where)
        return self.agg.covar_pop(left, right, where=where)

    @visit_node.register(ops.ApproxMedian)
    def visit_ApproxMedian(self, op, *, arg, where):
        return self.visit_Quantile(op, arg=arg, quantile=0.5, where=where)

    @visit_node.register(ops.Quantile)
    def visit_Quantile(self, op, *, arg, quantile, where):
        suffix = "cont" if op.arg.dtype.is_numeric() else "disc"
        funcname = f"percentile_{suffix}"

        if where is not None:
            arg = self.if_(where, arg)

        expr = sge.WithinGroup(
            this=self.f[funcname](quantile),
            expression=sge.Order(expressions=[sge.Ordered(this=arg)]),
        )
        return expr

    @visit_node.register(ops.CountDistinct)
    def visit_CountDistinct(self, op, *, arg, where):
        if where is not None:
            arg = self.if_(where, arg)

        return sge.Count(this=sge.Distinct(expressions=[arg]))

    @visit_node.register(ops.CountStar)
    def visit_CountStar(self, op, *, arg, where):
        if where is not None:
            return self.f.count(self.if_(where, 1, NULL))
        return self.f.count(STAR)

    @visit_node.register(ops.IdenticalTo)
    def visit_IdenticalTo(self, op, *, left, right):
        # sqlglot NullSafeEQ uses "is not distinct from" which isn't supported in oracle
        return (
            sg.case()
            .when(left.eq(right).or_(left.is_(NULL).and_(right.is_(NULL))), 0)
            .else_(1)
            .eq(0)
        )

    @visit_node.register(ops.Arbitrary)
    @visit_node.register(ops.ArgMax)
    @visit_node.register(ops.ArgMin)
    @visit_node.register(ops.ArrayCollect)
    @visit_node.register(ops.ArrayColumn)
    @visit_node.register(ops.ArrayFlatten)
    @visit_node.register(ops.ArrayMap)
    @visit_node.register(ops.ArrayStringJoin)
    @visit_node.register(ops.First)
    @visit_node.register(ops.Last)
    @visit_node.register(ops.Mode)
    @visit_node.register(ops.MultiQuantile)
    @visit_node.register(ops.RegexExtract)
    @visit_node.register(ops.RegexSplit)
    @visit_node.register(ops.RegexReplace)
    @visit_node.register(ops.StringSplit)
    def visit_Undefined(self, op, **_):
        raise com.OperationNotDefinedError(type(op).__name__)


_SIMPLE_OPS = {
    ops.ApproxCountDistinct: "approx_count_distinct",
    ops.BitAnd: "bit_and_agg",
    ops.BitOr: "bit_or_agg",
    ops.BitXor: "bit_xor_agg",
    ops.BitwiseAnd: "bitand",
    ops.Hash: "hash",
    ops.LPad: "lpad",
    ops.RPad: "rpad",
    ops.StringAscii: "ascii",
    ops.Strip: "trim",
    ops.Hash: "ora_hash",
}

for _op, _name in _SIMPLE_OPS.items():
    assert isinstance(type(_op), type), type(_op)
    if issubclass(_op, ops.Reduction):

        @OracleCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, where, **kw):
            return self.agg[_name](*kw.values(), where=where)

    else:

        @OracleCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, **kw):
            return self.f[_name](*kw.values())

    setattr(OracleCompiler, f"visit_{_op.__name__}", _fmt)


del _op, _name, _fmt
