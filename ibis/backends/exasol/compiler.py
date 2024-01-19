from __future__ import annotations

import contextlib
from functools import singledispatchmethod

import sqlglot.expressions as sge
from sqlglot.dialects import Postgres

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sqlglot.compiler import NULL, SQLGlotCompiler
from ibis.backends.base.sqlglot.datatypes import ExasolType
from ibis.backends.base.sqlglot.rewrites import (
    exclude_unsupported_window_frame_from_ops,
    exclude_unsupported_window_frame_from_row_number,
    rewrite_empty_order_by_window,
)
from ibis.common.patterns import replace
from ibis.expr.rewrites import p, rewrite_sample, y


def _interval(self, e):
    """Work around Exasol's inability to handle string literals in INTERVAL syntax."""
    arg = e.args["this"].this
    with contextlib.suppress(AttributeError):
        arg = arg.sql(self.dialect)
    res = f"INTERVAL '{arg}' {e.args['unit']}"
    return res


# Is postgres the best dialect to inherit from?
class Exasol(Postgres):
    """The exasol dialect."""

    class Generator(Postgres.Generator):
        TRANSFORMS = Postgres.Generator.TRANSFORMS.copy() | {
            sge.Interval: _interval,
        }

        TYPE_MAPPING = Postgres.Generator.TYPE_MAPPING.copy() | {
            sge.DataType.Type.TIMESTAMPTZ: "TIMESTAMP WITH LOCAL TIME ZONE",
        }


@replace(p.WindowFunction(p.MinRank | p.DenseRank, y @ p.WindowFrame(start=None)))
def exclude_unsupported_window_frame_from_rank(_, y):
    return ops.Subtract(
        _.copy(frame=y.copy(start=None, end=0, order_by=y.order_by or (ops.NULL,))), 1
    )


class ExasolCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = "exasol"
    type_mapper = ExasolType
    quoted = True
    rewrites = (
        rewrite_sample,
        exclude_unsupported_window_frame_from_ops,
        exclude_unsupported_window_frame_from_rank,
        exclude_unsupported_window_frame_from_row_number,
        rewrite_empty_order_by_window,
        *SQLGlotCompiler.rewrites,
    )

    @staticmethod
    def _minimize_spec(start, end, spec):
        if (
            start is None
            and isinstance(getattr(end, "value", None), ops.Literal)
            and end.value.value == 0
            and end.following
        ):
            return None
        return spec

    def _aggregate(self, funcname: str, *args, where):
        func = self.f[funcname]
        if where is not None:
            args = tuple(self.if_(where, arg, NULL) for arg in args)
        return func(*args)

    @staticmethod
    def _gen_valid_name(name: str) -> str:
        """Exasol does not allow dots in quoted column names."""
        return name.replace(".", "_")

    @singledispatchmethod
    def visit_node(self, op, **kw):
        return super().visit_node(op, **kw)

    def visit_NonNullLiteral(self, op, *, value, dtype):
        if dtype.is_date():
            return self.cast(value.isoformat(), dtype)
        elif dtype.is_timestamp():
            val = value.replace(tzinfo=None).isoformat(sep=" ", timespec="milliseconds")
            return self.cast(val, dtype)
        elif dtype.is_array() or dtype.is_struct() or dtype.is_map():
            raise com.UnsupportedBackendType(
                f"{type(dtype).__name__}s are not supported in Exasol"
            )
        elif dtype.is_uuid():
            return sge.convert(str(value))
        return super().visit_NonNullLiteral(op, value=value, dtype=dtype)

    @visit_node.register(ops.Date)
    def visit_Date(self, op, *, arg):
        return self.cast(arg, dt.date)

    @visit_node.register(ops.StartsWith)
    def visit_StartsWith(self, op, *, arg, start):
        return self.f.left(arg, self.f.length(start)).eq(start)

    @visit_node.register(ops.EndsWith)
    def visit_EndsWith(self, op, *, arg, end):
        return self.f.right(arg, self.f.length(end)).eq(end)

    @visit_node.register(ops.StringFind)
    def visit_StringFind(self, op, *, arg, substr, start, end):
        return self.f.locate(substr, arg, (start if start is not None else 0) + 1)

    @visit_node.register(ops.StringSQLILike)
    def visit_StringSQLILike(self, op, *, arg, pattern, escape):
        return self.f.upper(arg).like(self.f.upper(pattern))

    @visit_node.register(ops.StringContains)
    def visit_StringContains(self, op, *, haystack, needle):
        return self.f.locate(needle, haystack) > 0

    @visit_node.register(ops.ExtractSecond)
    def visit_ExtractSecond(self, op, *, arg):
        return self.f.floor(self.cast(self.f.extract(self.v.second, arg), op.dtype))

    @visit_node.register(ops.AnalyticVectorizedUDF)
    @visit_node.register(ops.ApproxMedian)
    @visit_node.register(ops.Arbitrary)
    @visit_node.register(ops.ArgMax)
    @visit_node.register(ops.ArgMin)
    @visit_node.register(ops.ArrayCollect)
    @visit_node.register(ops.ArrayDistinct)
    @visit_node.register(ops.ArrayFilter)
    @visit_node.register(ops.ArrayFlatten)
    @visit_node.register(ops.ArrayIntersect)
    @visit_node.register(ops.ArrayMap)
    @visit_node.register(ops.ArraySort)
    @visit_node.register(ops.ArrayStringJoin)
    @visit_node.register(ops.ArrayUnion)
    @visit_node.register(ops.ArrayZip)
    @visit_node.register(ops.BitwiseNot)
    @visit_node.register(ops.Covariance)
    @visit_node.register(ops.CumeDist)
    @visit_node.register(ops.DateAdd)
    @visit_node.register(ops.DateDelta)
    @visit_node.register(ops.DateSub)
    @visit_node.register(ops.DateFromYMD)
    @visit_node.register(ops.DayOfWeekIndex)
    @visit_node.register(ops.DayOfWeekName)
    @visit_node.register(ops.ElementWiseVectorizedUDF)
    @visit_node.register(ops.ExtractDayOfYear)
    @visit_node.register(ops.ExtractEpochSeconds)
    @visit_node.register(ops.ExtractQuarter)
    @visit_node.register(ops.ExtractWeekOfYear)
    @visit_node.register(ops.First)
    @visit_node.register(ops.IntervalFromInteger)
    @visit_node.register(ops.IsInf)
    @visit_node.register(ops.IsNan)
    @visit_node.register(ops.Last)
    @visit_node.register(ops.Levenshtein)
    @visit_node.register(ops.Median)
    @visit_node.register(ops.MultiQuantile)
    @visit_node.register(ops.Quantile)
    @visit_node.register(ops.ReductionVectorizedUDF)
    @visit_node.register(ops.RegexExtract)
    @visit_node.register(ops.RegexReplace)
    @visit_node.register(ops.RegexSearch)
    @visit_node.register(ops.RegexSplit)
    @visit_node.register(ops.RowID)
    @visit_node.register(ops.StandardDev)
    @visit_node.register(ops.Strftime)
    @visit_node.register(ops.StringJoin)
    @visit_node.register(ops.StringSplit)
    @visit_node.register(ops.StringToTimestamp)
    @visit_node.register(ops.TimeDelta)
    @visit_node.register(ops.TimestampAdd)
    @visit_node.register(ops.TimestampBucket)
    @visit_node.register(ops.TimestampDelta)
    @visit_node.register(ops.TimestampDiff)
    @visit_node.register(ops.TimestampNow)
    @visit_node.register(ops.TimestampSub)
    @visit_node.register(ops.TimestampTruncate)
    @visit_node.register(ops.TypeOf)
    @visit_node.register(ops.Unnest)
    @visit_node.register(ops.Variance)
    def visit_Undefined(self, op, **_):
        raise com.OperationNotDefinedError(type(op).__name__)

    @visit_node.register(ops.CountDistinctStar)
    def visit_Unsupported(self, op, **_):
        raise com.UnsupportedOperationError(type(op).__name__)


_SIMPLE_OPS = {
    ops.Log10: "log10",
    ops.Modulus: "mod",
    ops.All: "min",
    ops.Any: "max",
}

for _op, _name in _SIMPLE_OPS.items():
    assert isinstance(type(_op), type), type(_op)
    if issubclass(_op, ops.Reduction):

        @ExasolCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, where, **kw):
            return self.agg[_name](*kw.values(), where=where)

    else:

        @ExasolCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, **kw):
            return self.f[_name](*kw.values())

    setattr(ExasolCompiler, f"visit_{_op.__name__}", _fmt)
