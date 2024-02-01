from __future__ import annotations

import contextlib
from functools import singledispatchmethod

import sqlglot as sg
import sqlglot.expressions as sge
from sqlglot.dialects import Hive
from sqlglot.dialects.dialect import rename_func

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis import util
from ibis.backends.base.sqlglot.compiler import NULL, STAR, SQLGlotCompiler
from ibis.backends.base.sqlglot.datatypes import ImpalaType
from ibis.backends.base.sqlglot.rewrites import (
    rewrite_empty_order_by_window,
    rewrite_first_to_first_value,
    rewrite_last_to_last_value,
)
from ibis.expr.rewrites import rewrite_sample


def _interval(self, e):
    """Work around Impala's inability to handle string literals in INTERVAL syntax."""
    arg = e.args["this"].this
    with contextlib.suppress(AttributeError):
        arg = arg.sql(self.dialect)
    res = f"INTERVAL {arg} {e.args['unit']}"
    return res


class Impala(Hive):
    class Generator(Hive.Generator):
        TRANSFORMS = Hive.Generator.TRANSFORMS.copy() | {
            sge.ApproxDistinct: rename_func("ndv"),
            sge.IsNan: rename_func("is_nan"),
            sge.IsInf: rename_func("is_inf"),
            sge.DayOfWeek: rename_func("dayofweek"),
            sge.Interval: _interval,
        }


class ImpalaCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = "impala"
    type_mapper = ImpalaType
    rewrites = (
        rewrite_sample,
        rewrite_first_to_first_value,
        rewrite_last_to_last_value,
        rewrite_empty_order_by_window,
        *SQLGlotCompiler.rewrites,
    )
    quoted = True

    def _aggregate(self, funcname: str, *args, where):
        if where is not None:
            args = tuple(self.if_(where, arg, NULL) for arg in args)

        return self.f[funcname](*args, dialect=self.dialect)

    @staticmethod
    def _minimize_spec(start, end, spec):
        # start is None means unbounded preceding
        if start is None:
            # end is None: unbounded following
            # end == 0 => current row
            # these are treated the same because for the functions where these
            # are not allowed they end up behaving the same
            #
            # I think we're not covering some cases here:
            # These will be treated the same, even though they're not
            # - window(order_by=x, rows=(None, None))  # should be equivalent to `over ()`
            # - window(order_by=x, rows=(None, 0))     # equivalent to a cumulative aggregation
            #
            # TODO(cpcloud): we need to clean up the semantics of unbounded
            # following vs current row at the API level.
            #
            if end is None or (
                isinstance(getattr(end, "value", None), ops.Literal)
                and end.value.value == 0
                and end.following
            ):
                return None
        return spec

    @singledispatchmethod
    def visit_node(self, op, **kw):
        return super().visit_node(op, **kw)

    @visit_node.register(ops.Literal)
    def visit_Literal(self, op, *, value, dtype):
        if value is None and dtype.is_binary():
            return NULL
        return super().visit_Literal(op, value=value, dtype=dtype)

    @visit_node.register(ops.CountStar)
    def visit_CountStar(self, op, *, arg, where):
        if where is not None:
            return self.f.sum(self.cast(where, op.dtype))
        return self.f.count(STAR)

    @visit_node.register(ops.CountDistinctStar)
    def visit_CountDistinctStar(self, op, *, arg, where):
        expressions = (
            sg.column(name, table=arg.alias_or_name, quoted=self.quoted)
            for name in op.arg.schema.keys()
        )
        if where is not None:
            expressions = (self.if_(where, expr, NULL) for expr in expressions)
        return self.f.count(sge.Distinct(expressions=list(expressions)))

    @visit_node.register(ops.CountDistinct)
    def visit_CountDistinct(self, op, *, arg, where):
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return self.f.count(sge.Distinct(expressions=[arg]))

    @visit_node.register(ops.Xor)
    def visit_Xor(self, op, *, left, right):
        return sg.and_(sg.or_(left, right), sg.not_(sg.and_(left, right)))

    @visit_node.register(ops.RandomScalar)
    def visit_RandomScalar(self, op):
        return self.f.rand(self.f.utc_to_unix_micros(self.f.utc_timestamp()))

    @visit_node.register(ops.DayOfWeekIndex)
    def visit_DayOfWeekIndex(self, op, *, arg):
        return self.f.pmod(self.f.dayofweek(arg) - 2, 7)

    @visit_node.register(ops.ExtractMillisecond)
    def viist_ExtractMillisecond(self, op, *, arg):
        return self.f.extract(self.v.millisecond, arg) % 1_000

    @visit_node.register(ops.ExtractMicrosecond)
    def visit_ExtractMicrosecond(self, op, *, arg):
        return self.f.extract(self.v.microsecond, arg) % 1_000_000

    @visit_node.register(ops.Degrees)
    def visit_Degrees(self, op, *, arg):
        return 180.0 * arg / self.f.pi()

    @visit_node.register(ops.Radians)
    def visit_Radians(self, op, *, arg):
        return self.f.pi() * arg / 180.0

    @visit_node.register(ops.HashBytes)
    def visit_HashBytes(self, op, *, arg, how):
        if how not in ("md5", "sha1", "sha256", "sha512"):
            raise com.UnsupportedOperationError(how)
        return self.f[how](arg)

    @visit_node.register(ops.Log)
    def visit_Log(self, op, *, arg, base):
        if base is None:
            return self.f.ln(arg)
        return self.f.log(base, arg, dialect=self.dialect)

    @visit_node.register(ops.DateFromYMD)
    def visit_DateFromYMD(self, op, *, year, month, day):
        return self.cast(
            self.f.concat(
                self.f.lpad(self.cast(year, dt.string), 4, "0"),
                "-",
                self.f.lpad(self.cast(month, dt.string), 2, "0"),
                "-",
                self.f.lpad(self.cast(day, dt.string), 2, "0"),
            ),
            dt.date,
        )

    def visit_NonNullLiteral(self, op, *, value, dtype):
        if dtype.is_date() or dtype.is_timestamp():
            # hack to return a string literal because impala doesn't support a
            # wide range of properly-typed date values
            #
            # the date implementation is very unpolished: some proper dates are
            # supported, but only within a certain range, and the
            # implementation wraps on over- and underflow
            return sge.convert(value.isoformat())
        elif dtype.is_string():
            value = (
                value
                # Escape \ first so we don't double escape other characters.
                .replace("\\", "\\\\")
                # ASCII escape sequences that are recognized in Python:
                # https://docs.python.org/3/reference/lexical_analysis.html#string-and-bytes-literals
                .replace("\a", "\\a")  # Bell
                .replace("\b", "\\b")  # Backspace
                .replace("\f", "\\f")  # Formfeed
                .replace("\n", "\\n")  # Newline / Linefeed
                .replace("\r", "\\r")  # Carriage return
                .replace("\t", "\\t")  # Tab
                .replace("\v", "\\v")  # Vertical tab
            )
            return sge.convert(value)
        elif dtype.is_decimal() and not value.is_finite():
            raise com.UnsupportedOperationError(
                f"Non-finite decimal literal values are not supported by Impala; got: {value}"
            )
        elif dtype.is_array() or dtype.is_map() or dtype.is_struct():
            raise com.UnsupportedBackendType(
                f"Impala does not support {dtype.name.lower()} literals"
            )
        elif dtype.is_uuid():
            return sge.convert(str(value))
        return None

    @visit_node.register(ops.Cast)
    def visit_Cast(self, op, *, arg, to):
        from_ = op.arg.dtype
        if from_.is_integer() and to.is_interval():
            return sge.Interval(this=sge.convert(arg), unit=to.unit.singular.upper())
        elif from_.is_temporal() and to.is_integer():
            return 1_000_000 * self.f.unix_timestamp(arg)
        return super().visit_Cast(op, arg=arg, to=to)

    @visit_node.register(ops.StartsWith)
    def visit_StartsWith(self, op, *, arg, start):
        return arg.like(self.f.concat(start, "%"))

    @visit_node.register(ops.EndsWith)
    def visit_EndsWith(self, op, *, arg, end):
        return arg.like(self.f.concat("%", end))

    @visit_node.register(ops.FindInSet)
    def visit_FindInSet(self, op, *, needle, values):
        return self.f.find_in_set(needle, self.f.concat_ws(",", *values))

    @visit_node.register(ops.ExtractProtocol)
    @visit_node.register(ops.ExtractAuthority)
    @visit_node.register(ops.ExtractUserInfo)
    @visit_node.register(ops.ExtractHost)
    @visit_node.register(ops.ExtractFile)
    @visit_node.register(ops.ExtractPath)
    def visit_ExtractUrlField(self, op, *, arg):
        return self.f.parse_url(arg, type(op).__name__[len("Extract") :].upper())

    @visit_node.register(ops.ExtractQuery)
    def visit_ExtractQuery(self, op, *, arg, key):
        return self.f.parse_url(*filter(None, (arg, "QUERY", key)))

    @visit_node.register(ops.ExtractFragment)
    def visit_ExtractFragment(self, op, *, arg):
        return self.f.parse_url(arg, "REF")

    @visit_node.register(ops.StringFind)
    def visit_StringFind(self, op, *, arg, substr, start, end):
        if start is not None:
            return self.f.locate(substr, arg, start + 1)
        return self.f.locate(substr, arg)

    @visit_node.register(ops.StringContains)
    def visit_StringContains(self, op, *, haystack, needle):
        return self.f.locate(needle, haystack) > 0

    @visit_node.register(ops.TimestampDiff)
    def visit_TimestampDiff(self, op, *, left, right):
        return self.f.unix_timestamp(left) - self.f.unix_timestamp(right)

    @visit_node.register(ops.Strftime)
    def visit_Strftime(self, op, *, arg, format_str):
        if not isinstance(op.format_str, ops.Literal):
            raise com.UnsupportedOperationError(
                "strftime format string must be a literal; "
                f"got: {type(op.format_str).__name__}"
            )
        format_str = sg.time.format_time(
            op.format_str.value, {v: k for k, v in Impala.TIME_MAPPING.items()}
        )
        return self.f.from_unixtime(
            self.f.unix_timestamp(self.cast(arg, dt.string)), format_str
        )

    @visit_node.register(ops.ExtractWeekOfYear)
    def visit_ExtractWeekOfYear(self, op, *, arg):
        return self.f.anon.weekofyear(arg)

    @visit_node.register(ops.TimestampTruncate)
    def visit_TimestampTruncate(self, op, *, arg, unit):
        units = {
            "Y": "YEAR",
            "M": "MONTH",
            "W": "WEEK",
            "D": "DAY",
            "h": "HOUR",
            "m": "MINUTE",
            "s": "SECOND",
            "ms": "MILLISECONDS",
            "us": "MICROSECONDS",
        }
        if unit.short == "Q":
            return self.f.trunc(arg, "Q")
        if (impala_unit := units.get(unit.short)) is None:
            raise com.UnsupportedOperationError(
                f"{unit!r} unit is not supported in timestamp/date truncate"
            )
        return self.f.date_trunc(impala_unit, arg)

    @visit_node.register(ops.DateTruncate)
    def visit_DateTruncate(self, op, *, arg, unit):
        if unit.short == "Q":
            return self.f.trunc(arg, "Q")
        return self.f.date_trunc(unit.name.upper(), arg)

    @visit_node.register(ops.TimestampFromUNIX)
    def visit_TimestampFromUNIX(self, op, *, arg, unit):
        arg = self.cast(util.convert_unit(arg, unit.short, "s"), dt.int32)
        return self.cast(self.f.from_unixtime(arg, "yyyy-MM-dd HH:mm:ss"), dt.timestamp)

    @visit_node.register(ops.DateAdd)
    def visit_DateAdd(self, op, *, left, right):
        return self.cast(
            super().visit_DateAdd(op, left=self.cast(left, dt.date), right=right),
            dt.date,
        )

    @visit_node.register(ops.TimestampAdd)
    def visit_TimestampAdd(self, op, *, left, right):
        if not isinstance(right, sge.Interval):
            raise com.UnsupportedOperationError(
                "right operand to timestamp add operation must be a literal"
            )

        return self.cast(
            super().visit_TimestampAdd(
                op, left=self.cast(left, dt.timestamp), right=right
            ),
            dt.timestamp,
        )

    @visit_node.register(ops.DateDiff)
    def visit_DateDiff(self, op, *, left, right):
        return self.f.anon.datediff(left, right)

    @visit_node.register(ops.Date)
    def visit_Date(self, op, *, arg):
        return self.cast(self.f.to_date(arg), dt.date)

    @visit_node.register(ops.RegexReplace)
    def visit_RegexReplace(self, op, *, arg, pattern, replacement):
        return self.f.regexp_replace(arg, pattern, replacement, dialect=self.dialect)

    @visit_node.register(ops.Round)
    def visit_Round(self, op, *, arg, digits):
        rounded = self.f.round(*filter(None, (arg, digits)))

        dtype = op.dtype
        if dtype.is_integer():
            return self.cast(rounded, dtype)
        return rounded

    @visit_node.register(ops.Sign)
    def visit_Sign(self, op, *, arg):
        sign = self.f.sign(arg)
        dtype = op.dtype
        if not dtype.is_float32():
            return self.cast(sign, dtype)
        return sign

    @visit_node.register(ops.Arbitrary)
    @visit_node.register(ops.ArgMax)
    @visit_node.register(ops.ArgMin)
    @visit_node.register(ops.ArrayCollect)
    @visit_node.register(ops.ArrayPosition)
    @visit_node.register(ops.Array)
    @visit_node.register(ops.Covariance)
    @visit_node.register(ops.DateDelta)
    @visit_node.register(ops.ExtractDayOfYear)
    @visit_node.register(ops.First)
    @visit_node.register(ops.Last)
    @visit_node.register(ops.Levenshtein)
    @visit_node.register(ops.Map)
    @visit_node.register(ops.Median)
    @visit_node.register(ops.MultiQuantile)
    @visit_node.register(ops.NthValue)
    @visit_node.register(ops.Quantile)
    @visit_node.register(ops.RegexSplit)
    @visit_node.register(ops.RowID)
    @visit_node.register(ops.StringSplit)
    @visit_node.register(ops.StructColumn)
    @visit_node.register(ops.Time)
    @visit_node.register(ops.TimeDelta)
    @visit_node.register(ops.TimestampBucket)
    @visit_node.register(ops.TimestampDelta)
    @visit_node.register(ops.Unnest)
    def visit_Undefined(self, op, **_):
        raise com.OperationNotDefinedError(type(op).__name__)


_SIMPLE_OPS = {
    ops.All: "min",
    ops.Any: "max",
    ops.ApproxMedian: "appx_median",
    ops.BaseConvert: "conv",
    ops.BitwiseAnd: "bitand",
    ops.BitwiseLeftShift: "shiftleft",
    ops.BitwiseNot: "bitnot",
    ops.BitwiseOr: "bitor",
    ops.BitwiseRightShift: "shiftright",
    ops.BitwiseXor: "bitxor",
    ops.Cot: "cot",
    ops.DayOfWeekName: "dayname",
    ops.ExtractEpochSeconds: "unix_timestamp",
    ops.Hash: "fnv_hash",
    ops.LStrip: "ltrim",
    ops.Ln: "ln",
    ops.Log10: "log10",
    ops.Log2: "log2",
    ops.RStrip: "rtrim",
    ops.Strip: "trim",
    ops.TypeOf: "typeof",
}

for _op, _name in _SIMPLE_OPS.items():
    assert isinstance(type(_op), type), type(_op)
    if issubclass(_op, ops.Reduction):

        @ImpalaCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, where, **kw):
            return self.agg[_name](*kw.values(), where=where)

    else:

        @ImpalaCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, **kw):
            return self.f[_name](*kw.values())

    setattr(ImpalaCompiler, f"visit_{_op.__name__}", _fmt)

del _op, _name, _fmt
