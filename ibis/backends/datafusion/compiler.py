from __future__ import annotations

import calendar
import math
from functools import partial, singledispatchmethod
from itertools import starmap

import sqlglot as sg
from sqlglot import exp, transforms
from sqlglot.dialects import Postgres
from sqlglot.dialects.dialect import rename_func

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sqlglot.compiler import (
    FALSE,
    NULL,
    STAR,
    SQLGlotCompiler,
    paren,
)
from ibis.backends.base.sqlglot.datatypes import DataFusionType
from ibis.common.temporal import IntervalUnit, TimestampUnit
from ibis.expr.operations.udf import InputType
from ibis.expr.rewrites import rewrite_sample
from ibis.formats.pyarrow import PyArrowType


class DataFusion(Postgres):
    class Generator(Postgres.Generator):
        TRANSFORMS = Postgres.Generator.TRANSFORMS.copy() | {
            exp.Select: transforms.preprocess([transforms.eliminate_qualify]),
            exp.Pow: rename_func("pow"),
            exp.IsNan: rename_func("isnan"),
            exp.CurrentTimestamp: rename_func("now"),
            exp.Split: rename_func("string_to_array"),
            exp.Array: rename_func("make_array"),
            exp.ArrayContains: rename_func("array_has"),
            exp.ArraySize: rename_func("array_length"),
        }


class DataFusionCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = "datafusion"
    type_mapper = DataFusionType
    quoted = True
    rewrites = (rewrite_sample, *SQLGlotCompiler.rewrites)

    def _aggregate(self, funcname: str, *args, where):
        expr = self.f[funcname](*args)
        if where is not None:
            return sg.exp.Filter(this=expr, expression=sg.exp.Where(this=where))
        return expr

    def _to_timestamp(self, value, target_dtype, literal=False):
        tz = (
            f'Some("{timezone}")'
            if (timezone := target_dtype.timezone) is not None
            else "None"
        )
        unit = (
            target_dtype.unit.name.capitalize()
            if target_dtype.scale is not None
            else "Microsecond"
        )
        str_value = str(value) if literal else value
        return self.f.arrow_cast(str_value, f"Timestamp({unit}, {tz})")

    @singledispatchmethod
    def visit_node(self, op, **kw):
        return super().visit_node(op, **kw)

    @visit_node.register(ops.Literal)
    def visit_Literal(self, op, *, value, dtype, **kw):
        if value is None:
            return super().visit_node(op, value=value, dtype=dtype, **kw)
        elif dtype.is_decimal():
            return self.cast(
                sg.exp.convert(str(value)),
                dt.Decimal(precision=dtype.precision or 38, scale=dtype.scale or 9),
            )
        elif dtype.is_numeric():
            if isinstance(value, float):
                if math.isinf(value):
                    return self.cast("+Inf", dt.float64)
                elif math.isnan(value):
                    return self.cast("NaN", dt.float64)
            return sg.exp.convert(value)
        elif dtype.is_interval():
            if dtype.unit.short in ("ms", "us", "ns"):
                raise com.UnsupportedOperationError(
                    "DataFusion doesn't support subsecond interval resolutions"
                )

            return sg.exp.Interval(
                this=sg.exp.convert(str(value)),
                unit=sg.exp.var(dtype.unit.plural.lower()),
            )
        elif dtype.is_timestamp():
            return self._to_timestamp(value, dtype, literal=True)
        elif dtype.is_date():
            return self.f.date_trunc("day", value.isoformat())
        elif dtype.is_binary():
            return sg.exp.HexString(this=value.hex())
        else:
            return super().visit_node(op, value=value, dtype=dtype, **kw)

    @visit_node.register(ops.Cast)
    def visit_Cast(self, op, *, arg, to):
        if to.is_interval():
            unit = to.unit.name.lower()
            return sg.cast(
                self.f.concat(self.cast(arg, dt.string), f" {unit}"), "interval"
            )
        if to.is_timestamp():
            return self._to_timestamp(arg, to)
        if to.is_decimal():
            return self.f.arrow_cast(arg, f"{PyArrowType.from_ibis(to)}".capitalize())
        return self.cast(arg, to)

    @visit_node.register(ops.Substring)
    def visit_Substring(self, op, *, arg, start, length):
        start = self.if_(start < 0, self.f.length(arg) + start + 1, start + 1)
        if length is not None:
            return self.f.substr(arg, start, length)
        return self.f.substr(arg, start)

    @visit_node.register(ops.Variance)
    def visit_Variance(self, op, *, arg, how, where):
        if how == "sample":
            return self.agg.var_samp(arg, where=where)
        elif how == "pop":
            return self.agg.var_pop(arg, where=where)
        else:
            raise ValueError(f"Unrecognized how value: {how}")

    @visit_node.register(ops.StandardDev)
    def visit_StandardDev(self, op, *, arg, how, where):
        if how == "sample":
            return self.agg.stddev_samp(arg, where=where)
        elif how == "pop":
            return self.agg.stddev_pop(arg, where=where)
        else:
            raise ValueError(f"Unrecognized how value: {how}")

    @visit_node.register(ops.ScalarUDF)
    def visit_ScalarUDF(self, op, **kw):
        input_type = op.__input_type__
        if input_type in (InputType.PYARROW, InputType.BUILTIN):
            return self.f[op.__full_name__](*kw.values())
        else:
            raise NotImplementedError(
                f"DataFusion only supports PyArrow UDFs: got a {input_type.name.lower()} UDF"
            )

    @visit_node.register(ops.ElementWiseVectorizedUDF)
    def visit_ElementWiseVectorizedUDF(
        self, op, *, func, func_args, input_type, return_type
    ):
        return self.f[func.__name__](*func_args)

    @visit_node.register(ops.RegexExtract)
    def visit_RegexExtract(self, op, *, arg, pattern, index):
        if not isinstance(op.index, ops.Literal):
            raise ValueError(
                "re_extract `index` expressions must be literals. "
                "Arbitrary expressions are not supported in the DataFusion backend"
            )
        return self.f.regexp_match(arg, self.f.concat("(", pattern, ")"))[index]

    @visit_node.register(ops.StringFind)
    def visit_StringFind(self, op, *, arg, substr, start, end):
        if end is not None:
            raise NotImplementedError("`end` not yet implemented")

        if start is not None:
            pos = self.f.strpos(self.f.substr(arg, start + 1), substr)
            return self.f.coalesce(self.f.nullif(pos + start, start), 0)

        return self.f.strpos(arg, substr)

    @visit_node.register(ops.RegexSearch)
    def visit_RegexSearch(self, op, *, arg, pattern):
        return self.if_(
            sg.or_(arg.is_(NULL), pattern.is_(NULL)),
            NULL,
            self.f.coalesce(
                # null is returned for non-matching patterns, so coalesce to false
                # because that is the desired behavior for ops.RegexSearch
                self.f.array_length(self.f.regexp_match(arg, pattern)) > 0,
                FALSE,
            ),
        )

    @visit_node.register(ops.StringContains)
    def visit_StringContains(self, op, *, haystack, needle):
        return self.f.strpos(haystack, needle) > sg.exp.convert(0)

    @visit_node.register(ops.ExtractFragment)
    def visit_ExtractFragment(self, op, *, arg):
        return self.f.extract_url_field(arg, "fragment")

    @visit_node.register(ops.ExtractProtocol)
    def visit_ExtractProtocol(self, op, *, arg):
        return self.f.extract_url_field(arg, "scheme")

    @visit_node.register(ops.ExtractAuthority)
    def visit_ExtractAuthority(self, op, *, arg):
        return self.f.extract_url_field(arg, "netloc")

    @visit_node.register(ops.ExtractPath)
    def visit_ExtractPath(self, op, *, arg):
        return self.f.extract_url_field(arg, "path")

    @visit_node.register(ops.ExtractHost)
    def visit_ExtractHost(self, op, *, arg):
        return self.f.extract_url_field(arg, "hostname")

    @visit_node.register(ops.ExtractQuery)
    def visit_ExtractQuery(self, op, *, arg, key):
        if key is not None:
            return self.f.extract_query_param(arg, key)
        return self.f.extract_query(arg)

    @visit_node.register(ops.ExtractUserInfo)
    def visit_ExtractUserInfo(self, op, *, arg):
        return self.f.extract_user_info(arg)

    @visit_node.register(ops.ExtractYear)
    @visit_node.register(ops.ExtractMonth)
    @visit_node.register(ops.ExtractQuarter)
    @visit_node.register(ops.ExtractDay)
    def visit_ExtractYearMonthQuarterDay(self, op, *, arg):
        skip = len("Extract")
        part = type(op).__name__[skip:].lower()
        return self.f.date_part(part, arg)

    @visit_node.register(ops.ExtractDayOfYear)
    def visit_ExtractDayOfYear(self, op, *, arg):
        return self.f.date_part("doy", arg)

    @visit_node.register(ops.DayOfWeekIndex)
    def visit_DayOfWeekIndex(self, op, *, arg):
        return (self.f.date_part("dow", arg) + 6) % 7

    @visit_node.register(ops.DayOfWeekName)
    def visit_DayOfWeekName(self, op, *, arg):
        return sg.exp.Case(
            this=paren(self.f.date_part("dow", arg) + 6) % 7,
            ifs=list(starmap(self.if_, enumerate(calendar.day_name))),
        )

    @visit_node.register(ops.Date)
    def visit_Date(self, op, *, arg):
        return self.f.date_trunc("day", arg)

    @visit_node.register(ops.ExtractWeekOfYear)
    def visit_ExtractWeekOfYear(self, op, *, arg):
        return self.f.date_part("week", arg)

    @visit_node.register(ops.TimestampTruncate)
    def visit_TimestampTruncate(self, op, *, arg, unit):
        if unit in (
            IntervalUnit.MILLISECOND,
            IntervalUnit.MICROSECOND,
            IntervalUnit.NANOSECOND,
        ):
            raise com.UnsupportedOperationError(
                f"The function is not defined for time unit {unit}"
            )

        return self.f.date_trunc(unit.name.lower(), arg)

    @visit_node.register(ops.ExtractEpochSeconds)
    def visit_ExtractEpochSeconds(self, op, *, arg):
        if op.arg.dtype.is_date():
            return self.f.extract_epoch_seconds_date(arg)
        elif op.arg.dtype.is_timestamp():
            return self.f.extract_epoch_seconds_timestamp(arg)
        else:
            raise com.OperationNotDefinedError(
                f"The function is not defined for {op.arg.dtype}"
            )

    @visit_node.register(ops.ExtractMinute)
    def visit_ExtractMinute(self, op, *, arg):
        if op.arg.dtype.is_date():
            return self.f.date_part("minute", arg)
        elif op.arg.dtype.is_time():
            return self.f.extract_minute_time(arg)
        elif op.arg.dtype.is_timestamp():
            return self.f.extract_minute_timestamp(arg)
        else:
            raise com.OperationNotDefinedError(
                f"The function is not defined for {op.arg.dtype}"
            )

    @visit_node.register(ops.ExtractMillisecond)
    def visit_ExtractMillisecond(self, op, *, arg):
        if op.arg.dtype.is_time():
            return self.f.extract_millisecond_time(arg)
        elif op.arg.dtype.is_timestamp():
            return self.f.extract_millisecond_timestamp(arg)
        else:
            raise com.OperationNotDefinedError(
                f"The function is not defined for {op.arg.dtype}"
            )

    @visit_node.register(ops.ExtractHour)
    def visit_ExtractHour(self, op, *, arg):
        if op.arg.dtype.is_date() or op.arg.dtype.is_timestamp():
            return self.f.date_part("hour", arg)
        elif op.arg.dtype.is_time():
            return self.f.extract_hour_time(arg)
        else:
            raise com.OperationNotDefinedError(
                f"The function is not defined for {op.arg.dtype}"
            )

    @visit_node.register(ops.ExtractSecond)
    def visit_ExtractSecond(self, op, *, arg):
        if op.arg.dtype.is_date() or op.arg.dtype.is_timestamp():
            return self.f.extract_second_timestamp(arg)
        elif op.arg.dtype.is_time():
            return self.f.extract_second_time(arg)
        else:
            raise com.OperationNotDefinedError(
                f"The function is not defined for {op.arg.dtype}"
            )

    @visit_node.register(ops.ArrayRepeat)
    def visit_ArrayRepeat(self, op, *, arg, times):
        return self.f.flatten(self.f.array_repeat(arg, times))

    @visit_node.register(ops.ArrayPosition)
    def visit_ArrayPosition(self, op, *, arg, other):
        return self.f.coalesce(self.f.array_position(arg, other), 0)

    @visit_node.register(ops.Covariance)
    def visit_Covariance(self, op, *, left, right, how, where):
        x = op.left
        if x.dtype.is_boolean():
            left = self.cast(left, dt.float64)

        y = op.right
        if y.dtype.is_boolean():
            right = self.cast(right, dt.float64)

        if how == "sample":
            return self.agg.covar_samp(left, right, where=where)
        elif how == "pop":
            return self.agg.covar_pop(left, right, where=where)
        else:
            raise ValueError(f"Unrecognized how = `{how}` value")

    @visit_node.register(ops.Correlation)
    def visit_Correlation(self, op, *, left, right, where, how):
        x = op.left
        if x.dtype.is_boolean():
            left = self.cast(left, dt.float64)

        y = op.right
        if y.dtype.is_boolean():
            right = self.cast(right, dt.float64)

        return self.agg.corr(left, right, where=where)

    @visit_node.register(ops.IsNan)
    def visit_IsNan(self, op, *, arg):
        return sg.and_(arg.is_(sg.not_(NULL)), self.f.isnan(arg))

    @visit_node.register(ops.ArrayStringJoin)
    def visit_ArrayStringJoin(self, op, *, sep, arg):
        return self.f.array_join(arg, sep)

    @visit_node.register(ops.FindInSet)
    def visit_FindInSet(self, op, *, needle, values):
        return self.f.coalesce(
            self.f.array_position(self.f.make_array(*values), needle), 0
        )

    @visit_node.register(ops.TimestampFromUNIX)
    def visit_TimestampFromUNIX(self, op, *, arg, unit):
        if unit == TimestampUnit.SECOND:
            return self.f.from_unixtime(arg)
        elif unit in (
            TimestampUnit.MILLISECOND,
            TimestampUnit.MICROSECOND,
            TimestampUnit.NANOSECOND,
        ):
            return self.f.arrow_cast(arg, f"Timestamp({unit.name.capitalize()}, None)")
        else:
            raise com.UnsupportedOperationError(f"Unsupported unit {unit}")

    @visit_node.register(ops.DateFromYMD)
    def visit_DateFromYMD(self, op, *, year, month, day):
        return self.cast(
            self.f.concat(
                self.f.lpad(self.cast(self.cast(year, dt.int64), dt.string), 4, "0"),
                "-",
                self.f.lpad(self.cast(self.cast(month, dt.int64), dt.string), 2, "0"),
                "-",
                self.f.lpad(self.cast(self.cast(day, dt.int64), dt.string), 2, "0"),
            ),
            dt.date,
        )

    @visit_node.register(ops.TimestampFromYMDHMS)
    def visit_TimestampFromYMDHMS(
        self, op, *, year, month, day, hours, minutes, seconds, **_
    ):
        return self.f.to_timestamp_micros(
            self.f.concat(
                self.f.lpad(self.cast(self.cast(year, dt.int64), dt.string), 4, "0"),
                "-",
                self.f.lpad(self.cast(self.cast(month, dt.int64), dt.string), 2, "0"),
                "-",
                self.f.lpad(self.cast(self.cast(day, dt.int64), dt.string), 2, "0"),
                "T",
                self.f.lpad(self.cast(self.cast(hours, dt.int64), dt.string), 2, "0"),
                ":",
                self.f.lpad(self.cast(self.cast(minutes, dt.int64), dt.string), 2, "0"),
                ":",
                self.f.lpad(self.cast(self.cast(seconds, dt.int64), dt.string), 2, "0"),
                ".000000Z",
            )
        )

    @visit_node.register(ops.IsInf)
    def visit_IsInf(self, op, *, arg):
        return sg.and_(sg.not_(self.f.isnan(arg)), self.f.abs(arg).eq(self.POS_INF))

    @visit_node.register(ops.ArrayIndex)
    def visit_ArrayIndex(self, op, *, arg, index):
        return self.f.array_element(arg, index + self.cast(index >= 0, op.index.dtype))

    @visit_node.register(ops.Arbitrary)
    @visit_node.register(ops.ArgMax)
    @visit_node.register(ops.ArgMin)
    @visit_node.register(ops.ArrayDistinct)
    @visit_node.register(ops.ArrayFilter)
    @visit_node.register(ops.ArrayFlatten)
    @visit_node.register(ops.ArrayIntersect)
    @visit_node.register(ops.ArrayMap)
    @visit_node.register(ops.ArraySort)
    @visit_node.register(ops.ArrayUnion)
    @visit_node.register(ops.ArrayZip)
    @visit_node.register(ops.BitwiseNot)
    @visit_node.register(ops.Clip)
    @visit_node.register(ops.CountDistinctStar)
    @visit_node.register(ops.DateDelta)
    @visit_node.register(ops.Greatest)
    @visit_node.register(ops.GroupConcat)
    @visit_node.register(ops.IntervalFromInteger)
    @visit_node.register(ops.Least)
    @visit_node.register(ops.MultiQuantile)
    @visit_node.register(ops.Quantile)
    @visit_node.register(ops.RowID)
    @visit_node.register(ops.Strftime)
    @visit_node.register(ops.TimeDelta)
    @visit_node.register(ops.TimestampBucket)
    @visit_node.register(ops.TimestampDelta)
    @visit_node.register(ops.TimestampNow)
    @visit_node.register(ops.TypeOf)
    @visit_node.register(ops.Unnest)
    @visit_node.register(ops.EndsWith)
    @visit_node.register(ops.StringToTimestamp)
    @visit_node.register(ops.Levenshtein)
    def visit_Undefined(self, op, **_):
        raise com.OperationNotDefinedError(type(op).__name__)

    @visit_node.register(ops.Aggregate)
    def visit_Aggregate(self, op, *, parent, groups, metrics):
        """Support `GROUP BY` expressions in `SELECT` since DataFusion does not."""
        quoted = self.quoted
        metrics = tuple(starmap(self._dedup_name, metrics.items()))

        if groups:
            # datafusion doesn't support count distinct aggregations alongside
            # computed grouping keys so create a projection of the key and all
            # existing columns first, followed by the usual group by
            #
            # analogous to a user calling mutate -> group_by
            cols = list(
                map(
                    partial(
                        sg.column,
                        table=sg.to_identifier(parent.alias, quoted=quoted),
                        quoted=quoted,
                    ),
                    # can't use set subtraction here since the schema keys'
                    # order matters and set subtraction doesn't preserve order
                    (k for k in op.parent.schema.keys() if k not in groups),
                )
            )
            table = (
                sg.select(*cols, *starmap(self._dedup_name, groups.items()))
                .from_(parent)
                .subquery(parent.alias)
            )

            # datafusion lower cases all column names internally unless quoted so
            # quoted=True is required here for correctness
            by_names_quoted = tuple(
                sg.column(key, table=getattr(value, "table", None), quoted=True)
                for key, value in groups.items()
            )
            selections = by_names_quoted + metrics
        else:
            selections = metrics or (STAR,)
            table = parent

        sel = sg.select(*selections).from_(table)

        if groups:
            sel = sel.group_by(*by_names_quoted)

        return sel


_SIMPLE_OPS = {
    ops.ApproxMedian: "approx_median",
    ops.ArrayRemove: "array_remove_all",
    ops.BitAnd: "bit_and",
    ops.BitOr: "bit_or",
    ops.BitXor: "bit_xor",
    ops.Cot: "cot",
    ops.ExtractMicrosecond: "extract_microsecond",
    ops.First: "first_value",
    ops.LPad: "lpad",
    ops.Last: "last_value",
    ops.Median: "median",
    ops.RPad: "rpad",
    ops.StringAscii: "ascii",
    ops.StringLength: "character_length",
    ops.RegexSplit: "regex_split",
}

for _op, _name in _SIMPLE_OPS.items():
    assert isinstance(type(_op), type), type(_op)
    if issubclass(_op, ops.Reduction):

        @DataFusionCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, where, **kw):
            return self.agg[_name](*kw.values(), where=where)

    else:

        @DataFusionCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, **kw):
            return self.f[_name](*kw.values())

    setattr(DataFusionCompiler, f"visit_{_op.__name__}", _fmt)
