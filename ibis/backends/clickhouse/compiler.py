from __future__ import annotations

import calendar
import math
from functools import singledispatchmethod

import sqlglot as sg

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sqlglot import NULL, paren
from ibis.backends.base.sqlglot.compiler import SQLGlotCompiler
from ibis.backends.base.sqlglot.datatypes import ClickHouseType
from ibis.common.temporal import IntervalUnit, TimestampUnit
from ibis.expr.operations.udf import InputType
from ibis.expr.rewrites import rewrite_sample
from ibis.formats.pyarrow import PyArrowType


class ClickHouseCompiler(SQLGlotCompiler):
    dialect = "clickhouse"
    type_mapper = ClickHouseType
    rewrites = (*SQLGlotCompiler.rewrites, rewrite_sample)

    def _aggregate(self, funcname: str, *args, where):
        has_filter = where is not None
        func = self.f[funcname + "If" * has_filter]
        args += (where,) * has_filter
        return func(*args)

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
        if value is None and dtype.nullable:
            if dtype.is_null():
                return NULL
            return self.cast(NULL, dtype)
        elif dtype.is_boolean():
            return sg.exp.Boolean(this=bool(value))
        elif dtype.is_inet() or dtype.is_macaddr():
            # treat inet as strings for now
            return sg.exp.convert(str(value))
        elif dtype.is_decimal():
            return self.cast(
                sg.exp.convert(str(value)),
                dt.Decimal(precision=dtype.precision or 38, scale=dtype.scale or 9),
            )
        elif dtype.is_string():
            return sg.exp.convert(value)
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
        elif dtype.is_time():
            return self.cast(str(value), dt.time())
        elif dtype.is_array():
            vtype = dtype.value_type
            values = [
                self.visit_Literal(
                    ops.Literal(v, dtype=vtype), value=v, dtype=vtype, **kw
                )
                for v in value
            ]
            return self.f.array(*values)
        elif dtype.is_map():
            vtype = dtype.value_type
            keys = []
            values = []

            for k, v in value.items():
                keys.append(sg.exp.convert(k))
                values.append(
                    self.visit_Literal(
                        ops.Literal(v, dtype=vtype), value=v, dtype=vtype, **kw
                    )
                )

            return self.f.map(self.f.array(*keys), self.f.array(*values))
        elif dtype.is_struct():
            fields = [
                self.visit_Literal(
                    ops.Literal(v, dtype=ftype), value=v, dtype=ftype, **kw
                )
                for ftype, v in zip(dtype.types, value.values())
            ]
            return self.cast(sg.exp.Struct.from_arg_list(fields), dtype)
        elif dtype.is_binary():
            return sg.exp.HexString(this=value.hex())
        else:
            raise NotImplementedError(f"Unsupported type: {dtype!r}")

    @visit_node.register(ops.Cast)
    def visit_Cast(self, op, *, arg, to, **_):
        if to.is_interval():
            unit_name = to.unit.name.lower()
            return sg.cast(
                self.f.concat(self.cast(arg, dt.string), f" {unit_name}"), "interval"
            )
        if to.is_timestamp():
            return self._to_timestamp(arg, to)
        if to.is_decimal():
            return self.f.arrow_cast(arg, f"{PyArrowType.from_ibis(to)}".capitalize())
        return self.cast(arg, to)

    @visit_node.register(ops.Not)
    def visit_Not(self, op, *, arg, **_):
        if isinstance(arg, sg.exp.Filter):
            return sg.exp.Filter(
                this=self._de_morgan_law(arg.this), expression=arg.expression
            )  # transform the not expression using _de_morgan_law
        return sg.not_(paren(arg))

    @staticmethod
    def _de_morgan_law(logical_op: sg.exp.Expression):
        if isinstance(logical_op, sg.exp.LogicalAnd):
            return sg.exp.LogicalOr(this=sg.not_(paren(logical_op.this)))
        if isinstance(logical_op, sg.exp.LogicalOr):
            return sg.exp.LogicalAnd(this=sg.not_(paren(logical_op.this)))
        return None

    @visit_node.register(ops.Ceil)
    @visit_node.register(ops.Floor)
    def visit_CeilFloor(self, op, *, arg, **_):
        return self.cast(self.f[type(op).__name__.lower()](arg), dt.int64)

    @visit_node.register(ops.Substring)
    def visit_Substring(self, op, *, arg, start, length, **_):
        start = self.if_(start < 0, self.f.length(arg) + start + 1, start + 1)
        if length is not None:
            return self.f.substr(arg, start, length)
        return self.f.substr(arg, start)

    @visit_node.register(ops.Divide)
    def visit_Divide(self, op, *, left, right, **_):
        return self.cast(left, dt.float64) / self.cast(right, dt.float64)

    @visit_node.register(ops.FloorDivide)
    def visit_FloorDivide(self, op, *, left, right, **_):
        return self.f.floor(left / right)

    @visit_node.register(ops.Variance)
    def visit_Variance(self, op, *, arg, how, where, **_):
        if how == "sample":
            return self.agg.var_samp(arg, where=where)
        elif how == "pop":
            return self.agg.var_pop(arg, where=where)
        else:
            raise ValueError(f"Unrecognized how value: {how}")

    @visit_node.register(ops.StandardDev)
    def visit_StandardDev(self, op, *, arg, how, where, **_):
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
    def visit_ElementWiseVectorizedUDF(self, op, *, func, func_args, **_):
        return self.f[func.__name__](*func_args)

    @visit_node.register(ops.StringConcat)
    def visit_StringConcat(self, op, *, arg, **_):
        return self.f.concat(*arg)

    @visit_node.register(ops.RegexExtract)
    def visit_RegexExtract(self, op, *, arg, pattern, index, **_):
        if not isinstance(op.index, ops.Literal):
            raise ValueError(
                "re_extract `index` expressions must be literals. "
                "Arbitrary expressions are not supported in the DataFusion backend"
            )
        return self.f.regexp_match(arg, self.f.concat("(", pattern, ")"))[index]

    # @visit_node.register(ops.RegexReplace)
    # def regex_replace(self, op, *, arg, pattern, replacement, **_):
    #     return self.f.regexp_replace(arg, pattern, replacement, sg.exp.convert("g"))

    @visit_node.register(ops.StringFind)
    def visit_StringFind(self, op, *, arg, substr, start, end, **_):
        if end is not None:
            raise NotImplementedError("`end` not yet implemented")

        if start is not None:
            pos = self.f.strpos(self.f.substr(arg, start + 1), substr)
            return self.f.coalesce(self.f.nullif(pos + start, start), 0)

        return self.f.strpos(arg, substr)

    @visit_node.register(ops.RegexSearch)
    def visit_RegexSearch(self, op, *, arg, pattern, **_):
        return self.f.array_length(self.f.regexp_match(arg, pattern)) > 0

    @visit_node.register(ops.StringContains)
    def visit_StringContains(self, op, *, haystack, needle, **_):
        return self.f.strpos(haystack, needle) > sg.exp.convert(0)

    @visit_node.register(ops.StringJoin)
    def visit_StringJoin(self, op, *, sep, arg, **_):
        if not isinstance(op.sep, ops.Literal):
            raise ValueError(
                "join `sep` expressions must be literals. "
                "Arbitrary expressions are not supported in the DataFusion backend"
            )

        return self.f.concat_ws(sep, *arg)

    @visit_node.register(ops.ExtractFragment)
    def visit_ExtractFragment(self, op, *, arg, **_):
        return self.f.extract_url_field(arg, "fragment")

    @visit_node.register(ops.ExtractProtocol)
    def visit_ExtractProtocol(self, op, *, arg, **_):
        return self.f.extract_url_field(arg, "scheme")

    @visit_node.register(ops.ExtractAuthority)
    def visit_ExtractAuthority(self, op, *, arg, **_):
        return self.f.extract_url_field(arg, "netloc")

    @visit_node.register(ops.ExtractPath)
    def visit_ExtractPath(self, op, *, arg, **_):
        return self.f.extract_url_field(arg, "path")

    @visit_node.register(ops.ExtractHost)
    def visit_ExtractHost(self, op, *, arg, **_):
        return self.f.extract_url_field(arg, "hostname")

    @visit_node.register(ops.ExtractQuery)
    def visit_ExtractQuery(self, op, *, arg, key, **_):
        if key is not None:
            return self.f.extract_query_param(arg, key)
        return self.f.extract_query(arg)

    @visit_node.register(ops.ExtractUserInfo)
    def visit_ExtractUserInfo(self, op, *, arg, **_):
        return self.f.extract_user_info(arg)

    @visit_node.register(ops.ExtractYear)
    @visit_node.register(ops.ExtractMonth)
    @visit_node.register(ops.ExtractQuarter)
    @visit_node.register(ops.ExtractDay)
    def visit_ExtractYearMonthQuarterDay(self, op, *, arg, **_):
        skip = len("Extract")
        part = type(op).__name__[skip:].lower()
        return self.f.date_part(part, arg)

    @visit_node.register(ops.ExtractDayOfYear)
    def visit_ExtractDayOfYear(self, op, *, arg, **_):
        return self.f.date_part("doy", arg)

    @visit_node.register(ops.DayOfWeekIndex)
    def visit_DayOfWeekIndex(self, op, *, arg, **_):
        return (self.f.date_part("dow", arg) + 6) % 7

    @visit_node.register(ops.DayOfWeekName)
    def visit_DayOfWeekName(self, op, *, arg, **_):
        return sg.exp.Case(
            this=paren((self.f.date_part("dow", arg) + 6) % 7),
            ifs=list(map(self.if_, *zip(*enumerate(calendar.day_name)))),
        )

    @visit_node.register(ops.Date)
    def visit_Date(self, op, *, arg, **_):
        return self.f.date_trunc("day", arg)

    @visit_node.register(ops.ExtractWeekOfYear)
    def visit_ExtractWeekOfYear(self, op, *, arg, **_):
        return self.f.date_part("week", arg)

    @visit_node.register(ops.TimestampTruncate)
    def visit_TimestampTruncate(self, op, *, arg, **_):
        unit = op.unit
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
    def visit_ExtractEpochSeconds(self, op, *, arg, **_):
        if op.arg.dtype.is_date():
            return self.f.extract_epoch_seconds_date(arg)
        elif op.arg.dtype.is_timestamp():
            return self.f.extract_epoch_seconds_timestamp(arg)
        else:
            raise com.OperationNotDefinedError(
                f"The function is not defined for {op.arg.dtype}"
            )

    @visit_node.register(ops.ExtractMinute)
    def visit_ExtractMinute(self, op, *, arg, **_):
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
    def visit_ExtractMillisecond(self, op, *, arg, **_):
        if op.arg.dtype.is_time():
            return self.f.extract_millisecond_time(arg)
        elif op.arg.dtype.is_timestamp():
            return self.f.extract_millisecond_timestamp(arg)
        else:
            raise com.OperationNotDefinedError(
                f"The function is not defined for {op.arg.dtype}"
            )

    @visit_node.register(ops.ExtractHour)
    def visit_ExtractHour(self, op, *, arg, **_):
        if op.arg.dtype.is_date() or op.arg.dtype.is_timestamp():
            return self.f.date_part("hour", arg)
        elif op.arg.dtype.is_time():
            return self.f.extract_hour_time(arg)
        else:
            raise com.OperationNotDefinedError(
                f"The function is not defined for {op.arg.dtype}"
            )

    @visit_node.register(ops.ExtractSecond)
    def visit_ExtractSecond(self, op, *, arg, **_):
        if op.arg.dtype.is_date() or op.arg.dtype.is_timestamp():
            return self.f.extract_second_timestamp(arg)
        elif op.arg.dtype.is_time():
            return self.f.extract_second_time(arg)
        else:
            raise com.OperationNotDefinedError(
                f"The function is not defined for {op.arg.dtype}"
            )

    @visit_node.register(ops.ArrayRepeat)
    def visit_ArrayRepeat(self, op, *, arg, times, **_):
        return self.f.flatten(self.f.array_repeat(arg, times))

    @visit_node.register(ops.ArrayPosition)
    def visit_ArrayPosition(self, op, *, arg, other, **_):
        return self.f.coalesce(self.f.array_position(arg, other), 0)

    @visit_node.register(ops.Covariance)
    def visit_Covariance(self, op, *, left, right, how, where, **_):
        x = op.left
        if x.dtype.is_boolean():
            left = self.cast(left, dt.float64)

        y = op.right
        if y.dtype.is_boolean():
            right = self.cast(right, dt.float64)

        if how == "sample":
            return self.agg["covar_samp"](left, right, where=where)
        elif how == "pop":
            return self.agg["covar_pop"](left, right, where=where)
        else:
            raise ValueError(f"Unrecognized how = `{how}` value")

    @visit_node.register(ops.Correlation)
    def visit_Correlation(self, op, *, left, right, where, **_):
        x = op.left
        if x.dtype.is_boolean():
            left = self.cast(left, dt.float64)

        y = op.right
        if y.dtype.is_boolean():
            right = self.cast(right, dt.float64)

        return self.agg["corr"](left, right, where=where)

    @visit_node.register(ops.IsNan)
    def visit_IsNan(self, op, *, arg, **_):
        return self.f.isnan(self.f.coalesce(arg, self.cast("NaN", dt.float64)))

    @visit_node.register(ops.ArrayStringJoin)
    def visit_ArrayStringJoin(self, op, *, sep, arg, **_):
        return self.f.array_join(arg, sep)

    @visit_node.register(ops.FindInSet)
    def visit_FindInSet(self, op, *, needle, values, **_):
        return self.f.coalesce(
            self.f.array_position(self.f.make_array(*values), needle), 0
        )

    @visit_node.register(ops.TimestampFromUNIX)
    def visit_TimestampFromUNIX(self, op, *, arg, unit, **_):
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
    def visit_DateFromYMD(self, op, *, year, month, day, **_):
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
    def visit_IsInf(self, op, *, arg, **_):
        return sg.and_(
            sg.not_(self.f.isnan(arg)),
            self.f.abs(arg).eq(self.cast(sg.exp.convert("+Inf"), dt.float64)),
        )

    @visit_node.register(ops.ArrayIndex)
    def visit_ArrayIndex(self, op, *, arg, index, **_):
        return self.f.array_element(arg, index + self.cast(index >= 0, op.index.dtype))


_SIMPLE_OPS = {
    ops.Reverse: "reverse",
    ops.Strip: "trim",
    ops.LStrip: "ltrim",
    ops.RStrip: "rtrim",
    ops.Lowercase: "lower",
    ops.Uppercase: "upper",
    ops.StringLength: "character_length",
    ops.Capitalize: "initcap",
    ops.Repeat: "repeat",
    ops.LPad: "lpad",
    ops.RPad: "rpad",
    ops.Count: "count",
    ops.Median: "median",
    ops.ApproxMedian: "approx_median",
    ops.Power: "power",
    ops.RandomScalar: "random",
    ops.Translate: "translate",
    ops.StringAscii: "ascii",
    ops.StartsWith: "starts_with",
    ops.StrRight: "right",
    ops.StringReplace: "replace",
    ops.Sign: "sign",
    ops.ExtractMicrosecond: "extract_microsecond",
    ops.BitOr: "bit_or",
    ops.BitXor: "bit_xor",
    ops.BitAnd: "bit_and",
    ops.ApproxCountDistinct: "approx_distinct",
    ops.First: "first_value",
    ops.Last: "last_value",
    ops.Cot: "cot",
    ops.ArrayLength: "array_length",
    ops.ArrayRemove: "array_remove_all",
    ops.First: "first_value",
    ops.Last: "last_value",
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


_NOT_IMPLEMENTED_OPS = {
    ops.Arbitrary,
    ops.ArgMax,
    ops.ArgMin,
    ops.ArrayDistinct,
    ops.ArrayFilter,
    ops.ArrayFlatten,
    ops.ArrayIntersect,
    ops.ArrayMap,
    ops.ArrayUnion,
    ops.ArrayZip,
    ops.BitwiseNot,
    ops.Clip,
    ops.CountDistinctStar,
    ops.DateDelta,
    ops.Greatest,
    ops.GroupConcat,
    ops.IntervalFromInteger,
    ops.Least,
    ops.MultiQuantile,
    ops.Quantile,
    ops.RowID,
    ops.Strftime,
    ops.TimeDelta,
    ops.TimestampBucket,
    ops.TimestampDelta,
    ops.TimestampNow,
    ops.TypeOf,
    ops.Unnest,
    ops.EndsWith,
}

for _op in _NOT_IMPLEMENTED_OPS:

    @DataFusionCompiler.visit_node.register(_op)
    def _fmt(self, op, **_):
        raise com.OperationNotDefinedError(type(op).__name__)

    setattr(DataFusionCompiler, f"visit_{_op.__name__}", _fmt)


del _op, _fmt
