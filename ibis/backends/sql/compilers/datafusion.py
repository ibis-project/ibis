from __future__ import annotations

import calendar
import math
from functools import partial, reduce
from itertools import starmap

import sqlglot as sg
import sqlglot.expressions as sge
from packaging.version import parse as vparse

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis import util
from ibis.backends.sql.compilers.base import FALSE, NULL, STAR, AggGen, SQLGlotCompiler
from ibis.backends.sql.datatypes import DataFusionType
from ibis.backends.sql.dialects import DataFusion
from ibis.backends.sql.rewrites import split_select_distinct_with_order_by
from ibis.common.temporal import IntervalUnit, TimestampUnit
from ibis.expr.operations.udf import InputType


class DataFusionCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = DataFusion
    type_mapper = DataFusionType

    agg = AggGen(supports_filter=True, supports_order_by=True)

    post_rewrites = (split_select_distinct_with_order_by,)

    UNSUPPORTED_OPS = (
        ops.ArrayFilter,
        ops.ArrayMap,
        ops.ArrayZip,
        ops.CountDistinctStar,
        ops.DateDelta,
        ops.RowID,
        ops.Strftime,
        ops.TimeDelta,
        ops.TimestampBucket,
        ops.TimestampDelta,
        ops.TypeOf,
        ops.StringToDate,
        ops.StringToTimestamp,
        ops.StringToTime,
        # in theory possible via
        # https://github.com/datafusion-contrib/datafusion-functions-extra but
        # not clear how to use that library from Python
        ops.Kurtosis,
    )

    SIMPLE_OPS = {
        ops.ApproxMedian: "approx_median",
        ops.ArrayDistinct: "array_distinct",
        ops.ArrayRemove: "array_remove_all",
        ops.BitAnd: "bit_and",
        ops.BitOr: "bit_or",
        ops.BitXor: "bit_xor",
        ops.Cot: "cot",
        ops.ExtractMicrosecond: "extract_microsecond",
        ops.Median: "median",
        ops.StringLength: "character_length",
        ops.RegexSplit: "regex_split",
        ops.EndsWith: "ends_with",
        ops.ArrayIntersect: "array_intersect",
        ops.ArrayUnion: "array_union",
        ops.MapKeys: "map_keys",
        ops.MapValues: "map_values",
    }

    def _to_timestamp(self, value, target_dtype: dt.Timestamp, from_: dt.DataType):
        tz = (
            f'Some("{timezone}")'
            if (timezone := target_dtype.timezone) is not None
            else "None"
        )
        if target_dtype.scale is None:
            if from_.is_integer():
                unit = TimestampUnit.SECOND
            else:
                unit = TimestampUnit.MICROSECOND
        else:
            unit = target_dtype.unit
        if from_.is_numeric():
            if unit == TimestampUnit.NANOSECOND:
                raise com.UnsupportedOperationError(
                    "Ibis has not implemented casting numeric to timestamp with nanosecond precision"
                )
            scale = TimestampUnit.to_scale(unit)
            value *= 10**scale
        return self.f.arrow_cast(value, f"Timestamp({unit.name.capitalize()}, {tz})")

    def visit_NonNullLiteral(self, op: ops.Literal, *, value, dtype: dt.DataType):
        if dtype.is_decimal():
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
            return self._to_timestamp(str(value), dtype, from_=dt.infer(value))
        elif dtype.is_date():
            return self.f.date_trunc("day", value.isoformat())
        elif dtype.is_binary():
            return sg.exp.HexString(this=value.hex())
        elif dtype.is_uuid():
            return sge.convert(str(value))
        elif dtype.is_struct():
            args = []
            for name, field_value in value.items():
                args.append(sge.convert(name))
                args.append(field_value)
            return self.f.named_struct(*args)
        else:
            return None

    def visit_Cast(self, op, *, arg, to):
        from_ = op.arg.dtype
        if to.is_interval():
            unit = to.unit.name.lower()
            return sg.cast(
                self.f.concat(self.cast(arg, dt.string), f" {unit}"), "interval"
            )
        if to.is_timestamp():
            return self._to_timestamp(arg, to, from_=from_)
        if to.is_decimal():
            from ibis.formats.pyarrow import PyArrowType

            return self.f.arrow_cast(arg, f"{PyArrowType.from_ibis(to)}".capitalize())
        return self.cast(arg, to)

    def visit_Arbitrary(self, op, *, arg, where):
        cond = ~arg.is_(NULL)
        if where is not None:
            cond &= where
        return self.agg.first_value(arg, where=cond)

    def visit_Variance(self, op, *, arg, how, where):
        if how == "sample":
            return self.agg.var_samp(arg, where=where)
        elif how == "pop":
            return self.agg.var_pop(arg, where=where)
        else:
            raise ValueError(f"Unrecognized how value: {how}")

    def visit_StandardDev(self, op, *, arg, how, where):
        if how == "sample":
            return self.agg.stddev_samp(arg, where=where)
        elif how == "pop":
            return self.agg.stddev_pop(arg, where=where)
        else:
            raise ValueError(f"Unrecognized how value: {how}")

    def visit_ScalarUDF(self, op, **kw):
        input_type = op.__input_type__
        if input_type in (InputType.PYARROW, InputType.BUILTIN):
            return self.f.anon[op.__func_name__](*kw.values())
        else:
            raise NotImplementedError(
                f"DataFusion only supports PyArrow UDFs: got a {input_type.name.lower()} UDF"
            )

    def visit_ElementWiseVectorizedUDF(
        self, op, *, func, func_args, input_type, return_type
    ):
        return self.f[func.__name__](*func_args)

    def visit_RegexExtract(self, op, *, arg, pattern, index):
        if not isinstance(op.index, ops.Literal):
            raise ValueError(
                "re_extract `index` expressions must be literals. "
                "Arbitrary expressions are not supported in the DataFusion backend"
            )
        return self.f.regexp_match(arg, self.f.concat("(", pattern, ")"))[index]

    def visit_StringFind(self, op, *, arg, substr, start, end):
        if end is not None:
            raise NotImplementedError("`end` not yet implemented")

        if start is not None:
            pos = self.f.strpos(self.f.substr(arg, start + 1), substr)
            return self.f.coalesce(self.f.nullif(pos + start, start), 0)

        return self.f.strpos(arg, substr)

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

    def visit_StringContains(self, op, *, haystack, needle):
        return self.f.strpos(haystack, needle) > sg.exp.convert(0)

    def visit_LPad(self, op, *, arg, length, pad):
        return self.if_(
            length <= self.f.length(arg),
            arg,
            self.f.concat(self.f.repeat(pad, length - self.f.length(arg)), arg),
        )

    def visit_RPad(self, op, *, arg, length, pad):
        return self.if_(
            length <= self.f.length(arg),
            arg,
            self.f.concat(arg, self.f.repeat(pad, length - self.f.length(arg))),
        )

    def visit_ExtractFragment(self, op, *, arg):
        return self.f.extract_url_field(arg, "fragment")

    def visit_ExtractProtocol(self, op, *, arg):
        return self.f.extract_url_field(arg, "scheme")

    def visit_ExtractAuthority(self, op, *, arg):
        return self.f.extract_url_field(arg, "netloc")

    def visit_ExtractPath(self, op, *, arg):
        return self.f.extract_url_field(arg, "path")

    def visit_ExtractHost(self, op, *, arg):
        return self.f.extract_url_field(arg, "hostname")

    def visit_ExtractQuery(self, op, *, arg, key):
        if key is not None:
            return self.f.extract_query_param(arg, key)
        return self.f.extract_query(arg)

    def visit_ExtractUserInfo(self, op, *, arg):
        return self.f.extract_user_info(arg)

    def visit_ExtractYearMonthQuarterDay(self, op, *, arg):
        skip = len("Extract")
        part = type(op).__name__[skip:].lower()
        return self.f.date_part(part, arg)

    visit_ExtractYear = visit_ExtractMonth = visit_ExtractQuarter = visit_ExtractDay = (
        visit_ExtractYearMonthQuarterDay
    )

    def visit_ExtractDayOfYear(self, op, *, arg):
        return self.f.date_part("doy", arg)

    def visit_DayOfWeekIndex(self, op, *, arg):
        return (self.f.date_part("dow", arg) + 6) % 7

    def visit_DayOfWeekName(self, op, *, arg):
        return sg.exp.Case(
            this=sge.paren(self.f.date_part("dow", arg) + 6, copy=False) % 7,
            ifs=list(starmap(self.if_, enumerate(calendar.day_name))),
        )

    def visit_Date(self, op, *, arg):
        return self.f.date_trunc("day", arg)

    def visit_ExtractWeekOfYear(self, op, *, arg):
        return self.f.date_part("week", arg)

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

    def visit_ExtractEpochSeconds(self, op, *, arg):
        if op.arg.dtype.is_date():
            return self.f.extract_epoch_seconds_date(arg)
        elif op.arg.dtype.is_timestamp():
            return self.f.extract_epoch_seconds_timestamp(arg)
        else:
            raise com.OperationNotDefinedError(
                f"The function is not defined for {op.arg.dtype}"
            )

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

    def visit_ExtractMillisecond(self, op, *, arg):
        if op.arg.dtype.is_time():
            return self.f.extract_millisecond_time(arg)
        elif op.arg.dtype.is_timestamp():
            return self.f.extract_millisecond_timestamp(arg)
        else:
            raise com.OperationNotDefinedError(
                f"The function is not defined for {op.arg.dtype}"
            )

    def visit_ExtractHour(self, op, *, arg):
        if op.arg.dtype.is_date() or op.arg.dtype.is_timestamp():
            return self.f.date_part("hour", arg)
        elif op.arg.dtype.is_time():
            return self.f.extract_hour_time(arg)
        else:
            raise com.OperationNotDefinedError(
                f"The function is not defined for {op.arg.dtype}"
            )

    def visit_ExtractSecond(self, op, *, arg):
        if op.arg.dtype.is_date() or op.arg.dtype.is_timestamp():
            return self.f.extract_second_timestamp(arg)
        elif op.arg.dtype.is_time():
            return self.f.extract_second_time(arg)
        else:
            raise com.OperationNotDefinedError(
                f"The function is not defined for {op.arg.dtype}"
            )

    def visit_ArrayRepeat(self, op, *, arg, times):
        return self.f.flatten(self.f.array_repeat(arg, times))

    def visit_ArrayPosition(self, op, *, arg, other):
        return self.f.coalesce(self.f.array_position(arg, other), 0)

    def visit_ArrayCollect(self, op, *, arg, where, order_by, include_null, distinct):
        if distinct:
            raise com.UnsupportedOperationError(
                "`collect` with `distinct=True` is not supported"
            )
        if not include_null:
            cond = arg.is_(sg.not_(NULL, copy=False))
            where = cond if where is None else sge.And(this=cond, expression=where)
        return self.agg.array_agg(arg, where=where, order_by=order_by)

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

    def visit_Correlation(self, op, *, left, right, where, how):
        x = op.left
        if x.dtype.is_boolean():
            left = self.cast(left, dt.float64)

        y = op.right
        if y.dtype.is_boolean():
            right = self.cast(right, dt.float64)

        return self.agg.corr(left, right, where=where)

    def visit_IsNan(self, op, *, arg):
        return sg.and_(arg.is_(sg.not_(NULL)), self.f.isnan(arg))

    def visit_ArrayStringJoin(self, op, *, sep, arg):
        return self.if_(self.f.array_length(arg) > 0, self.f.array_join(arg, sep), NULL)

    def visit_FindInSet(self, op, *, needle, values):
        return self.f.coalesce(
            self.f.array_position(self.f.make_array(*values), needle), 0
        )

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

    def visit_IsInf(self, op, *, arg):
        return sg.and_(sg.not_(self.f.isnan(arg)), self.f.abs(arg).eq(self.POS_INF))

    def visit_ArrayIndex(self, op, *, arg, index):
        return self.f.array_element(arg, index + self.cast(index >= 0, op.index.dtype))

    def visit_StringConcat(self, op, *, arg):
        any_args_null = (a.is_(NULL) for a in arg)
        return self.if_(
            sg.or_(*any_args_null), self.cast(NULL, dt.string), self.f.concat(*arg)
        )

    def visit_First(self, op, *, arg, where, order_by, include_null):
        if not include_null:
            cond = arg.is_(sg.not_(NULL, copy=False))
            where = cond if where is None else sge.And(this=cond, expression=where)
        return self.agg.first_value(arg, where=where, order_by=order_by)

    def visit_Last(self, op, *, arg, where, order_by, include_null):
        if not include_null:
            cond = arg.is_(sg.not_(NULL, copy=False))
            where = cond if where is None else sge.And(this=cond, expression=where)
        return self.agg.last_value(arg, where=where, order_by=order_by)

    def visit_ArgMin(self, op, *, arg, key, where):
        return self.agg.first_value(arg, where=where, order_by=[sge.Ordered(this=key)])

    def visit_ArgMax(self, op, *, arg, key, where):
        return self.agg.first_value(
            arg, where=where, order_by=[sge.Ordered(this=key, desc=True)]
        )

    def visit_Aggregate(self, op, *, parent, groups, metrics):
        """Support `GROUP BY` expressions in `SELECT` since DataFusion does not."""
        quoted = self.quoted
        metrics = tuple(self._cleanup_names(metrics))

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
                sg.select(*cols, *self._cleanup_names(groups))
                .from_(parent)
                .subquery(parent.alias)
            )

            # datafusion lower cases all column names internally unless quoted so
            # quoted=True is required here for correctness
            by_names_quoted = tuple(
                sg.column(key, table=getattr(value, "table", None), quoted=quoted)
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

    def visit_StructColumn(self, op, *, names, values):
        args = []
        for name, value in zip(names, values):
            args.append(sge.convert(name))
            args.append(value)
        return self.f.named_struct(*args)

    def visit_StructField(self, op, *, arg, field):
        return sge.Bracket(this=arg, expressions=[sge.convert(field)])

    def visit_GroupConcat(self, op, *, arg, sep, where, order_by):
        if order_by:
            raise com.UnsupportedOperationError(
                "DataFusion does not support order-sensitive group_concat"
            )
        return super().visit_GroupConcat(
            op, arg=arg, sep=sep, where=where, order_by=order_by
        )

    def visit_ArrayFlatten(self, op, *, arg):
        return self.if_(arg.is_(NULL), NULL, self.f.flatten(arg))

    def visit_RandomUUID(self, op):
        return self.f.anon.uuid()

    def visit_ArrayConcat(self, op, *, arg):
        return reduce(
            lambda x, y: self.if_(
                x.is_(NULL).or_(y.is_(NULL)), NULL, self.f.array_cat(x, y)
            ),
            map(partial(self.cast, to=op.dtype), arg),
        )

    def visit_MapGet(self, op, *, arg, key, default):
        if op.dtype.is_null():
            return NULL
        return self.f.coalesce(self.f.map_extract(arg, key)[1], default)

    def visit_MapContains(self, op, *, arg, key):
        return self.f.array_has(self.f.map_keys(arg), key)

    def visit_MapLength(self, op, *, arg):
        return self.f.array_length(self.f.map_keys(arg))

    def visit_ArrayAll(self, op, *, arg):
        value_type = op.arg.dtype.value_type
        return self.if_(
            arg.is_(NULL),
            self.cast(NULL, dt.bool),
            self.if_(
                self.f.array_length(arg) > 0,
                self.if_(
                    self.f.array_has_all(
                        self.f.make_array(self.cast(NULL, value_type)), arg
                    ),
                    self.cast(NULL, dt.bool),
                    self.f.array_has_all(
                        self.f.make_array(True, self.cast(NULL, value_type)), arg
                    ),
                ),
                self.cast(NULL, dt.bool),
            ),
        )

    def visit_ArrayAny(self, op, *, arg):
        value_type = op.arg.dtype.value_type
        return self.if_(
            arg.is_(NULL),
            self.cast(NULL, dt.bool),
            self.if_(
                self.f.array_length(arg) > 0,
                self.if_(
                    self.f.array_has_all(
                        self.f.make_array(self.cast(NULL, value_type)), arg
                    ),
                    self.cast(NULL, dt.bool),
                    self.f.array_has_any(self.f.make_array(True), arg),
                ),
                self.cast(NULL, dt.bool),
            ),
        )

    def visit_BitwiseNot(self, op, *, arg):
        # https://stackoverflow.com/q/69648488/4001592
        return sge.BitwiseXor(this=arg, expression=sge.Literal.number(-1))

    def visit_IntervalFromInteger(self, op, *, arg, unit):
        unit = unit.name.lower()
        return sg.cast(self.f.concat(self.cast(arg, dt.string), f" {unit}"), "interval")

    if (version := util.version("datafusion")) is not None and version >= vparse(
        "48.0.0"
    ):

        def visit_ApproxQuantile(self, op, *, arg, quantile, where):
            expr = sge.WithinGroup(
                this=self.f.approx_percentile_cont(quantile),
                expression=sge.Order(expressions=[sge.Ordered(this=arg)]),
            )
            if where is not None:
                return sge.Filter(this=expr, expression=sge.Where(this=where))
            return expr
    else:

        def visit_ApproxQuantile(self, op, *, arg, quantile, where):
            return self.agg.approx_percentile_cont(arg, quantile, where=where)


compiler = DataFusionCompiler()
