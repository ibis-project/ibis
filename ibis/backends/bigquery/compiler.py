from __future__ import annotations

import contextlib
import math
from functools import singledispatchmethod

import sqlglot as sg
import sqlglot.expressions as sge
from public import public

# from sqlglot import exp
# from sqlglot.dialects import BigQuery
# from sqlglot.dialects.dialect import rename_func
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis import util
from ibis.backends.base.sqlglot.compiler import NULL, C, SQLGlotCompiler
from ibis.backends.bigquery.datatypes import BigQueryType
from ibis.common.temporal import DateUnit, IntervalUnit, TimeUnit

# from ibis.common.patterns import replace
# from ibis.expr.analysis import p, x, y


# @replace(p.WindowFunction(p.First(x, y)))
# def rewrite_first(_, x, y):
#     if y is not None:
#         raise com.UnsupportedOperationError(
#             "`first` aggregate over window does not support `where`"
#         )
#     return _.copy(func=ops.FirstValue(x))
#
#
# @replace(p.WindowFunction(p.Last(x, y)))
# def rewrite_last(_, x, y):
#     if y is not None:
#         raise com.UnsupportedOperationError(
#             "`last` aggregate over window does not support `where`"
#         )
#     return _.copy(func=ops.LastValue(x))
#
#
# @replace(p.WindowFunction(frame=x @ p.WindowFrame(order_by=())))
# def rewrite_empty_order_by_window(_, x):
#     return _.copy(frame=x.copy(order_by=(ibis.NA,)))
#
#
# @replace(p.WindowFunction(p.RowNumber | p.NTile, x))
# def exclude_unsupported_window_frame_from_row_number(_, x):
#     return ops.Subtract(_.copy(frame=x.copy(start=None, end=None)), 1)
#
#
# @replace(
#     p.WindowFunction(
#         p.Lag | p.Lead | p.PercentRank | p.CumeDist | p.Any | p.All,
#         x @ p.WindowFrame(start=None),
#     )
# )
# def exclude_unsupported_window_frame_from_ops(_, x):
#     return _.copy(frame=x.copy(start=None, end=None))
#
#
# @replace(p.ToJSONMap | p.ToJSONArray)
# def replace_to_json(_):
#     return ops.Cast(_.arg, to=_.dtype)


@public
class BigQueryCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = "bigquery"
    quoted = True
    type_mapper = BigQueryType

    NAN = sge.Cast(
        this=sge.convert("NaN"), to=sge.DataType(this=sge.DataType.Type.DOUBLE)
    )
    POS_INF = sge.Cast(
        this=sge.convert("Inf"), to=sge.DataType(this=sge.DataType.Type.DOUBLE)
    )
    NEG_INF = sge.Cast(
        this=sge.convert("-Inf"), to=sge.DataType(this=sge.DataType.Type.DOUBLE)
    )
    # rewrites = (
    #     replace_to_json,
    #     exclude_unsupported_window_frame_from_row_number,
    #     exclude_unsupported_window_frame_from_ops,
    #     rewrite_first,
    #     rewrite_last,
    #     rewrite_empty_order_by_window,
    #     *SQLGlotCompiler.rewrites,
    # )

    def _aggregate(self, funcname: str, *args, where):
        if where is not None:
            args = [self.if_(where, arg, NULL) for arg in args]

        func = self.f[funcname]
        return func(*args)

    @singledispatchmethod
    def visit_node(self, op, **kw):
        return super().visit_node(op, **kw)

    @visit_node.register(ops.IntegerRange)
    def visit_IntegerRange(self, op, *, start, stop, step):
        n = self.f.floor((stop - start) / self.f.nullif(step, 0))
        gen_array = self.f.generate_array(start, stop, step)
        inner = (
            sg.select(C.x)
            .from_(self.unnest(gen_array, as_="x"))
            .where(sg.and_(C.x != stop))
        )
        return self.if_(n > 0, self.f.array(inner), self.f.array())

    @visit_node.register(ops.TimeDelta)
    def visit_TimeDelta(self, op, *, part, left, right):
        return self.f.time_diff(left, right, part.value.upper())

    @visit_node.register(ops.TimeDelta)
    def visit_DateDelta(self, op, *, part, left, right):
        return self.f.date_diff(left, right, part.value.upper(), dialect=self.dialect)

    @visit_node.register(ops.TimeDelta)
    def visit_TimestampDelta(self, op, *, part, left, right):
        left_tz = op.left.dtype.timezone
        right_tz = op.right.dtype.timezone

        if left_tz is None and right_tz is None:
            return self.f.datetime_diff(left, right, part.value.upper())
        elif left_tz is not None and right_tz is not None:
            return self.f.timestamp_diff(left, right, part.value.upper())
        else:
            raise NotImplementedError(
                "timestamp difference with mixed timezone/timezoneless values is not implemented"
            )

    @visit_node.register(ops.Pi)
    def visit_Pi(self, op):
        return self.f.acos(-1)

    @visit_node.register(ops.FindInSet)
    @visit_node.register(ops.CountDistinctStar)
    @visit_node.register(ops.DateDiff)
    @visit_node.register(ops.TimestampDiff)
    @visit_node.register(ops.ExtractAuthority)
    @visit_node.register(ops.ExtractFile)
    @visit_node.register(ops.ExtractFragment)
    @visit_node.register(ops.ExtractHost)
    @visit_node.register(ops.ExtractPath)
    @visit_node.register(ops.ExtractProtocol)
    @visit_node.register(ops.ExtractQuery)
    @visit_node.register(ops.ExtractUserInfo)
    def visit_Undefined(self, op, **_):
        raise com.OperationNotDefinedError(type(op).__name__)

    @visit_node.register(ops.WindowBoundary)
    def visit_WindowBoundary(self, op, *, value, preceding):
        if not isinstance(op.value, ops.Literal):
            raise com.OperationNotDefinedError(
                "Expressions in window bounds are not supported by BigQuery"
            )
        return super().visit_WindowBoundary(op, value=value, preceding=preceding)

    @visit_node.register(ops.ArrayIndex)
    def visit_ArrayIndex(self, op, *, arg, index):
        return arg[self.f.safe_offset(index)]

    @visit_node.register(ops.StructField)
    def visit_StructField(self, op, *, arg, field):
        return sge.Dot(this=arg, expression=sg.to_identifier(field))

    @visit_node.register(ops.ArrayMap)
    def visit_ArrayMap(self, op, *, arg, body, param):
        return self.f.array(sg.select(body).from_(self.unnest(arg, as_=param)))

    @visit_node.register(ops.ArrayFilter)
    def visit_ArrayFilter(self, op, *, arg, body, param):
        return self.f.array(
            sg.select(param).from_(self.unnest(arg, as_=param)).where(body)
        )

    @visit_node.register(ops.StringJoin)
    def visit_StringJoin(self, op, *, sep, arg):
        return self.f.array_to_string(self.f.array(*arg), sep)

    @visit_node.register(ops.StringAscii)
    def visit_StringAscii(self, op, *, arg):
        return self.f.to_code_points(arg)[self.f.safe_offset(0)]

    @visit_node.register(ops.StrRight)
    def visit_StringRight(self, op, *, arg, nchars):
        return self.f.substr(arg, -self.f.least(self.f.length(arg), nchars))

    @visit_node.register(ops.DayOfWeekIndex)
    def visit_DayOfWeekIndex(self, op, *, arg):
        return self.mod(self.f.extract("dayofweek", arg) + 5, 7)

    @visit_node.register(ops.DayOfWeekName)
    def visit_DayOfWeekName(self, op, *, arg):
        return self.f.initcap(
            sge.Cast(
                this=arg,
                to=self.type_mapper.from_ibis(dt.string),
                format=sge.convert("day"),
            )
        )

    @visit_node.register(ops.ApproxMedian)
    def visit_ApproxMedian(self, op, *, arg, where):
        return self.agg.approx_quantiles(arg, 2, where=where)[self.f.offset(1)]

    @visit_node.register(ops.FloorDivide)
    def visit_FloorDivide(self, op, *, left, right):
        return self.f.floor(self.f.ieee_divide(left, right))

    @visit_node.register(ops.ArrayRepeat)
    def visit_ArrayRepeat(self, op, *, arg, times):
        start = step = 1
        array_length = self.f.array_length(arg)
        stop = self.f.greatest(times, 0) * array_length
        i = sg.to_identifier("i")
        idx = self.f.coalesce(self.f.nullif(i % array_length, 0), array_length)
        series = self.f.generate_array(start, stop, step)
        return self.f.array(
            sg.select(arg[self.f.safe_ordinal(idx)]).from_(self.unnest(series, as_=i))
        )

    @visit_node.register(ops.ArrayCollect)
    def visit_ArrayCollect(self, op, *, arg, where):
        return self.agg.array_agg(sge.IgnoreNulls(this=arg), where=where)

    @visit_node.register(ops.ArrayContains)
    def visit_ArrayContains(self, op, *, arg, other):
        name = util.gen_name("bq_arr")
        return (
            sg.select(self.f.logical_or(C[name].eq(other)))
            .from_(self.unnest(arg, as_=name))
            .subquery()
        )

    @visit_node.register(ops.ArrayRemove)
    def visit_ArrayRemove(self, op, *, arg, other):
        name = util.gen_name("bq_arr")
        return self.f.array(
            sg.select(name).from_(self.unnest(arg, as_=name)).where(C[name].neq(other))
        )

    @visit_node.register(ops.ArrayDistinct)
    def visit_ArrayDistinct(self, op, *, arg, other):
        name = util.gen_name("bq_arr")
        return self.f.array(
            sg.select(sge.Distinct(expressions=[C[name]])).from_(
                self.unnest(arg, as_=name)
            )
        )

    @visit_node.register(ops.ArraySort)
    def visit_ArraySort(self, op, *, arg):
        name = util.gen_name("bq_arr")
        return self.f.array(
            sg.select(name).from_(self.unnest(arg, as_=name)).order_by(name)
        )

    @visit_node.register(ops.StructColumn)
    def visit_StructColumn(self, op, *, names, values):
        return sge.Struct(
            expressions=[value.as_(name) for name, value in zip(names, values)]
        )

    @visit_node.register(ops.Strftime)
    def visit_Strftime(self, op, *, arg, format_str):
        arg_type = op.arg.dtype
        if arg_type.is_timestamp():
            args = [format_str, arg]
            if (tz := arg_type.timezone) is not None:
                args.append(sge.convert(tz))
            return self.f.format_datetime(*args)
        elif arg_type.is_date():
            return self.f.format_date(format_str, arg)
        else:
            assert arg_type.is_time(), arg_type
            return self.f.format_time(format_str, arg)

    @visit_node.register(ops.Log2)
    def visit_Log2(self, op, *, arg):
        return self.f.log(2, arg)

    @visit_node.register(ops.Log10)
    def visit_Log10(self, op, *, arg):
        return self.f.log10(arg)

    @visit_node.register(ops.Ln)
    def visit_Ln(self, op, *, arg):
        return self.f.ln(arg)

    @visit_node.register(ops.Log)
    def visit_Log(self, op, *, arg, base):
        if base is None:
            return self.f.ln(arg)
        return self.f.log(base, arg)

    @visit_node.register(ops.JSONGetItem)
    def visit_JSONGetItem(self, op, *, arg, index):
        return arg[index]

    @visit_node.register(ops.ArrayZip)
    def visit_ArrayZip(self, op, *, arg):
        idx = sg.to_identifier(util.gen_name("bq_arr_idx"))
        indices = self.unnest(
            self.f.generate_array(1, self.f.greatest(*map(self.f.array_length, arg))),
            as_=idx,
        )
        struct_fields = [
            arr[self.f.safe_ordinal(idx)].as_(name)
            for name, arr in zip(op.dtype.value_type.names, arg)
        ]
        return self.f.array(
            sge.Select(kind="STRUCT", expressions=struct_fields).from_(indices)
        )

    @visit_node.register(ops.StringFind)
    def visit_StringFind(self, op, *, arg, substr, start, end):
        if start is not None:
            raise NotImplementedError("start not implemented for string find")
        if end is not None:
            raise NotImplementedError("end not implemented for string find")

        return self.f.strpos(arg, substr) - 1

    @visit_node.register(ops.ArrayIntersect)
    def visit_ArrayIntersect(self, op, *, left, right):
        name = util.gen_name("bq_arr")
        lname = sg.to_identifier(util.gen_name("bq_arr_left"))
        rname = sg.to_identifier(util.gen_name("bq_arr_right"))
        return self.f.array(
            sg.intersect(
                sg.select(lname.as_(name)).from_(self.unnest(left, as_=lname)),
                sg.select(rname.as_(name)).from_(self.unnest(right, as_=rname)),
                distinct=True,
            )
        )

    @visit_node.register(ops.ArrayUnion)
    def visit_ArrayUnion(self, op, *, left, right):
        name = util.gen_name("bq_arr")
        lname = sg.to_identifier(util.gen_name("bq_arr_left"))
        rname = sg.to_identifier(util.gen_name("bq_arr_right"))
        return self.f.array(
            sg.union(
                sg.select(lname.as_(name)).from_(self.unnest(left, as_=lname)),
                sg.select(rname.as_(name)).from_(self.unnest(right, as_=rname)),
                distinct=True,
            )
        )

    @visit_node.register(ops.ArrayPosition)
    def visit_ArrayPosition(self, op, *, arg, other):
        name = util.gen_name("bq_arr")
        idx = util.gen_name("bq_arr_idx")
        unnest = self.unnest(arg, as_=name, offset=idx)
        return self.f.coalesce(
            sg.select(idx).from_(unnest).where(C.name.eq(other)).limit(1).subquery(), -1
        )

    def unnest(
        self, *args: sge.Expression, as_: str | None = None, offset: str | None = None
    ):
        """Generate an UNNEST expression specific to BigQuery."""
        return sge.Unnest(expressions=list(args), alias=as_, offset=offset)

    @visit_node.register(ops.Literal)
    def visit_Literal(self, op, *, value, dtype):
        if value is None:
            if not dtype.is_null():
                return f"CAST(NULL AS {self.type_mapper.to_string(dtype)})"
            return "NULL"
        elif dtype.is_string() or dtype.is_inet() or dtype.is_macaddr():
            return sge.convert(
                op.value
                # Escape \ first so we don't double escape other characters.
                .replace("\\", "\\\\")
                # Escape ' since we're using those for the string literal.
                .replace("'", "\\'")
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
        elif dtype.is_decimal():
            if value.is_nan():
                return self.NAN
            elif value.is_infinite():
                return self.POS_INF if value > 0 else self.NEG_INF
            else:
                return self.cast(value, dtype)
        elif dtype.is_uuid():
            return sge.convert(str(value))
        elif dtype.is_numeric():
            if dtype.is_float64():
                return sge.convert(value)
            elif not math.isfinite(value):
                return self.cast(value, dt.float64)
            return sge.convert(value)
        elif dtype.is_date():
            with contextlib.suppress(AttributeError):
                value = value.date()
            return self.f.date(str(value))
        elif dtype.is_timestamp():
            func = "datetime" if dtype.timezone is None else "timestamp"
            return self.f[func](str(value))
        elif dtype.is_time():
            # TODO: define extractors on TimeValue expressions
            return self.f.time(str(value))
        elif dtype.is_binary():
            return sge.convert(repr(value))
        elif dtype.is_struct():
            cols = [
                self.visit_Literal(
                    ops.Literal(value[name], dtype=typ), value=value[name], dtype=typ
                ).as_(name)
                for name, typ in dtype.items()
            ]
            return self.f.struct(*cols)
        elif dtype.is_array():
            val_type = dtype.value_type
            values = (
                self.visit_Literal(
                    ops.Literal(element, dtype=val_type), value=element, dtype=val_type
                )
                for element in value
            )
            return self.f.array(*values)
        elif dtype.is_interval():
            return sge.Interval(this=value, unit=dtype.resolution.upper())
        else:
            return super().visit_node(op, value=value, dtype=dtype)

    @visit_node.register(ops.GeoSimplify)
    def visit_GeoSimplify(self, op, *, arg, tolerance, preserve_collapsed):
        if preserve_collapsed.value:
            raise com.UnsupportedOperationError(
                "BigQuery simplify does not support preserving collapsed geometries, "
                "must pass preserve_collapsed=False"
            )
        return self.f.st_simplify(arg, tolerance)

    @visit_node.register(ops.GeoXMax)
    @visit_node.register(ops.GeoXMin)
    @visit_node.register(ops.GeoYMax)
    @visit_node.register(ops.GeoYMin)
    def visit_GeoXYMaxMin(self, op, *, arg):
        dimension_name = op.__class__.__name__.lower().replace("geo", "")
        return sge.Dot(
            this=self.f.st_boundingbox(arg), expression=sg.to_identifier(dimension_name)
        )

    def _neg_idx_to_pos(self, array, idx):
        return self.if_(idx < 0, self.f.array_length(array) + idx, idx)

    @visit_node.register(ops.ArraySlice)
    def visit_ArraySlice(self, op, *, arg, start, stop):
        index = sg.to_identifier("index")
        el = sg.to_identifier("el")

        cond = [index >= self._neg_idx_to_pos(arg, start)]
        if stop is not None:
            cond.append(index < self._neg_idx_to_pos(arg, stop))
        return self.f.array(
            sg.select(el)
            .from_(self.unnest(arg, as_=el, offset=index))
            .where(sg.and_(*cond))
        )

    @visit_node.register(ops.Substring)
    def visit_Substring(self, op, *, arg, start, length):
        # if (length := getattr(length, "value", None)) is not None and length < 0:
        #     raise ValueError("Length parameter must be a non-negative value.")
        args = []
        if length is not None:
            args.append(length)

        if_pos = self.f.substr(arg, start + 1, *args)
        if_neg = self.f.substr(arg, self.f.length(arg) + start + 1, *args)
        return self.if_(start >= 0, if_pos, if_neg)

    @visit_node.register(ops.Arbitrary)
    def visit_Arbitrary(self, op, *, arg, how, where):
        if how != "first":
            raise com.UnsupportedOperationError(
                f"{how!r} value not supported for arbitrary in BigQuery"
            )
        return self.agg.any_value(arg, where=where)

    @visit_node.register(ops.ExtractEpochSeconds)
    def visit_ExtractEpochSeconds(self, op, *, arg):
        return self.f.unix_seconds(arg)

    @visit_node.register(ops.ExtractWeekOfYear)
    def visit_ExtractWeekOfYear(self, op, *, arg):
        return self.f.extract("isoweek", arg)

    @visit_node.register(ops.ExtractDayOfYear)
    def visit_ExtractDayOfYear(self, op, *, arg):
        return self.f.extract("dayofyear", arg)

    @visit_node.register(ops.TimestampFromUNIX)
    def visit_TimestampFomUNIX(self, op, *, arg, unit) -> str:
        """Interprets an integer as a timestamp."""
        unit = unit.short

        if unit == "s":
            return self.f.timestamp_seconds(arg)
        elif unit == "ms":
            return self.f.timestamp_millis(arg)
        elif unit == "us":
            return self.f.timestamp_micros(arg)
        elif unit == "ns":
            # Timestamps are represented internally as elapsed microseconds, so some
            # rounding is required if an integer represents nanoseconds.
            # https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#timestamp_type
            return self.f.timestamp_micros(
                self.cast(self.f.round(arg / 1000), dt.int64)
            )

        raise NotImplementedError(f"cannot cast unit {op.unit}")

    @visit_node.register(ops.IntervalMultiply)
    def visit_IntervalMultiply(self, op, *, left, right):
        unit = op.left.dtype.resolution.upper()
        return sge.Interval(this=self.f.extract(unit, left) * right, unit=unit)

    @visit_node.register(ops.First)
    def visit_First(self, op, *, arg, where):
        return self.agg.array_agg(
            sge.Limit(this=sge.IgnoreNulls(this=arg), expression=sge.convert(1)),
            where=where,
        )[self.f.safe_offset(0)]

    @visit_node.register(ops.Last)
    def visit_Last(self, op, *, arg, where):
        return self.agg.array_agg(sge.IgnoreNulls(this=arg), where=where)[
            self.safe_ordinal(self.agg.count(arg, where=where))
        ]

    @visit_node.register(ops.DateTruncate)
    def visit_DateTruncate(self, op, *, arg, unit):
        if unit not in DateUnit:
            raise com.UnsupportedOperationError(
                f"BigQuery does not support truncating {arg.dtype} values to unit {unit!r}"
            )
        if unit.name == "WEEK":
            unit = "WEEK(MONDAY)"
        else:
            unit = unit.name
        return self.date_trunc(arg, sge.var(unit))

    @visit_node.register(ops.TimestampTruncate)
    def visit_TimestampTruncate(self, op, *, arg, unit):
        if unit not in frozenset(IntervalUnit).difference((IntervalUnit.NANOSECOND,)):
            raise com.UnsupportedOperationError(
                f"BigQuery does not support truncating {arg.dtype} values to unit {unit!r}"
            )
        if unit.name == "WEEK":
            unit = "WEEK(MONDAY)"
        else:
            unit = unit.name
        return self.timestamp_trunc(arg, sge.var(unit))

    @visit_node.register(ops.TimeTruncate)
    def visit_TimeTruncate(self, op, *, arg, unit):
        if unit not in frozenset(TimeUnit).difference((TimeUnit.NANOSECOND,)):
            raise com.UnsupportedOperationError(
                f"BigQuery does not support truncating {arg.dtype} values to unit {unit!r}"
            )
        unit = unit.name
        return self.time_trunc(arg, sge.var(unit))

    @visit_node.register(ops.Covariance)
    def visit_Covariance(self, op, *, left, right, how, where):
        if op.left.dtype.is_boolean():
            left = self.cast(left, dt.int64)
        if op.right.dtype.is_boolean():
            right = self.cast(right, dt.int64)
        return self.agg[f"covar_{how[:4]}"](left, right, where=where)

    @visit_node.register(ops.Correlation)
    def visit_Correlation(self, op, *, left, right, how, where):
        if how == "sample":
            raise ValueError(f"Correlation with how={how!r} is not supported.")

        if op.left.dtype.is_boolean():
            left = self.cast(left, dt.int64)
        if op.right.dtype.is_boolean():
            right = self.cast(right, dt.int64)

        return self.agg.corr(left, right, where=where)

    @visit_node.register(ops.RegexExtract)
    def visit_RegexExtract(self, op, *, arg, pattern, index):
        matches = self.f.regexp_contains(arg, pattern)
        nonzero_index_replace = self.f.regexp_replace(
            arg,
            self.f.concat(".*?", pattern, ".*"),
            self.f.concat(r"\\", self.cast(index, dt.string)),
        )
        zero_index_replace = self.f.reegxp_replace(
            arg, self.f.concat(".*?", self.f.concat("(", pattern, ")"), ".*"), r"\\1"
        )
        extract = self.if_(index.eq(0), zero_index_replace, nonzero_index_replace)

        return self.if_(matches, extract, NULL)

    @visit_node.register(ops.Cast)
    def visit_Cast(self, op, *, arg, to):
        from_ = op.arg.dtype
        if from_.is_timestamp() and to.is_integer():
            return self.f.unix_micros(arg)
        elif from_.is_integer() and to.is_timestamp():
            if to.timezone is None:
                return self.f.datetime_seconds(arg)
            else:
                return self.f.timestamp_seconds(arg)
        elif from_.is_interval() and to.is_integer():
            if from_.unit in {
                IntervalUnit.WEEK,
                IntervalUnit.QUARTER,
                IntervalUnit.NANOSECOND,
            }:
                raise com.UnsupportedOperationError(
                    f"BigQuery does not allow extracting date part `{from_.unit}` from intervals"
                )

            return self.f.extract(from_.resolution.upper(), arg)
        elif from_.is_floating() and to.is_integer():
            return self.cast(self.f.trunc(arg), dt.int64)
        else:
            return super().visit_node(op, arg=arg, to=to)


_SIMPLE_OPS = {
    ops.IsNan: "is_nan",
    ops.IsInf: "is_inf",
    ops.Divide: "ieee_divide",
    ops.DateAdd: "date_add",
    ops.DateSub: "date_sub",
    ops.Time: "time",
    ops.Date: "date",
    ops.TimeFromHMS: "time",
    ops.TimestampAdd: "timestamp_add",
    ops.TimestampSub: "timestamp_sub",
    ops.TimestampFromYMDHMS: "datetime",
    ops.TimestampNow: "current_timestamp",
    ops.Repeat: "repeat",
    ops.RandomScalar: "rand",
    ops.StartsWith: "starts_with",
    ops.EndsWith: "ends_with",
    ops.BitAnd: "bit_and",
    ops.BitOr: "bit_or",
    ops.BitXor: "bit_xor",
    ops.ApproxCountDistinct: "approx_count_distinct",
    ops.Hash: "farm_fingerprint",
    ops.RegexSearch: "regexp_contains",
    ops.RegexReplace: "regexp_replace",
    ops.ArrayStringJoin: "array_to_string",
    ops.GeoUnaryUnion: "st_union_agg",
    ops.GeoArea: "st_area",
    ops.GeoAsBinary: "st_asbinary",
    ops.GeoAsText: "st_astext",
    ops.GeoAzimuth: "st_azimuth",
    ops.GeoBuffer: "st_buffer",
    ops.GeoCentroid: "st_centroid",
    ops.GeoContains: "st_contains",
    ops.GeoCovers: "st_covers",
    ops.GeoCoveredBy: "st_coveredby",
    ops.GeoDWithin: "st_dwithin",
    ops.GeoDifference: "st_difference",
    ops.GeoDisjoint: "st_disjoint",
    ops.GeoDistance: "st_distance",
    ops.GeoEndPoint: "st_endpoint",
    ops.GeoEquals: "st_equals",
    ops.GeoGeometryType: "st_geometrytype",
    ops.GeoIntersection: "st_intersection",
    ops.GeoIntersects: "st_intersects",
    ops.GeoLength: "st_length",
    ops.GeoMaxDistance: "st_maxdistance",
    ops.GeoNPoints: "st_numpoints",
    ops.GeoPerimeter: "st_perimeter",
    ops.GeoPoint: "st_geogpoint",
    ops.GeoPointN: "st_pointn",
    ops.GeoStartPoint: "st_startpoint",
    ops.GeoTouches: "st_touches",
    ops.GeoUnion: "st_union",
    ops.GeoWithin: "st_within",
    ops.GeoX: "st_x",
    ops.GeoY: "st_y",
}

for _op, _name in _SIMPLE_OPS.items():
    assert isinstance(type(_op), type), type(_op)
    if issubclass(_op, ops.Reduction):

        @BigQueryCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, where, **kw):
            return self.agg[_name](*kw.values(), where=where)

    else:

        @BigQueryCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, **kw):
            return self.f[_name](*kw.values())

    setattr(BigQueryCompiler, f"visit_{_op.__name__}", _fmt)


del _op, _name, _fmt
