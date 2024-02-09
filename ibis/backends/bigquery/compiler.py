"""Module to convert from Ibis expression to SQL string."""

from __future__ import annotations

import re

import sqlglot as sg
import sqlglot.expressions as sge
from sqlglot.dialects import BigQuery

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis import util
from ibis.backends.sql.compiler import NULL, STAR, SQLGlotCompiler, paren
from ibis.backends.sql.datatypes import BigQueryType, BigQueryUDFType
from ibis.backends.sql.rewrites import (
    exclude_unsupported_window_frame_from_ops,
    exclude_unsupported_window_frame_from_rank,
    exclude_unsupported_window_frame_from_row_number,
    rewrite_first_to_first_value,
    rewrite_last_to_last_value,
    rewrite_sample_as_filter,
)
from ibis.common.temporal import DateUnit, IntervalUnit, TimestampUnit, TimeUnit

_NAME_REGEX = re.compile(r'[^!"$()*,./;?@[\\\]^`{}~\n]+')


class BigQueryCompiler(SQLGlotCompiler):
    dialect = BigQuery
    type_mapper = BigQueryType
    udf_type_mapper = BigQueryUDFType
    rewrites = (
        rewrite_sample_as_filter,
        rewrite_first_to_first_value,
        rewrite_last_to_last_value,
        exclude_unsupported_window_frame_from_ops,
        exclude_unsupported_window_frame_from_row_number,
        exclude_unsupported_window_frame_from_rank,
        *SQLGlotCompiler.rewrites,
    )

    UNSUPPORTED_OPERATIONS = frozenset(
        (
            ops.CountDistinctStar,
            ops.DateDiff,
            ops.ExtractAuthority,
            ops.ExtractFile,
            ops.ExtractFragment,
            ops.ExtractHost,
            ops.ExtractPath,
            ops.ExtractProtocol,
            ops.ExtractQuery,
            ops.ExtractUserInfo,
            ops.FindInSet,
            ops.Median,
            ops.Quantile,
            ops.MultiQuantile,
            ops.RegexSplit,
            ops.RowID,
            ops.TimestampBucket,
            ops.TimestampDiff,
        )
    )

    NAN = sge.Cast(
        this=sge.convert("NaN"), to=sge.DataType(this=sge.DataType.Type.DOUBLE)
    )
    POS_INF = sge.Cast(
        this=sge.convert("Infinity"), to=sge.DataType(this=sge.DataType.Type.DOUBLE)
    )
    NEG_INF = sge.Cast(
        this=sge.convert("-Infinity"), to=sge.DataType(this=sge.DataType.Type.DOUBLE)
    )

    SIMPLE_OPS = {
        ops.StringAscii: "ascii",
        ops.BitAnd: "bit_and",
        ops.BitOr: "bit_or",
        ops.BitXor: "bit_xor",
        ops.DateFromYMD: "date",
        ops.Divide: "ieee_divide",
        ops.EndsWith: "ends_with",
        ops.GeoArea: "st_area",
        ops.GeoAsBinary: "st_asbinary",
        ops.GeoAsText: "st_astext",
        ops.GeoAzimuth: "st_azimuth",
        ops.GeoBuffer: "st_buffer",
        ops.GeoCentroid: "st_centroid",
        ops.GeoContains: "st_contains",
        ops.GeoCoveredBy: "st_coveredby",
        ops.GeoCovers: "st_covers",
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
        ops.GeoUnaryUnion: "st_union_agg",
        ops.GeoUnion: "st_union",
        ops.GeoWithin: "st_within",
        ops.GeoX: "st_x",
        ops.GeoY: "st_y",
        ops.Hash: "farm_fingerprint",
        ops.IsInf: "is_inf",
        ops.IsNan: "is_nan",
        ops.Log10: "log10",
        ops.LPad: "lpad",
        ops.RPad: "rpad",
        ops.Levenshtein: "edit_distance",
        ops.Modulus: "mod",
        ops.RandomScalar: "rand",
        ops.RandomUUID: "generate_uuid",
        ops.RegexReplace: "regexp_replace",
        ops.RegexSearch: "regexp_contains",
        ops.Time: "time",
        ops.TimeFromHMS: "time",
        ops.TimestampFromYMDHMS: "datetime",
        ops.TimestampNow: "current_timestamp",
    }

    def _aggregate(self, funcname: str, *args, where):
        func = self.f[funcname]

        if where is not None:
            args = tuple(self.if_(where, arg, NULL) for arg in args)

        return func(*args, dialect=self.dialect)

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

    def visit_BoundingBox(self, op, *, arg):
        name = type(op).__name__[len("Geo") :].lower()
        return sge.Dot(
            this=self.f.st_boundingbox(arg), expression=sg.to_identifier(name)
        )

    visit_GeoXMax = visit_GeoXMin = visit_GeoYMax = visit_GeoYMin = visit_BoundingBox

    def visit_GeoSimplify(self, op, *, arg, tolerance, preserve_collapsed):
        if (
            not isinstance(op.preserve_collapsed, ops.Literal)
            or op.preserve_collapsed.value
        ):
            raise com.UnsupportedOperationError(
                "BigQuery simplify does not support preserving collapsed geometries, "
                "pass preserve_collapsed=False"
            )
        return self.f.st_simplify(arg, tolerance)

    def visit_ApproxMedian(self, op, *, arg, where):
        return self.agg.approx_quantiles(arg, 2, where=where)[self.f.offset(1)]

    def visit_Pi(self, op):
        return self.f.acos(-1)

    def visit_E(self, op):
        return self.f.exp(1)

    def visit_TimeDelta(self, op, *, left, right, part):
        return self.f.time_diff(left, right, part, dialect=self.dialect)

    def visit_DateDelta(self, op, *, left, right, part):
        return self.f.date_diff(left, right, part, dialect=self.dialect)

    def visit_TimestampDelta(self, op, *, left, right, part):
        left_tz = op.left.dtype.timezone
        right_tz = op.right.dtype.timezone

        if left_tz is None and right_tz is None:
            return self.f.datetime_diff(left, right, part)
        elif left_tz is not None and right_tz is not None:
            return self.f.timestamp_diff(left, right, part)

        raise com.UnsupportedOperationError(
            "timestamp difference with mixed timezone/timezoneless values is not implemented"
        )

    def visit_GroupConcat(self, op, *, arg, sep, where):
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return self.f.string_agg(arg, sep)

    def visit_FloorDivide(self, op, *, left, right):
        return self.cast(self.f.floor(self.f.ieee_divide(left, right)), op.dtype)

    def visit_Log2(self, op, *, arg):
        return self.f.log(arg, 2, dialect=self.dialect)

    def visit_Log(self, op, *, arg, base):
        if base is None:
            return self.f.ln(arg)
        return self.f.log(arg, base, dialect=self.dialect)

    def visit_ArrayRepeat(self, op, *, arg, times):
        start = step = 1
        array_length = self.f.array_length(arg)
        stop = self.f.greatest(times, 0) * array_length
        i = sg.to_identifier("i")
        idx = self.f.coalesce(
            self.f.nullif(self.f.mod(i, array_length), 0), array_length
        )
        series = self.f.generate_array(start, stop, step)
        return self.f.array(
            sg.select(arg[self.f.safe_ordinal(idx)]).from_(self._unnest(series, as_=i))
        )

    def visit_NthValue(self, op, *, arg, nth):
        if not isinstance(op.nth, ops.Literal):
            raise com.UnsupportedOperationError(
                f"BigQuery `nth` must be a literal; got {type(op.nth)}"
            )
        return self.f.nth_value(arg, nth)

    def visit_StrRight(self, op, *, arg, nchars):
        return self.f.substr(arg, -self.f.least(self.f.length(arg), nchars))

    def visit_StringJoin(self, op, *, arg, sep):
        return self.f.array_to_string(self.f.array(*arg), sep)

    def visit_DayOfWeekIndex(self, op, *, arg):
        return self.f.mod(self.f.extract(self.v.dayofweek, arg) + 5, 7)

    def visit_DayOfWeekName(self, op, *, arg):
        return self.f.initcap(sge.Cast(this=arg, to="STRING FORMAT 'DAY'"))

    def visit_StringToTimestamp(self, op, *, arg, format_str):
        if (timezone := op.dtype.timezone) is not None:
            return self.f.parse_timestamp(format_str, arg, timezone)
        return self.f.parse_datetime(format_str, arg)

    def visit_ArrayCollect(self, op, *, arg, where):
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return self.f.array_agg(sge.IgnoreNulls(this=arg))

    def _neg_idx_to_pos(self, arg, idx):
        return self.if_(idx < 0, self.f.array_length(arg) + idx, idx)

    def visit_ArraySlice(self, op, *, arg, start, stop):
        index = sg.to_identifier("bq_arr_slice")
        cond = [index >= self._neg_idx_to_pos(arg, start)]

        if stop is not None:
            cond.append(index < self._neg_idx_to_pos(arg, stop))

        el = sg.to_identifier("el")
        return self.f.array(
            sg.select(el).from_(self._unnest(arg, as_=el, offset=index)).where(*cond)
        )

    def visit_ArrayIndex(self, op, *, arg, index):
        return arg[self.f.safe_offset(index)]

    def visit_ArrayContains(self, op, *, arg, other):
        name = sg.to_identifier(util.gen_name("bq_arr_contains"))
        return sge.Exists(
            this=sg.select(sge.convert(1))
            .from_(self._unnest(arg, as_=name))
            .where(name.eq(other))
        )

    def visit_StringContains(self, op, *, haystack, needle):
        return self.f.strpos(haystack, needle) > 0

    def visti_StringFind(self, op, *, arg, substr, start, end):
        if start is not None:
            raise NotImplementedError(
                "`start` not implemented for BigQuery string find"
            )
        if end is not None:
            raise NotImplementedError("`end` not implemented for BigQuery string find")
        return self.f.strpos(arg, substr)

    def visit_NonNullLiteral(self, op, *, value, dtype):
        if dtype.is_string():
            return sge.convert(
                str(value)
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
        elif dtype.is_inet() or dtype.is_macaddr():
            return sge.convert(str(value))
        elif dtype.is_timestamp():
            funcname = "datetime" if dtype.timezone is None else "timestamp"
            return self.f[funcname](value.isoformat())
        elif dtype.is_date():
            return self.f.datefromparts(value.year, value.month, value.day)
        elif dtype.is_time():
            return self.f.time(value.hour, value.minute, value.second)
        elif dtype.is_binary():
            return sge.Cast(
                this=sge.convert(value.hex()),
                to=sge.DataType(this=sge.DataType.Type.BINARY),
                format=sge.convert("HEX"),
            )
        elif dtype.is_interval():
            if dtype.unit == IntervalUnit.NANOSECOND:
                raise com.UnsupportedOperationError(
                    "BigQuery does not support nanosecond intervals"
                )
        elif dtype.is_uuid():
            return sge.convert(str(value))
        return None

    def visit_IntervalFromInteger(self, op, *, arg, unit):
        if unit == IntervalUnit.NANOSECOND:
            raise com.UnsupportedOperationError(
                "BigQuery does not support nanosecond intervals"
            )
        return sge.Interval(this=arg, unit=self.v[unit.singular])

    def visit_Strftime(self, op, *, arg, format_str):
        arg_dtype = op.arg.dtype
        if arg_dtype.is_timestamp():
            if (timezone := arg_dtype.timezone) is None:
                return self.f.format_datetime(format_str, arg)
            else:
                return self.f.format_timestamp(format_str, arg, timezone)
        elif arg_dtype.is_date():
            return self.f.format_date(format_str, arg)
        else:
            assert arg_dtype.is_time(), arg_dtype
            return self.f.format_time(format_str, arg)

    def visit_IntervalMultiply(self, op, *, left, right):
        unit = self.v[op.left.dtype.resolution.upper()]
        return sge.Interval(this=self.f.extract(unit, left) * right, unit=unit)

    def visit_TimestampFromUNIX(self, op, *, arg, unit):
        unit = op.unit
        if unit == TimestampUnit.SECOND:
            return self.f.timestamp_seconds(arg)
        elif unit == TimestampUnit.MILLISECOND:
            return self.f.timestamp_millis(arg)
        elif unit == TimestampUnit.MICROSECOND:
            return self.f.timestamp_micros(arg)
        elif unit == TimestampUnit.NANOSECOND:
            return self.f.timestamp_micros(
                self.cast(self.f.round(arg / 1_000), dt.int64)
            )
        else:
            raise com.UnsupportedOperationError(f"Unit not supported: {unit}")

    def visit_Cast(self, op, *, arg, to):
        from_ = op.arg.dtype
        if from_.is_timestamp() and to.is_integer():
            return self.f.unix_micros(arg)
        elif from_.is_integer() and to.is_timestamp():
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
            return self.f.extract(self.v[to.resolution.upper()], arg)
        elif from_.is_integer() and to.is_interval():
            return sge.Interval(this=arg, unit=self.v[to.unit.singular])
        elif from_.is_floating() and to.is_integer():
            return self.cast(self.f.trunc(arg), dt.int64)
        return super().visit_Cast(op, arg=arg, to=to)

    def visit_JSONGetItem(self, op, *, arg, index):
        return arg[index]

    def visit_ExtractEpochSeconds(self, op, *, arg):
        return self.f.unix_seconds(arg)

    def visit_ExtractWeekOfYear(self, op, *, arg):
        return self.f.extract(self.v.isoweek, arg)

    def visit_ExtractMillisecond(self, op, *, arg):
        return self.f.extract(self.v.millisecond, arg)

    def visit_ExtractMicrosecond(self, op, *, arg):
        return self.f.extract(self.v.microsecond, arg)

    def visit_TimestampTruncate(self, op, *, arg, unit):
        if unit == IntervalUnit.NANOSECOND:
            raise com.UnsupportedOperationError(
                f"BigQuery does not support truncating {op.arg.dtype} values to unit {unit!r}"
            )
        elif unit == IntervalUnit.WEEK:
            unit = "WEEK(MONDAY)"
        else:
            unit = unit.name
        return self.f.timestamp_trunc(arg, self.v[unit], dialect=self.dialect)

    def visit_DateTruncate(self, op, *, arg, unit):
        if unit == DateUnit.WEEK:
            unit = "WEEK(MONDAY)"
        else:
            unit = unit.name
        return self.f.date_trunc(arg, self.v[unit], dialect=self.dialect)

    def visit_TimeTruncate(self, op, *, arg, unit):
        if unit == TimeUnit.NANOSECOND:
            raise com.UnsupportedOperationError(
                f"BigQuery does not support truncating {op.arg.dtype} values to unit {unit!r}"
            )
        else:
            unit = unit.name
        return self.f.time_trunc(arg, self.v[unit], dialect=self.dialect)

    def _nullifzero(self, step, zero, step_dtype):
        if step_dtype.is_interval():
            return self.if_(step.eq(zero), NULL, step)
        return self.f.nullif(step, zero)

    def _zero(self, dtype):
        if dtype.is_interval():
            return self.f.make_interval()
        return sge.convert(0)

    def _sign(self, value, dtype):
        if dtype.is_interval():
            zero = self._zero(dtype)
            return sge.Case(
                ifs=[
                    self.if_(value < zero, -1),
                    self.if_(value.eq(zero), 0),
                    self.if_(value > zero, 1),
                ],
                default=NULL,
            )
        return self.f.sign(value)

    def _make_range(self, func, start, stop, step, step_dtype):
        step_sign = self._sign(step, step_dtype)
        delta_sign = self._sign(stop - start, step_dtype)
        zero = self._zero(step_dtype)
        nullifzero = self._nullifzero(step, zero, step_dtype)
        condition = sg.and_(sg.not_(nullifzero.is_(NULL)), step_sign.eq(delta_sign))
        gen_array = func(start, stop, step)
        name = sg.to_identifier(util.gen_name("bq_arr_range"))
        inner = (
            sg.select(name)
            .from_(self._unnest(gen_array, as_=name))
            .where(name.neq(stop))
        )
        return self.if_(condition, self.f.array(inner), self.f.array())

    def visit_IntegerRange(self, op, *, start, stop, step):
        return self._make_range(self.f.generate_array, start, stop, step, op.step.dtype)

    def visit_TimestampRange(self, op, *, start, stop, step):
        if op.start.dtype.timezone is None or op.stop.dtype.timezone is None:
            raise com.IbisTypeError(
                "Timestamps without timezone values are not supported when generating timestamp ranges"
            )
        return self._make_range(
            self.f.generate_timestamp_array, start, stop, step, op.step.dtype
        )

    def visit_First(self, op, *, arg, where):
        if where is not None:
            arg = self.if_(where, arg, NULL)
        array = self.f.array_agg(
            sge.Limit(this=sge.IgnoreNulls(this=arg), expression=sge.convert(1)),
        )
        return array[self.f.safe_offset(0)]

    def visit_Last(self, op, *, arg, where):
        if where is not None:
            arg = self.if_(where, arg, NULL)
        array = self.f.array_reverse(self.f.array_agg(sge.IgnoreNulls(this=arg)))
        return array[self.f.safe_offset(0)]

    def visit_Arbitrary(self, op, *, arg, how, where):
        if how != "first":
            raise com.UnsupportedOperationError(
                f"{how!r} value not supported for arbitrary in BigQuery"
            )

        return self.agg.any_value(arg, where=where)

    def visit_ArrayFilter(self, op, *, arg, body, param):
        return self.f.array(
            sg.select(param).from_(self._unnest(arg, as_=param)).where(body)
        )

    def visit_ArrayMap(self, op, *, arg, body, param):
        return self.f.array(sg.select(body).from_(self._unnest(arg, as_=param)))

    def visit_ArrayZip(self, op, *, arg):
        lengths = [self.f.array_length(arr) - 1 for arr in arg]
        idx = sg.to_identifier(util.gen_name("bq_arr_idx"))
        indices = self._unnest(
            self.f.generate_array(0, self.f.greatest(*lengths)), as_=idx
        )
        struct_fields = [
            arr[self.f.safe_offset(idx)].as_(name)
            for name, arr in zip(op.dtype.value_type.names, arg)
        ]
        return self.f.array(
            sge.Select(kind="STRUCT", expressions=struct_fields).from_(indices)
        )

    def visit_ArrayPosition(self, op, *, arg, other):
        name = sg.to_identifier(util.gen_name("bq_arr"))
        idx = sg.to_identifier(util.gen_name("bq_arr_idx"))
        unnest = self._unnest(arg, as_=name, offset=idx)
        return self.f.coalesce(
            sg.select(idx + 1).from_(unnest).where(name.eq(other)).limit(1).subquery(),
            0,
        )

    def _unnest(self, expression, *, as_, offset=None):
        alias = sge.TableAlias(columns=[sg.to_identifier(as_)])
        return sge.Unnest(expressions=[expression], alias=alias, offset=offset)

    def visit_ArrayRemove(self, op, *, arg, other):
        name = sg.to_identifier(util.gen_name("bq_arr"))
        unnest = self._unnest(arg, as_=name)
        return self.f.array(sg.select(name).from_(unnest).where(name.neq(other)))

    def visit_ArrayDistinct(self, op, *, arg):
        name = util.gen_name("bq_arr")
        return self.f.array(
            sg.select(name).distinct().from_(self._unnest(arg, as_=name))
        )

    def visit_ArraySort(self, op, *, arg):
        name = util.gen_name("bq_arr")
        return self.f.array(
            sg.select(name).from_(self._unnest(arg, as_=name)).order_by(name)
        )

    def visit_ArrayUnion(self, op, *, left, right):
        lname = util.gen_name("bq_arr_left")
        rname = util.gen_name("bq_arr_right")
        lhs = sg.select(lname).from_(self._unnest(left, as_=lname))
        rhs = sg.select(rname).from_(self._unnest(right, as_=rname))
        return self.f.array(sg.union(lhs, rhs, distinct=True))

    def visit_ArrayIntersect(self, op, *, left, right):
        lname = util.gen_name("bq_arr_left")
        rname = util.gen_name("bq_arr_right")
        lhs = sg.select(lname).from_(self._unnest(left, as_=lname))
        rhs = sg.select(rname).from_(self._unnest(right, as_=rname))
        return self.f.array(sg.intersect(lhs, rhs, distinct=True))

    def visit_Substring(self, op, *, arg, start, length):
        if isinstance(op.length, ops.Literal) and (value := op.length.value) < 0:
            raise com.IbisInputError(
                f"Length parameter must be a non-negative value; got {value}"
            )
        suffix = (length,) * (length is not None)
        if_pos = self.f.substr(arg, start + 1, *suffix)
        if_neg = self.f.substr(arg, self.f.length(arg) + start + 1, *suffix)
        return self.if_(start >= 0, if_pos, if_neg)

    def visit_RegexExtract(self, op, *, arg, pattern, index):
        matches = self.f.regexp_contains(arg, pattern)
        nonzero_index_replace = self.f.regexp_replace(
            arg,
            self.f.concat(".*?", pattern, ".*"),
            self.f.concat("\\\\", self.cast(index, dt.string)),
        )
        zero_index_replace = self.f.regexp_replace(
            arg, self.f.concat(".*?", self.f.concat("(", pattern, ")"), ".*"), "\\\\1"
        )
        extract = self.if_(index.eq(0), zero_index_replace, nonzero_index_replace)
        return self.if_(matches, extract, NULL)

    def visit_TimestampAddSub(self, op, *, left, right):
        if not isinstance(right, sge.Interval):
            raise com.OperationNotDefinedError(
                "BigQuery does not support non-literals on the right side of timestamp add/subtract"
            )
        if (unit := op.right.dtype.unit) == IntervalUnit.NANOSECOND:
            raise com.UnsupportedOperationError(
                f"BigQuery does not allow binary operation {type(op).__name__} with "
                f"INTERVAL offset {unit}"
            )

        opname = type(op).__name__[len("Timestamp") :]
        funcname = f"TIMESTAMP_{opname.upper()}"
        return self.f.anon[funcname](left, right)

    visit_TimestampAdd = visit_TimestampSub = visit_TimestampAddSub

    def visit_DateAddSub(self, op, *, left, right):
        if not isinstance(right, sge.Interval):
            raise com.OperationNotDefinedError(
                "BigQuery does not support non-literals on the right side of date add/subtract"
            )
        if not (unit := op.right.dtype.unit).is_date():
            raise com.UnsupportedOperationError(
                f"BigQuery does not allow binary operation {type(op).__name__} with "
                f"INTERVAL offset {unit}"
            )
        opname = type(op).__name__[len("Date") :]
        funcname = f"DATE_{opname.upper()}"
        return self.f.anon[funcname](left, right)

    visit_DateAdd = visit_DateSub = visit_DateAddSub

    def visit_Covariance(self, op, *, left, right, how, where):
        if where is not None:
            left = self.if_(where, left, NULL)
            right = self.if_(where, right, NULL)

        if op.left.dtype.is_boolean():
            left = self.cast(left, dt.int64)

        if op.right.dtype.is_boolean():
            right = self.cast(right, dt.int64)

        how = op.how[:4].upper()
        assert how in ("POP", "SAMP"), 'how not in ("POP", "SAMP")'
        return self.agg[f"COVAR_{how}"](left, right, where=where)

    def visit_Correlation(self, op, *, left, right, how, where):
        if how == "sample":
            raise ValueError(f"Correlation with how={how!r} is not supported.")

        if where is not None:
            left = self.if_(where, left, NULL)
            right = self.if_(where, right, NULL)

        if op.left.dtype.is_boolean():
            left = self.cast(left, dt.int64)

        if op.right.dtype.is_boolean():
            right = self.cast(right, dt.int64)

        return self.agg.corr(left, right, where=where)

    def visit_TypeOf(self, op, *, arg):
        name = sg.to_identifier(util.gen_name("bq_typeof"))
        from_ = self._unnest(self.f.array(self.f.format("%T", arg)), as_=name)
        ifs = [
            self.if_(
                self.f.regexp_contains(name, '^[A-Z]+ "'),
                self.f.regexp_extract(name, '^([A-Z]+) "'),
            ),
            self.if_(self.f.regexp_contains(name, "^-?[0-9]*$"), "INT64"),
            self.if_(
                self.f.regexp_contains(
                    name, r'^(-?[0-9]+[.e].*|CAST\\("([^"]*)" AS FLOAT64\\))$'
                ),
                "FLOAT64",
            ),
            self.if_(name.isin(sge.convert("true"), sge.convert("false")), "BOOL"),
            self.if_(
                sg.or_(self.f.starts_with(name, '"'), self.f.starts_with(name, "'")),
                "STRING",
            ),
            self.if_(self.f.starts_with(name, 'b"'), "BYTES"),
            self.if_(self.f.starts_with(name, "["), "ARRAY"),
            self.if_(self.f.regexp_contains(name, r"^(STRUCT)?\\("), "STRUCT"),
            self.if_(self.f.starts_with(name, "ST_"), "GEOGRAPHY"),
            self.if_(name.eq(sge.convert("NULL")), "NULL"),
        ]
        case = sge.Case(ifs=ifs, default=sge.convert("UNKNOWN"))
        return sg.select(case).from_(from_).subquery()

    def visit_Xor(self, op, *, left, right):
        return sg.or_(sg.and_(left, sg.not_(right)), sg.and_(sg.not_(left), right))

    def visit_HashBytes(self, op, *, arg, how):
        if how not in ("md5", "sha1", "sha256", "sha512"):
            raise NotImplementedError(how)
        return self.f[how](arg)

    @staticmethod
    def _gen_valid_name(name: str) -> str:
        return "_".join(_NAME_REGEX.findall(name)) or "tmp"

    def visit_CountStar(self, op, *, arg, where):
        if where is not None:
            return self.f.countif(where)
        return self.f.count(STAR)

    def visit_Degrees(self, op, *, arg):
        return paren(180 * arg / self.f.acos(-1))

    def visit_Radians(self, op, *, arg):
        return paren(self.f.acos(-1) * arg / 180)

    def visit_CountDistinct(self, op, *, arg, where):
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return self.f.count(sge.Distinct(expressions=[arg]))
