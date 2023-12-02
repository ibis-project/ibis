from __future__ import annotations

import itertools
from functools import reduce, singledispatchmethod

import sqlglot as sg
import sqlglot.expressions as sge
from public import public
from sqlglot import exp
from sqlglot.dialects import Snowflake
from sqlglot.dialects.dialect import rename_func

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis import util
from ibis.backends.base.sqlglot.compiler import NULL, STAR, SQLGlotCompiler
from ibis.backends.snowflake.datatypes import SnowflakeType
from ibis.common.patterns import replace
from ibis.expr.analysis import p, x, y

Snowflake.Generator.TRANSFORMS[exp.ApproxDistinct] = rename_func(
    "approx_count_distinct"
)


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


@replace(p.WindowFunction(p.RowNumber, x))
def exclude_unsupported_window_frame_from_row_number(_, x):
    return ops.Subtract(_.copy(frame=x.copy(start=None, end=None)), 1)


@replace(p.Log2(x))
def replace_log2(_, x):
    return ops.Log(2, x)


@replace(p.Log10(x))
def replace_log10(_, x):
    return ops.Log(10, x)


@replace(p.ToJSONArray)
@replace(p.ToJSONMap)
def replace_to_json(_):
    return ops.Cast(_.arg, to=_.dtype)


@public
class SnowflakeCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = "snowflake"
    quoted = True
    type_mapper = SnowflakeType
    rewrites = (
        exclude_unsupported_window_frame_from_row_number,
        rewrite_first,
        rewrite_last,
        rewrite_empty_order_by_window,
        replace_log2,
        replace_log10,
        replace_to_json,
        *SQLGlotCompiler.rewrites,
    )

    def _aggregate(self, funcname: str, *args, where):
        if where is not None:
            args = [self.if_(where, arg, NULL) for arg in args]

        func = self.f[funcname]
        return func(*args)

    @singledispatchmethod
    def visit_node(self, op, **kw):
        return super().visit_node(op, **kw)

    @visit_node.register(ops.Literal)
    def visit_Literal(self, op, *, value, dtype):
        if value is None:
            return super().visit_Literal(op, value=value, dtype=dtype)
        elif dtype.is_string():
            # sqlglot doesn't escape backslashes in strings
            return sge.convert(value.replace("\\", "\\\\"))
        elif dtype.is_timestamp():
            args = (
                value.year,
                value.month,
                value.day,
                value.hour,
                value.minute,
                value.second,
                value.microsecond * 1_000,
            )
            if value.tzinfo is not None:
                return self.f.timestamp_tz_from_parts(*args, dtype.timezone)
            else:
                return self.f.timestamp_from_parts(*args)
        elif dtype.is_time():
            nanos = value.microsecond * 1_000
            return self.f.time_from_parts(value.hour, value.minute, value.second, nanos)
        elif dtype.is_map():
            key_type = dtype.key_type
            keys = (
                self.visit_Literal(ops.Literal(k, key_type), value=k, dtype=key_type)
                for k in value.keys()
            )

            value_type = dtype.value_type
            values = (
                self.visit_Literal(
                    ops.Literal(v, value_type), value=v, dtype=value_type
                )
                for v in value.values()
            )

            return self.f.object_construct_keep_null(
                *itertools.chain.from_iterable(zip(keys, values))
            )
        elif dtype.is_struct():
            pairs = []
            for k, v in value.items():
                pairs.append(k)
                pairs.append(
                    self.visit_Literal(ops.Literal(v, dtype), value=v, dtype=dtype)
                )
            return self.f.object_construct_keep_null(*pairs)

        elif dtype.is_uuid():
            return sge.convert(str(value))
        elif dtype.is_binary():
            return sge.HexString(this=value.hex())
        return super().visit_node(op, value=value, dtype=dtype)

    @visit_node.register(ops.Cast)
    def visit_Cast(self, op, *, arg, to):
        if to.is_struct() or to.is_map():
            return self.if_(self.f.is_object(arg), arg, NULL)
        elif to.is_array():
            return self.if_(self.f.is_array(arg), arg, NULL)
        return self.cast(arg, to)

    @visit_node.register(ops.IsNan)
    def visit_IsNan(self, op, *, arg):
        return arg.eq(self.NAN)

    @visit_node.register(ops.IsInf)
    def visit_IsInf(self, op, *, arg):
        return arg.isin(self.POS_INF, self.NEG_INF)

    @visit_node.register(ops.JSONGetItem)
    def visit_JSONGetItem(self, op, *, arg, index):
        return self.f.get(arg, index)

    @visit_node.register(ops.StringFind)
    def visit_StringFind(self, op, *, arg, substr, start, end):
        args = [substr, arg]
        if start is not None:
            start += 1
            args.append(start)
        return self.f.position(*args)

    def _call_udf(self, name: str, *args, **kwargs) -> str:
        return self.f[f"ibis_udfs.public.{name}"](*args, **kwargs)

    @visit_node.register(ops.Map)
    def visit_Map(self, op, *, keys, values):
        return self.if_(
            sg.and_(self.f.is_array(keys), self.f.is_array(values)),
            self._call_udf("object_from_arrays", keys, values),
            NULL,
        )

    @visit_node.register(ops.MapKeys)
    def visit_MapKeys(self, op, *, arg):
        return self.if_(self.f.is_object(arg), self.f.object_keys(arg), NULL)

    @visit_node.register(ops.MapValues)
    def visit_MapValues(self, op, *, arg):
        return self.if_(
            self.f.is_object(arg), self._call_udf("object_values", arg), NULL
        )

    @visit_node.register(ops.MapGet)
    def visit_MapGet(self, op, *, arg, key, default):
        dtype = op.dtype
        expr = self.f.coalesce(self.f.get(arg, key), self.f.to_variant(default))
        if dtype.is_json() or dtype.is_null():
            return expr
        return self.cast(expr, dtype)

    @visit_node.register(ops.MapContains)
    def visit_MapContains(self, op, *, arg, key):
        return self.f.array_contains(
            self.f.to_variant(key),
            self.if_(self.f.is_object(arg), self.f.object_keys(arg), NULL),
        )

    @visit_node.register(ops.MapMerge)
    def visit_MapMerge(self, op, *, left, right):
        return self.if_(
            sg.and_(self.f.is_object(left), self.f.is_object(right)),
            self._call_udf("object_merge", left, right),
            NULL,
        )

    @visit_node.register(ops.MapLength)
    def visit_MapLength(self, op, *, arg):
        return self.if_(
            self.f.is_object(arg), self.f.array_size(self.f.object_keys(arg)), NULL
        )

    @visit_node.register(ops.Log)
    def visit_Log(self, op, *, arg, base):
        return self.f.log(base, arg, dialect=self.dialect)

    @visit_node.register(ops.RandomScalar)
    def visit_RandomScalar(self, op):
        return self.f.uniform(
            self.f.to_double(0.0), self.f.to_double(1.0), self.f.random()
        )

    @visit_node.register(ops.ApproxMedian)
    def visit_ApproxMedian(self, op, *, arg, where):
        return self.agg.approx_percentile(arg, 0.5, where=where)

    @visit_node.register(ops.TimeDelta)
    def visit_TimeDelta(self, op, *, part, left, right):
        return self.f.timediff(part, right, left, dialect=self.dialect)

    @visit_node.register(ops.DateDelta)
    def visit_DateDelta(self, op, *, part, left, right):
        return self.f.datediff(part, right, left, dialect=self.dialect)

    @visit_node.register(ops.TimestampDelta)
    def visit_TimestampDelta(self, op, *, part, left, right):
        return self.f.timestampdiff(part, right, left, dialect=self.dialect)

    @visit_node.register(ops.IntegerRange)
    def visit_IntegerRange(self, op, *, start, stop, step):
        return self.if_(
            step.neq(0), self.f.array_generate_range(start, stop, step), self.f.array()
        )

    @visit_node.register(ops.StructColumn)
    def visit_StructColumn(self, op, *, names, values):
        return self.f.object_construct_keep_null(
            *itertools.chain.from_iterable(zip(names, values))
        )

    @visit_node.register(ops.StructField)
    def visit_StructField(self, op, *, arg, field):
        # TODO(cpcloud): why is coming in as an Alias?
        if isinstance(arg, sge.Alias):
            arg = arg.this
        return self.cast(self.f.get(arg, field), op.dtype)

    @visit_node.register(ops.RegexSearch)
    def visit_RegexSearch(self, op, *, arg, pattern):
        return self.f.regexp_instr(arg, pattern).neq(0)

    @visit_node.register(ops.TypeOf)
    def visit_TypeOf(self, op, *, arg):
        return self.f.typeof(self.f.to_variant(arg))

    @visit_node.register(ops.ArrayRepeat)
    def visit_ArrayRepeat(self, op, *, arg, times):
        return self._call_udf("array_repeat", arg, times)

    @visit_node.register(ops.ArrayUnion)
    def visit_ArrayUnion(self, op, *, left, right):
        return self.f.array_distinct(self.f.array_cat(left, right))

    @visit_node.register(ops.ArrayContains)
    def visit_ArrayContains(self, op, *, arg, other):
        return self.f.array_contains(self.f.to_variant(other), arg)

    @visit_node.register(ops.ArrayCollect)
    def visit_ArrayCollect(self, op, *, arg, where):
        return self.agg.array_agg(
            self.f.ifnull(arg, self.f.parse_json("null")), where=where
        )

    @visit_node.register(ops.ArrayConcat)
    def visit_ArrayConcat(self, op, *, arg):
        return reduce(self.f.array_cat, arg)

    @visit_node.register(ops.ArrayPosition)
    def visit_ArrayPosition(self, op, *, arg, other):
        # snowflake is zero-based here, so we don't need to subtract 1 from the
        # result
        return self.f.coalesce(
            self.f.array_position(self.f.to_variant(other), arg) + 1, 0
        )

    @visit_node.register(ops.RegexExtract)
    def visit_RegexExtract(self, op, *, arg, pattern, index):
        # https://docs.snowflake.com/en/sql-reference/functions/regexp_substr
        return self.f.regexp_substr(arg, pattern, 1, 1, "ce", index)

    @visit_node.register(ops.ArrayZip)
    def visit_ArrayZip(self, op, *, arg):
        return self._call_udf("array_zip", self.f.array(*arg))

    @visit_node.register(ops.DayOfWeekName)
    def visit_DayOfWeekName(self, op, *, arg):
        return sge.Case(
            this=self.f.dayname(arg),
            ifs=[
                self.if_("Sun", "Sunday"),
                self.if_("Mon", "Monday"),
                self.if_("Tue", "Tuesday"),
                self.if_("Wed", "Wednesday"),
                self.if_("Thu", "Thursday"),
                self.if_("Fri", "Friday"),
                self.if_("Sat", "Saturday"),
            ],
            default=NULL,
        )

    @visit_node.register(ops.TimestampFromUNIX)
    def visit_TimestampFromUNIX(self, op, *, arg, unit):
        timestamp_units_to_scale = {"s": 0, "ms": 3, "us": 6, "ns": 9}
        return self.f.to_timestamp(arg, timestamp_units_to_scale[unit.short])

    @visit_node.register(ops.First)
    def visit_First(self, op, *, arg, where):
        return self.f.get(self.agg.array_agg(arg, where=where), 0)

    @visit_node.register(ops.Last)
    def visit_Last(self, op, *, arg, where):
        expr = self.agg.array_agg(arg, where=where)
        offset = self.f.array_size(expr) - 1
        return self.f.get(expr, offset)

    @visit_node.register(ops.GroupConcat)
    def visit_GroupConcat(self, op, *, arg, where, sep):
        if where is None:
            return self.f.listagg(arg, sep)

        return self.if_(
            self.f.count_if(where) > 0,
            self.f.listagg(self.if_(where, arg, None), sep),
            NULL,
        )

    @visit_node.register(ops.TimestampBucket)
    def visit_TimestampBucket(self, op, *, arg, interval, offset):
        if offset is not None:
            raise com.UnsupportedOperationError(
                "`offset` is not supported in the Snowflake backend for timestamp bucketing"
            )

        interval = op.interval
        if not isinstance(interval, ops.Literal):
            raise com.UnsupportedOperationError(
                f"Interval must be a literal for the Snowflake backend, got {type(interval)}"
            )

        return self.f.time_slice(arg, interval.value, interval.dtype.unit.name)

    @visit_node.register(ops.Arbitrary)
    def visit_Arbitrary(self, op, *, arg, how, where):
        if how == "first":
            return self.f.get(self.agg.array_agg(arg, where=where), 0)
        elif how == "last":
            expr = self.agg.array_agg(arg, where=where)
            return self.f.get(expr, self.f.array_size(expr) - 1)
        else:
            raise com.UnsupportedOperationError("how must be 'first' or 'last'")

    @visit_node.register(ops.ArraySlice)
    def visit_ArraySlice(self, op, *, arg, start, stop):
        if start is None:
            start = 0

        if stop is None:
            stop = self.f.array_size(arg)
        return self.f.array_slice(arg, start, stop)

    @visit_node.register(ops.ExtractEpochSeconds)
    def visit_ExtractExtractEpochSeconds(self, op, *, arg):
        return self.f.extract("epoch", arg)

    @visit_node.register(ops.ExtractMicrosecond)
    def visit_ExtractMicrosecond(self, op, *, arg):
        return self.cast(self.f.extract("epoch_microsecond", arg) % 1_000_000, op.dtype)

    @visit_node.register(ops.ExtractMillisecond)
    def visit_ExtractMillisecond(self, op, *, arg):
        return self.cast(self.f.extract("epoch_millisecond", arg) % 1_000, op.dtype)

    @visit_node.register(ops.ExtractQuery)
    def visit_ExtractQuery(self, op, *, arg, key):
        parsed_url = self.f.parse_url(arg, 1)
        if key is not None:
            r = self.f.get(self.f.get(parsed_url, "parameters"), key)
        else:
            r = self.f.get(parsed_url, "query")
        return self.f.nullif(self.f.as_varchar(r), "")

    @visit_node.register(ops.ExtractProtocol)
    def visit_ExtractProtocol(self, op, *, arg):
        return self.f.nullif(
            self.f.as_varchar(self.f.get(self.f.parse_url(arg, 1), "scheme")), ""
        )

    @visit_node.register(ops.ExtractAuthority)
    def visit_ExtractAuthority(self, op, *, arg):
        return self.f.concat_ws(
            ":",
            self.f.as_varchar(self.f.get(self.f.parse_url(arg, 1), "host")),
            self.f.as_varchar(self.f.get(self.f.parse_url(arg, 1), "port")),
        )

    @visit_node.register(ops.ExtractFile)
    def visit_ExtractFile(self, op, *, arg):
        return self.f.concat_ws(
            "?",
            self.visit_ExtractPath(op, arg=arg),
            self.visit_ExtractQuery(op, arg=arg, key=None),
        )

    @visit_node.register(ops.ExtractPath)
    def visit_ExtractPath(self, op, *, arg):
        return self.f.concat(
            "/", self.f.as_varchar(self.f.get(self.f.parse_url(arg, 1), "path"))
        )

    @visit_node.register(ops.ExtractFragment)
    def visit_ExtractFragment(self, op, *, arg):
        return self.f.nullif(
            self.f.as_varchar(self.f.get(self.f.parse_url(arg, 1), "fragment")), ""
        )

    @visit_node.register(ops.Unnest)
    def visit_Unnest(self, op, *, arg):
        sep = sge.convert(util.guid())
        split = self.f.split(self.f.array_to_string(arg, sep), sep)
        expr = self.f.nullif(sge.Explode(this=split), "")
        return self.cast(expr, op.dtype)

    @visit_node.register(ops.StringJoin)
    def visit_StringJoin(self, op, *, arg, sep):
        return self.f.array_to_string(self.f.array(*arg), sep)

    @visit_node.register(ops.Quantile)
    def visit_Quantile(self, op, *, arg, quantile, where):
        # can't use `self.agg` here because `quantile` must be a constant and
        # the agg method filters using `where` for every argument which turns
        # the constant into an expression
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return self.f.percentile_cont(arg, quantile)

    @visit_node.register(ops.CountStar)
    def visit_CountStar(self, op, *, arg, where):
        if where is None:
            return super().visit_node(op, arg=arg, where=where)
        return self.f.count_if(where)

    @visit_node.register(ops.CountDistinct)
    def visit_CountDistinct(self, op, *, arg, where):
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return self.f.count(sge.Distinct(expressions=[arg]))

    @visit_node.register(ops.CountDistinctStar)
    def visit_CountDistinctStar(self, op, *, arg, where):
        if where is None:
            expressions = [
                sg.column(name, quoted=self.quoted) for name in op.arg.schema.names
            ]
        else:
            # any null columns will cause the entire row not to be counted
            expressions = [
                self.if_(where, sg.column(name, quoted=self.quoted), NULL)
                for name in op.arg.schema.names
            ]
        return self.f.count(sge.Distinct(expressions=expressions))

    @visit_node.register(ops.Xor)
    def visit_Xor(self, op, *, left, right):
        # boolxor accepts numerics ... and returns a boolean? wtf?
        return self.f.boolxor(self.cast(left, dt.int8), self.cast(right, dt.int8))

    @visit_node.register(ops.Limit)
    def visit_Limit(self, op, *, parent, n, offset):
        result = sg.select(STAR).from_(parent)

        if isinstance(n, int):
            result = result.limit(n)
        elif n is not None:
            result = result.limit(sg.select(n).from_(parent).subquery())
        else:
            assert n is None, n
            result = result.limit(NULL)

        assert offset is not None, "offset is None"

        if not isinstance(offset, int):
            skip = offset
            skip = sg.select(skip).from_(parent).subquery()
        elif not offset:
            return result
        else:
            skip = offset

        return result.offset(skip)

    @visit_node.register(ops.ArrayMap)
    @visit_node.register(ops.ArrayFilter)
    @visit_node.register(ops.RowID)
    @visit_node.register(ops.MultiQuantile)
    @visit_node.register(ops.IntervalFromInteger)
    @visit_node.register(ops.IntervalAdd)
    @visit_node.register(ops.TimestampDiff)
    @visit_node.register(ops.TryCast)
    def visit_Undefined(self, op, **_):
        raise com.OperationNotDefinedError(type(op).__name__)

    @visit_node.register(ops.WindowBoundary)
    def visit_WindowBoundary(self, op, *, value, preceding):
        if not isinstance(op.value, ops.Literal):
            raise com.OperationNotDefinedError(
                "Expressions in window bounds are not supported by Snowflake"
            )
        return super().visit_WindowBoundary(op, value=value, preceding=preceding)


_SIMPLE_OPS = {
    ops.Mode: "mode",
    ops.Translate: "translate",
    ops.TimeFromHMS: "time_from_parts",
    ops.ArrayIndex: "get",
    ops.ArrayLength: "array_size",
    ops.ArrayDistinct: "array_distinct",
    ops.ArrayRemove: "array_remove",
    ops.ArrayIntersect: "array_intersection",
    ops.ArraySort: "array_sort",
    ops.ArrayFlatten: "array_flatten",
    ops.StringSplit: "split",
    ops.All: "booland_agg",
    ops.Any: "boolor_agg",
    ops.BitAnd: "bitand_agg",
    ops.BitOr: "bitor_agg",
    ops.BitXor: "bitxor_agg",
    ops.DateFromYMD: "date_from_parts",
    ops.StringToTimestamp: "to_timestamp_tz",
    ops.RegexReplace: "regexp_replace",
    ops.ArgMin: "min_by",
    ops.ArgMax: "max_by",
    ops.StartsWith: "startswith",
    ops.EndsWith: "endswith",
    ops.Hash: "hash",
    ops.Median: "median",
    ops.Levenshtein: "editdistance",
    ops.TimestampFromYMDHMS: "timestamp_from_parts",
    ops.BitwiseAnd: "bitand",
    ops.BitwiseOr: "bitor",
    ops.BitwiseXor: "bitxor",
    ops.BitwiseNot: "bitnot",
    ops.BitwiseLeftShift: "bitshiftleft",
    ops.BitwiseRightShift: "bitshiftright",
    ops.LPad: "lpad",
    ops.RPad: "rpad",
    ops.Reverse: "reverse",
    ops.StringAscii: "ascii",
    ops.StringReplace: "replace",
    ops.Strftime: "to_varchar",
}

for _op, _name in _SIMPLE_OPS.items():
    assert isinstance(type(_op), type), type(_op)
    if issubclass(_op, ops.Reduction):

        @SnowflakeCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, where, **kw):
            return self.agg[_name](*kw.values(), where=where)

    else:

        @SnowflakeCompiler.visit_node.register(_op)
        def _fmt(self, op, *, _name: str = _name, **kw):
            return self.f[_name](*kw.values())

    setattr(SnowflakeCompiler, f"visit_{_op.__name__}", _fmt)


del _op, _name, _fmt
