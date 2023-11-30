from __future__ import annotations

import itertools
from functools import reduce, singledispatchmethod

import sqlglot as sg
from public import public

import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.base.sqlglot.compiler import NULL, SQLGlotCompiler
from ibis.backends.snowflake.datatypes import SnowflakeType


@public
class SnowflakeCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = "snowflake"
    quoted = True
    type_mapper = SnowflakeType

    def _aggregate(self, funcname: str, *args, where):
        if where is not None:
            args = [self.if_(where, arg, NULL) for arg in args]

        func = self.f[funcname]
        return func(*args)

    @singledispatchmethod
    def visit_node(self, op, **kwargs):
        return super().visit_node(op, **kwargs)

    @visit_node.register(ops.Literal)
    def visit_Literal(self, op, *, value, dtype):
        if value is None:
            return super().visit_Literal(op, value=value, dtype=dtype)
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
        elif dtype.is_map() or dtype.is_struct():
            # TODO: handle conversion of keys and values to expressions
            return self.f.object_construct_keep_null(
                *itertools.chain.from_iterable(value.items())
            )
        elif dtype.is_uuid():
            return sg.exp.convert(str(value))
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

    @visit_node.register(ops.Log2)
    def visit_Log2(self, op, *, arg):
        return self.f.log(2, arg)

    @visit_node.register(ops.Log10)
    def visit_Log10(self, op, *, arg):
        return self.f.log(10, arg)

    @visit_node.register(ops.Log)
    def visit_Log(self, op, *, arg, base):
        return self.f.log(base, arg, dialect=self.dialect)

    @visit_node.register(ops.RandomScalar)
    def visit_RandomScalar(self, op):
        return self.f.uniform(
            self.f.to_double(0.0), self.f.to_double(1.0), self.f.random()
        )

    @visit_node.register(ops.ToJSONArray)
    @visit_node.register(ops.ToJSONMap)
    def visit_ToJSON(self, op, *, arg):
        return self.visit_Cast(ops.Cast(op.arg, to=op.dtype), arg=arg, to=op.dtype)

    @visit_node.register(ops.ApproxMedian)
    def visit_ApproxMedian(self, op, *, arg):
        return self.f.approx_percentile(arg, 0.5)

    @visit_node.register(ops.TimeDelta)
    def visit_TimeDelta(self, op, *, part, left, right):
        return self.f.timediff(part, right, left)

    @visit_node.register(ops.DateDelta)
    def visit_DateDelta(self, op, *, part, left, right):
        return self.f.datediff(part, right, left)

    @visit_node.register(ops.TimestampDelta)
    def visit_TimestampDelta(self, op, *, part, left, right):
        return self.f.timestampdiff(part, right, left)

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
        return sg.exp.Case(
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

        arg = self.if_(where, arg, None)

        return self.if_(
            self.f.count_if(arg.is_(sg.not_(NULL))).neq(0),
            self.f.listagg(arg, sep),
            NULL,
        )

    @visit_node.register(ops.TimestampBucket)
    def visit_TimestampBucket(self, op, *, arg, interval, offset):
        if offset is not None:
            raise com.UnsupportedOperationError(
                "`offset` is not supported in the Snowflake backend for timestamp bucketing"
            )

        interval = op.interval
        if not isinstance(interval, sg.exp.Literal):
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
    def visit_ArraySclie(self, op, *, arg, start, stop):
        if start is None:
            start = 0

        if stop is None:
            stop = self.f.array_size(arg)
        return self.f.array_slice(arg, start, stop)

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
        return "/" + self.f.as_varchar(self.f.get(self.f.parse_url(arg, 1), "path"))

    @visit_node.register(ops.ExtractFragment)
    def visit_ExtractFragment(self, op, *, arg):
        return self.f.nullif(
            self.f.as_varchar(self.f.get(self.f.parse_url(arg, 1), "fragment")), ""
        )

    @visit_node.register(ops.Unnest)
    def visit_Unnest(self, op, *, arg):
        return sg.exp.Explode(this=arg)

    @visit_node.register(ops.StringJoin)
    def visit_StringJoin(self, op, *, arg, sep):
        return self.f.array_to_string(self.f.array(*arg), sep)

    @visit_node.register(ops.ArrayMap)
    @visit_node.register(ops.ArrayFilter)
    @visit_node.register(ops.RowID)
    @visit_node.register(ops.Translate)
    def visit_Undefined(self, op, **_):
        raise com.OperationNotDefinedError(type(op).__name__)


_SIMPLE_OPS = {
    ops.Mode: "mode",
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
    ops.ApproxCountDistinct: "approx_count_distinct",
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
