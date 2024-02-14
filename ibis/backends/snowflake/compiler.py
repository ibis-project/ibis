from __future__ import annotations

import itertools
from functools import partial

import sqlglot as sg
import sqlglot.expressions as sge
from public import public

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis import util
from ibis.backends.base.sqlglot.compiler import NULL, C, FuncGen, SQLGlotCompiler
from ibis.backends.base.sqlglot.datatypes import SnowflakeType
from ibis.backends.base.sqlglot.dialects import Snowflake
from ibis.backends.base.sqlglot.rewrites import (
    exclude_unsupported_window_frame_from_ops,
    exclude_unsupported_window_frame_from_row_number,
    replace_log2,
    replace_log10,
    rewrite_empty_order_by_window,
    rewrite_first_to_first_value,
    rewrite_last_to_last_value,
)


class SnowflakeFuncGen(FuncGen):
    udf = FuncGen(namespace="ibis_udfs.public")


@public
class SnowflakeCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = Snowflake
    type_mapper = SnowflakeType
    no_limit_value = NULL
    rewrites = (
        exclude_unsupported_window_frame_from_row_number,
        exclude_unsupported_window_frame_from_ops,
        rewrite_first_to_first_value,
        rewrite_last_to_last_value,
        rewrite_empty_order_by_window,
        replace_log2,
        replace_log10,
        *SQLGlotCompiler.rewrites,
    )

    UNSUPPORTED_OPERATIONS = frozenset(
        (
            ops.ArrayMap,
            ops.ArrayFilter,
            ops.RowID,
            ops.MultiQuantile,
            ops.IntervalFromInteger,
            ops.IntervalAdd,
            ops.TimestampDiff,
        )
    )

    SIMPLE_OPS = {
        ops.Any: "max",
        ops.All: "min",
        ops.ArrayDistinct: "array_distinct",
        ops.ArrayFlatten: "array_flatten",
        ops.ArrayIndex: "get",
        ops.ArrayIntersect: "array_intersection",
        ops.ArrayRemove: "array_remove",
        ops.BitAnd: "bitand_agg",
        ops.BitOr: "bitor_agg",
        ops.BitXor: "bitxor_agg",
        ops.BitwiseAnd: "bitand",
        ops.BitwiseLeftShift: "bitshiftleft",
        ops.BitwiseNot: "bitnot",
        ops.BitwiseOr: "bitor",
        ops.BitwiseRightShift: "bitshiftright",
        ops.BitwiseXor: "bitxor",
        ops.EndsWith: "endswith",
        ops.Hash: "hash",
        ops.Median: "median",
        ops.Mode: "mode",
        ops.StringToTimestamp: "to_timestamp_tz",
        ops.TimeFromHMS: "time_from_parts",
        ops.TimestampFromYMDHMS: "timestamp_from_parts",
    }

    def __init__(self):
        super().__init__()
        self.f = SnowflakeFuncGen()

    def _aggregate(self, funcname: str, *args, where):
        if where is not None:
            args = [self.if_(where, arg, NULL) for arg in args]

        func = self.f[funcname]
        return func(*args)

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
                # workaround sqlglot not supporting more than 6 arguments by
                # using an anonymous function
                return self.f.anon.timestamp_from_parts(*args)
        elif dtype.is_time():
            nanos = value.microsecond * 1_000
            return self.f.time_from_parts(value.hour, value.minute, value.second, nanos)
        elif dtype.is_map():
            key_type = dtype.key_type
            value_type = dtype.value_type

            pairs = []

            for k, v in value.items():
                pairs.append(
                    self.visit_Literal(
                        ops.Literal(k, key_type), value=k, dtype=key_type
                    )
                )
                pairs.append(
                    self.visit_Literal(
                        ops.Literal(v, value_type), value=v, dtype=value_type
                    )
                )

            return self.f.object_construct_keep_null(*pairs)
        elif dtype.is_struct():
            pairs = []
            for k, v in value.items():
                pairs.append(k)
                pairs.append(
                    self.visit_Literal(
                        ops.Literal(v, dtype[k]), value=v, dtype=dtype[k]
                    )
                )
            return self.f.object_construct_keep_null(*pairs)
        elif dtype.is_uuid():
            return sge.convert(str(value))
        elif dtype.is_binary():
            return sge.HexString(this=value.hex())
        return super().visit_Literal(op, value=value, dtype=dtype)

    def visit_Cast(self, op, *, arg, to):
        if to.is_struct() or to.is_map():
            return self.if_(self.f.is_object(arg), arg, NULL)
        elif to.is_array():
            return self.if_(self.f.is_array(arg), arg, NULL)
        return self.cast(arg, to)

    def visit_ToJSONMap(self, op, *, arg):
        return self.if_(self.f.is_object(arg), arg, NULL)

    def visit_ToJSONArray(self, op, *, arg):
        return self.if_(self.f.is_array(arg), arg, NULL)

    def visit_IsNan(self, op, *, arg):
        return arg.eq(self.NAN)

    def visit_IsInf(self, op, *, arg):
        return arg.isin(self.POS_INF, self.NEG_INF)

    def visit_JSONGetItem(self, op, *, arg, index):
        return self.f.get(arg, index)

    def visit_StringFind(self, op, *, arg, substr, start, end):
        args = [substr, arg]
        if start is not None:
            args.append(start + 1)
        return self.f.position(*args)

    def visit_RegexSplit(self, op, *, arg, pattern):
        return self.f.udf.regexp_split(arg, pattern)

    def visit_Map(self, op, *, keys, values):
        return self.if_(
            sg.and_(self.f.is_array(keys), self.f.is_array(values)),
            self.f.udf.object_from_arrays(keys, values),
            NULL,
        )

    def visit_MapKeys(self, op, *, arg):
        return self.if_(self.f.is_object(arg), self.f.object_keys(arg), NULL)

    def visit_MapValues(self, op, *, arg):
        return self.if_(self.f.is_object(arg), self.f.udf.object_values(arg), NULL)

    def visit_MapGet(self, op, *, arg, key, default):
        dtype = op.dtype
        expr = self.f.coalesce(self.f.get(arg, key), self.f.to_variant(default))
        if dtype.is_json() or dtype.is_null():
            return expr
        return self.cast(expr, dtype)

    def visit_MapContains(self, op, *, arg, key):
        return self.f.array_contains(
            self.if_(self.f.is_object(arg), self.f.object_keys(arg), NULL),
            self.f.to_variant(key),
        )

    def visit_MapMerge(self, op, *, left, right):
        return self.if_(
            sg.and_(self.f.is_object(left), self.f.is_object(right)),
            self.f.udf.object_merge(left, right),
            NULL,
        )

    def visit_MapLength(self, op, *, arg):
        return self.if_(
            self.f.is_object(arg), self.f.array_size(self.f.object_keys(arg)), NULL
        )

    def visit_Log(self, op, *, arg, base):
        return self.f.log(base, arg, dialect=self.dialect)

    def visit_RandomScalar(self, op):
        return self.f.uniform(
            self.f.to_double(0.0), self.f.to_double(1.0), self.f.random()
        )

    def visit_ApproxMedian(self, op, *, arg, where):
        return self.agg.approx_percentile(arg, 0.5, where=where)

    def visit_TimeDelta(self, op, *, part, left, right):
        return self.f.timediff(part, right, left, dialect=self.dialect)

    def visit_DateDelta(self, op, *, part, left, right):
        return self.f.datediff(part, right, left, dialect=self.dialect)

    def visit_TimestampDelta(self, op, *, part, left, right):
        return self.f.timestampdiff(part, right, left, dialect=self.dialect)

    def visit_TimestampDateAdd(self, op, *, left, right):
        if not isinstance(op.right, ops.Literal):
            raise com.OperationNotDefinedError(
                f"right side of {type(op).__name__} operation must be an interval literal"
            )
        return sg.exp.Add(this=left, expression=right)

    visit_DateAdd = visit_TimestampAdd = visit_TimestampDateAdd

    def visit_IntegerRange(self, op, *, start, stop, step):
        return self.if_(
            step.neq(0), self.f.array_generate_range(start, stop, step), self.f.array()
        )

    def visit_StructColumn(self, op, *, names, values):
        return self.f.object_construct_keep_null(
            *itertools.chain.from_iterable(zip(names, values))
        )

    def visit_StructField(self, op, *, arg, field):
        return self.cast(self.f.get(arg, field), op.dtype)

    def visit_RegexSearch(self, op, *, arg, pattern):
        return sge.RegexpLike(
            this=arg,
            expression=self.f.concat(".*", pattern, ".*"),
            flag=sge.convert("cs"),
        )

    def visit_RegexReplace(self, op, *, arg, pattern, replacement):
        return sge.RegexpReplace(this=arg, expression=pattern, replacement=replacement)

    def visit_TypeOf(self, op, *, arg):
        return self.f.typeof(self.f.to_variant(arg))

    def visit_ArrayRepeat(self, op, *, arg, times):
        return self.f.udf.array_repeat(arg, times)

    def visit_ArrayUnion(self, op, *, left, right):
        return self.f.array_distinct(self.f.array_cat(left, right))

    def visit_ArrayContains(self, op, *, arg, other):
        return self.f.array_contains(arg, self.f.to_variant(other))

    def visit_ArrayCollect(self, op, *, arg, where):
        return self.agg.array_agg(
            self.f.ifnull(arg, self.f.parse_json("null")), where=where
        )

    def visit_ArrayConcat(self, op, *, arg):
        # array_cat only accepts two arguments
        return self.f.array_flatten(self.f.array(*arg))

    def visit_ArrayPosition(self, op, *, arg, other):
        # snowflake is zero-based here, so we don't need to subtract 1 from the
        # result
        return self.f.coalesce(
            self.f.array_position(self.f.to_variant(other), arg) + 1, 0
        )

    def visit_RegexExtract(self, op, *, arg, pattern, index):
        # https://docs.snowflake.com/en/sql-reference/functions/regexp_substr
        return sge.RegexpExtract(
            this=arg,
            expression=pattern,
            position=sge.convert(1),
            group=index,
            parameters=sge.convert("ce"),
        )

    def visit_ArrayZip(self, op, *, arg):
        return self.f.udf.array_zip(self.f.array(*arg))

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

    def visit_TimestampFromUNIX(self, op, *, arg, unit):
        timestamp_units_to_scale = {"s": 0, "ms": 3, "us": 6, "ns": 9}
        return self.f.to_timestamp(arg, timestamp_units_to_scale[unit.short])

    def visit_First(self, op, *, arg, where):
        return self.f.get(self.agg.array_agg(arg, where=where), 0)

    def visit_Last(self, op, *, arg, where):
        expr = self.agg.array_agg(arg, where=where)
        return self.f.get(expr, self.f.array_size(expr) - 1)

    def visit_GroupConcat(self, op, *, arg, where, sep):
        if where is None:
            return self.f.listagg(arg, sep)

        return self.if_(
            self.f.count_if(where) > 0,
            self.f.listagg(self.if_(where, arg, NULL), sep),
            NULL,
        )

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

    def visit_Arbitrary(self, op, *, arg, how, where):
        if how == "first":
            return self.f.get(self.agg.array_agg(arg, where=where), 0)
        elif how == "last":
            expr = self.agg.array_agg(arg, where=where)
            return self.f.get(expr, self.f.array_size(expr) - 1)
        else:
            raise com.UnsupportedOperationError("how must be 'first' or 'last'")

    def visit_ArraySlice(self, op, *, arg, start, stop):
        if start is None:
            start = 0

        if stop is None:
            stop = self.f.array_size(arg)
        return self.f.array_slice(arg, start, stop)

    def visit_ExtractEpochSeconds(self, op, *, arg):
        return self.f.extract("epoch", arg)

    def visit_ExtractMicrosecond(self, op, *, arg):
        return self.f.extract("epoch_microsecond", arg) % 1_000_000

    def visit_ExtractMillisecond(self, op, *, arg):
        return self.f.extract("epoch_millisecond", arg) % 1_000

    def visit_ExtractQuery(self, op, *, arg, key):
        parsed_url = self.f.parse_url(arg, 1)
        if key is not None:
            r = self.f.get(self.f.get(parsed_url, "parameters"), key)
        else:
            r = self.f.get(parsed_url, "query")
        return self.f.nullif(self.f.as_varchar(r), "")

    def visit_ExtractProtocol(self, op, *, arg):
        return self.f.nullif(
            self.f.as_varchar(self.f.get(self.f.parse_url(arg, 1), "scheme")), ""
        )

    def visit_ExtractAuthority(self, op, *, arg):
        return self.f.concat_ws(
            ":",
            self.f.as_varchar(self.f.get(self.f.parse_url(arg, 1), "host")),
            self.f.as_varchar(self.f.get(self.f.parse_url(arg, 1), "port")),
        )

    def visit_ExtractFile(self, op, *, arg):
        return self.f.concat_ws(
            "?",
            self.visit_ExtractPath(op, arg=arg),
            self.visit_ExtractQuery(op, arg=arg, key=None),
        )

    def visit_ExtractPath(self, op, *, arg):
        return self.f.concat(
            "/", self.f.as_varchar(self.f.get(self.f.parse_url(arg, 1), "path"))
        )

    def visit_ExtractFragment(self, op, *, arg):
        return self.f.nullif(
            self.f.as_varchar(self.f.get(self.f.parse_url(arg, 1), "fragment")), ""
        )

    def visit_Unnest(self, op, *, arg):
        sep = sge.convert(util.guid())
        split = self.f.split(
            self.f.array_to_string(self.f.nullif(arg, self.f.array()), sep), sep
        )
        expr = self.f.nullif(self.f.explode(split), "")
        return self.cast(expr, op.dtype)

    def visit_Quantile(self, op, *, arg, quantile, where):
        # can't use `self.agg` here because `quantile` must be a constant and
        # the agg method filters using `where` for every argument which turns
        # the constant into an expression
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return self.f.percentile_cont(arg, quantile)

    def visit_CountStar(self, op, *, arg, where):
        if where is None:
            return super().visit_CountStar(op, arg=arg, where=where)
        return self.f.count_if(where)

    def visit_CountDistinct(self, op, *, arg, where):
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return self.f.count(sge.Distinct(expressions=[arg]))

    def visit_CountDistinctStar(self, op, *, arg, where):
        columns = op.arg.schema.names
        quoted = self.quoted
        col = partial(sg.column, quoted=quoted)
        if where is None:
            expressions = list(map(col, columns))
        else:
            # any null columns will cause the entire row not to be counted
            expressions = [self.if_(where, col(name), NULL) for name in columns]
        return self.f.count(sge.Distinct(expressions=expressions))

    def visit_Xor(self, op, *, left, right):
        # boolxor accepts numerics ... and returns a boolean? wtf?
        return self.f.boolxor(self.cast(left, dt.int8), self.cast(right, dt.int8))

    def visit_WindowBoundary(self, op, *, value, preceding):
        if not isinstance(op.value, ops.Literal):
            raise com.OperationNotDefinedError(
                "Expressions in window bounds are not supported by Snowflake"
            )
        return super().visit_WindowBoundary(op, value=value, preceding=preceding)

    def visit_Correlation(self, op, *, left, right, how, where):
        if how == "sample":
            raise com.UnsupportedOperationError(
                f"{self.dialect} only implements `pop` correlation coefficient"
            )

        # TODO: rewrite rule?
        if (left_type := op.left.dtype).is_boolean():
            left = self.cast(left, dt.Int32(nullable=left_type.nullable))

        if (right_type := op.right.dtype).is_boolean():
            right = self.cast(right, dt.Int32(nullable=right_type.nullable))

        return self.agg.corr(left, right, where=where)

    def visit_TimestampRange(self, op, *, start, stop, step):
        raw_step = op.step

        if not isinstance(raw_step, ops.Literal):
            raise com.UnsupportedOperationError("`step` argument must be a literal")

        unit = raw_step.dtype.unit.name.lower()
        step = raw_step.value

        value_type = op.dtype.value_type

        if step == 0:
            return self.f.array()

        return (
            sg.select(
                self.f.array_agg(
                    self.f.replace(
                        # conversion to varchar is necessary to control
                        # the timestamp format
                        #
                        # otherwise, since timestamps in arrays become strings
                        # anyway due to lack of parameterized type support in
                        # Snowflake the format depends on a session parameter
                        self.f.to_varchar(
                            self.f.dateadd(unit, C.value, start, dialect=self.dialect),
                            'YYYY-MM-DD"T"HH24:MI:SS.FF6'
                            + (value_type.timezone is not None) * "TZH:TZM",
                        ),
                        # timezones are always hour:minute offsets from UTC, not
                        # named, so replacing "Z" shouldn't be an issue
                        "Z",
                        "+00:00",
                    ),
                )
            )
            .from_(
                sge.Table(
                    this=sge.Unnest(
                        expressions=[
                            self.f.array_generate_range(
                                0,
                                self.f.datediff(
                                    unit, start, stop, dialect=self.dialect
                                ),
                                step,
                            )
                        ]
                    )
                )
            )
            .subquery()
        )
