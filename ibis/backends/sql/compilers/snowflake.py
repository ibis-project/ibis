from __future__ import annotations

import itertools
from functools import partial

import sqlglot as sg
import sqlglot.expressions as sge

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis import util
from ibis.backends.sql.compilers.base import NULL, STAR, C, FuncGen, SQLGlotCompiler
from ibis.backends.sql.datatypes import SnowflakeType
from ibis.backends.sql.dialects import Snowflake
from ibis.backends.sql.rewrites import (
    exclude_unsupported_window_frame_from_ops,
    exclude_unsupported_window_frame_from_row_number,
    lower_log2,
    lower_log10,
    rewrite_empty_order_by_window,
)


class SnowflakeFuncGen(FuncGen):
    udf = FuncGen(namespace="ibis_udfs.public")


class SnowflakeCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = Snowflake
    type_mapper = SnowflakeType
    no_limit_value = NULL
    rewrites = (
        exclude_unsupported_window_frame_from_row_number,
        exclude_unsupported_window_frame_from_ops,
        rewrite_empty_order_by_window,
        *SQLGlotCompiler.rewrites,
    )

    LOWERED_OPS = {
        ops.Log2: lower_log2,
        ops.Log10: lower_log10,
        ops.Sample: None,
    }

    UNSUPPORTED_OPS = (
        ops.RowID,
        ops.MultiQuantile,
        ops.IntervalFromInteger,
        ops.IntervalAdd,
        ops.TimestampDiff,
    )

    SIMPLE_OPS = {
        ops.All: "min",
        ops.Any: "max",
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
        ops.ExtractIsoYear: "yearofweekiso",
        ops.Hash: "hash",
        ops.Median: "median",
        ops.Mode: "mode",
        ops.StringToDate: "to_date",
        ops.StringToTimestamp: "to_timestamp_tz",
        ops.TimeFromHMS: "time_from_parts",
        ops.TimestampFromYMDHMS: "timestamp_from_parts",
    }

    def __init__(self):
        super().__init__()
        self.f = SnowflakeFuncGen()

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

    def visit_Arbitrary(self, op, *, arg, where):
        return self.f.get(self.agg.array_agg(arg, where=where), 0)

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

    def visit_UnwrapJSONString(self, op, *, arg):
        return self.if_(self.f.is_varchar(arg), self.f.as_varchar(arg), NULL)

    def visit_UnwrapJSONInt64(self, op, *, arg):
        return self.if_(self.f.is_integer(arg), self.f.as_integer(arg), NULL)

    def visit_UnwrapJSONFloat64(self, op, *, arg):
        return self.if_(self.f.is_double(arg), self.f.as_double(arg), NULL)

    def visit_UnwrapJSONBoolean(self, op, *, arg):
        return self.if_(self.f.is_boolean(arg), self.f.as_boolean(arg), NULL)

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
            self.f.arrays_to_object(keys, values),
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

    def visit_RandomScalar(self, op, **kwargs):
        return self.f.uniform(
            self.f.to_double(0.0), self.f.to_double(1.0), self.f.random()
        )

    def visit_RandomUUID(self, op, **kwargs):
        return self.f.uuid_string()

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
        return self.if_(
            sg.not_(sg.or_(*(arr.is_(NULL) for arr in arg))),
            self.f.udf.array_zip(self.f.array(*arg)),
            NULL,
        )

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

        # The Snowflake SQLGlot dialect rewrites calls to `percentile_cont` to
        # include     WITHIN GROUP (ORDER BY ...)
        # as per https://docs.snowflake.com/en/sql-reference/functions/percentile_cont
        # using the rule `add_within_group_for_percentiles`
        #
        # If we have copy=False set in our call to `compile`, if there is more
        # than one quantile, the rewrite rule fails on the second pass because
        # of some mutation in the first pass. To avoid this error, we create the
        # expression with the within group included already and skip the (now
        # unneeded) rewrite rule.
        order_by = sge.Order(expressions=[sge.Ordered(this=arg)])
        quantile = self.f.percentile_cont(quantile)
        return sge.WithinGroup(this=quantile, expression=order_by)

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

    def visit_WindowFunction(self, op, *, how, func, start, end, group_by, order_by):
        if start is None:
            start = {}
        if end is None:
            end = {}

        start_value = start.get("value", "UNBOUNDED")
        start_side = start.get("side", "PRECEDING")
        end_value = end.get("value", "UNBOUNDED")
        end_side = end.get("side", "FOLLOWING")

        if getattr(start_value, "this", None) == "0":
            start_value = "CURRENT ROW"
            start_side = None

        if getattr(end_value, "this", None) == "0":
            end_value = "CURRENT ROW"
            end_side = None

        spec = sge.WindowSpec(
            kind=how.upper(),
            start=start_value,
            start_side=start_side,
            end=end_value,
            end_side=end_side,
            over="OVER",
        )
        order = sge.Order(expressions=order_by) if order_by else None

        orig_spec = spec
        spec = self._minimize_spec(op.start, op.end, orig_spec)

        # due to https://docs.snowflake.com/en/sql-reference/functions-analytic#window-frame-usage-notes
        # we need to make the default window rows (since range isn't supported)
        # and we need to make the default frame unbounded preceding to current
        # row
        if spec is None and isinstance(op.func, (ops.First, ops.Last, ops.NthValue)):
            spec = orig_spec
            spec.args["kind"] = "ROWS"

        return sge.Window(this=func, partition_by=group_by, order=order, spec=spec)

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

    def visit_Sample(
        self, op, *, parent, fraction: float, method: str, seed: int | None, **_
    ):
        sample = sge.TableSample(
            this=parent,
            method="bernoulli" if method == "row" else "system",
            percent=sge.convert(fraction * 100.0),
            seed=None if seed is None else sge.convert(seed),
        )
        return sg.select(STAR).from_(sample)

    def visit_ArrayMap(self, op, *, arg, param, body):
        return self.f.transform(arg, sge.Lambda(this=body, expressions=[param]))

    def visit_ArrayFilter(self, op, *, arg, param, body):
        return self.f.filter(
            arg,
            sge.Lambda(
                this=sg.and_(
                    body,
                    # necessary otherwise null values are treated as JSON nulls
                    # instead of SQL NULLs
                    self.cast(sg.to_identifier(param), op.dtype.value_type).is_(
                        sg.not_(NULL)
                    ),
                ),
                expressions=[param],
            ),
        )

    def visit_JoinLink(self, op, *, how, table, predicates):
        assert (
            predicates or how == "cross"
        ), "expected non-empty predicates when not a cross join"

        if how == "asof":
            # the asof join match condition is always the first predicate by
            # construction
            match_condition, *predicates = predicates
            on = sg.and_(*predicates) if predicates else None
            return sge.Join(
                this=table, kind=how, on=on, match_condition=match_condition
            )
        return super().visit_JoinLink(op, how=how, table=table, predicates=predicates)

    def visit_DropColumns(self, op, *, parent, columns_to_drop):
        quoted = self.quoted
        excludes = [sg.column(column, quoted=quoted) for column in columns_to_drop]
        star = sge.Star(**{"except": excludes})
        table = sg.to_identifier(parent.alias_or_name, quoted=quoted)
        column = sge.Column(this=star, table=table)
        return sg.select(column).from_(parent)

    def visit_TableUnnest(
        self, op, *, parent, column, offset: str | None, keep_empty: bool
    ):
        quoted = self.quoted

        column_alias = sg.to_identifier(
            util.gen_name("table_unnest_column"), quoted=quoted
        )

        sep = sge.convert(util.guid())
        null_sentinel = sge.convert(util.guid())

        table = sg.to_identifier(parent.alias_or_name, quoted=quoted)

        selcols = []

        opcol = op.column
        opname = opcol.name
        overlaps_with_parent = opname in op.parent.schema
        computed_column = self.cast(
            self.f.nullif(column_alias, null_sentinel), opcol.dtype.value_type
        ).as_(opname, quoted=quoted)

        if overlaps_with_parent:
            selcols.append(
                sge.Column(this=sge.Star(replace=[computed_column]), table=table)
            )
        else:
            selcols.append(sge.Column(this=STAR, table=table))
            selcols.append(computed_column)

        if offset is not None:
            offset = sg.to_identifier(offset, quoted=quoted)
            selcols.append(offset)

        alias = sge.TableAlias(
            this=sg.to_identifier(util.gen_name("table_unnest"), quoted=quoted),
            columns=[column_alias],
        )

        # there has to be a better way
        param = sg.to_identifier(util.gen_name("table_unnest_param"))
        column = self.f.transform(
            column,
            sge.Lambda(
                this=self.f.coalesce(self.cast(param, dt.string), null_sentinel),
                expressions=[param],
            ),
        )
        empty_array = self.f.array()
        split = self.f.coalesce(
            self.f.nullif(
                self.f.split(
                    self.f.array_to_string(self.f.nullif(column, empty_array), sep), sep
                ),
                empty_array,
            ),
            self.f.array(null_sentinel),
        )

        unnest = sge.Unnest(expressions=[split], alias=alias, offset=offset)
        return (
            sg.select(*selcols)
            .from_(parent)
            .join(unnest, join_type="CROSS" if not keep_empty else "LEFT")
        )
