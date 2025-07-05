from __future__ import annotations

import math
import operator
from functools import partial, reduce

import sqlglot as sg
import sqlglot.expressions as sge
import toolz

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.sql.compilers.base import (
    FALSE,
    NULL,
    STAR,
    TRUE,
    AggGen,
    SQLGlotCompiler,
)
from ibis.backends.sql.datatypes import TrinoType
from ibis.backends.sql.dialects import Trino
from ibis.backends.sql.rewrites import (
    FirstValue,
    LastValue,
    exclude_unsupported_window_frame_from_ops,
    exclude_unsupported_window_frame_from_rank,
    exclude_unsupported_window_frame_from_row_number,
    lower_sample,
    split_select_distinct_with_order_by,
)
from ibis.util import gen_name


class TrinoCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = Trino
    type_mapper = TrinoType

    agg = AggGen(supports_filter=True, supports_order_by=True)

    rewrites = (
        exclude_unsupported_window_frame_from_row_number,
        exclude_unsupported_window_frame_from_rank,
        exclude_unsupported_window_frame_from_ops,
        *SQLGlotCompiler.rewrites,
    )
    post_rewrites = (split_select_distinct_with_order_by,)
    quoted = True

    NAN = sg.func("nan")
    POS_INF = sg.func("infinity")
    NEG_INF = -POS_INF

    UNSUPPORTED_OPS = (
        ops.Median,
        ops.RowID,
        ops.TimestampBucket,
        ops.StringToTime,
    )

    LOWERED_OPS = {
        ops.Sample: lower_sample(supports_seed=False),
    }

    SIMPLE_OPS = {
        ops.Arbitrary: "any_value",
        ops.Pi: "pi",
        ops.E: "e",
        ops.RegexReplace: "regexp_replace",
        ops.Map: "map",
        ops.MapKeys: "map_keys",
        ops.MapLength: "cardinality",
        ops.MapMerge: "map_concat",
        ops.MapValues: "map_values",
        ops.Log2: "log2",
        ops.Log10: "log10",
        ops.IsNan: "is_nan",
        ops.IsInf: "is_infinite",
        ops.StringToTimestamp: "date_parse",
        ops.Strftime: "date_format",
        ops.ExtractEpochSeconds: "to_unixtime",
        ops.ExtractWeekOfYear: "week_of_year",
        ops.ExtractDayOfYear: "day_of_year",
        ops.ExtractMillisecond: "millisecond",
        ops.ArrayUnion: "array_union",
        ops.ArrayRemove: "array_remove",
        ops.ArrayFlatten: "flatten",
        ops.ArraySort: "array_sort",
        ops.ArrayDistinct: "array_distinct",
        ops.ArrayLength: "cardinality",
        ops.ArrayIntersect: "array_intersect",
        ops.BitAnd: "bitwise_and_agg",
        ops.BitOr: "bitwise_or_agg",
        ops.BitXor: "bitwise_xor_agg",
        ops.TypeOf: "typeof",
        ops.Levenshtein: "levenshtein_distance",
        ops.ExtractProtocol: "url_extract_protocol",
        ops.ExtractHost: "url_extract_host",
        ops.ExtractPath: "url_extract_path",
        ops.ExtractFragment: "url_extract_fragment",
        ops.ArrayPosition: "array_position",
        ops.ExtractIsoYear: "year_of_week",
    }

    @staticmethod
    def _minimize_spec(op, spec):
        if isinstance(func := op.func, ops.Analytic) and not isinstance(
            func, (ops.First, ops.Last, FirstValue, LastValue, ops.NthValue)
        ):
            return None
        return spec

    def visit_Correlation(self, op, *, left, right, how, where):
        if how == "sample":
            raise com.UnsupportedOperationError(
                "Trino does not support `sample` correlation"
            )
        if (left_type := op.left.dtype).is_boolean():
            left = self.cast(left, dt.Int32(nullable=left_type.nullable))

        if (right_type := op.right.dtype).is_boolean():
            right = self.cast(right, dt.Int32(nullable=right_type.nullable))

        return self.agg.corr(left, right, where=where)

    def visit_ApproxQuantile(self, op, *, arg, quantile, where):
        if not op.arg.dtype.is_floating():
            arg = self.cast(arg, dt.float64)
        return self.agg.approx_quantile(arg, quantile, where=where)

    visit_ApproxMultiQuantile = visit_ApproxQuantile

    def visit_BitXor(self, op, *, arg, where):
        a, b = map(sg.to_identifier, "ab")
        input_fn = combine_fn = sge.Lambda(
            this=sge.BitwiseXor(this=a, expression=b), expressions=[a, b]
        )
        return self.agg.reduce_agg(arg, 0, input_fn, combine_fn, where=where)

    def visit_ArrayRepeat(self, op, *, arg, times):
        return self.f.flatten(self.f.repeat(arg, times))

    def visit_ArraySlice(self, op, *, arg, start, stop):
        def _neg_idx_to_pos(n, idx):
            return self.if_(idx < 0, n + self.f.greatest(idx, -n), idx)

        arg_length = self.f.cardinality(arg)

        if start is None:
            start = 0
        else:
            start = self.f.least(arg_length, _neg_idx_to_pos(arg_length, start))

        if stop is None:
            stop = arg_length
        else:
            stop = _neg_idx_to_pos(arg_length, stop)

        return self.f.slice(arg, start + 1, stop - start)

    def visit_ArrayMap(self, op, *, arg, param, body, index):
        if index is None:
            return self.f.transform(arg, sge.Lambda(this=body, expressions=[param]))
        else:
            return self.f.zip_with(
                arg,
                self.f.sequence(0, self.f.cardinality(arg) - 1),
                sge.Lambda(this=body, expressions=[param, index]),
            )

    def visit_ArrayFilter(self, op, *, arg, param, body, index):
        # no index, life is simpler
        if index is None:
            return self.f.filter(arg, sge.Lambda(this=body, expressions=[param]))

        placeholder = sg.to_identifier("__trino_filter__")
        index = sg.to_identifier(index)
        keep, value = map(sg.to_identifier, ("keep", "value"))

        # first, zip the array with the index and call the user's function,
        # returning a struct of {"keep": value-of-predicate, "value": array-element}
        zipped = self.f.zip_with(
            arg,
            # users are limited to 10_000 elements here because it
            # seems like trino won't ever actually address the limit
            self.f.sequence(0, self.f.cardinality(arg) - 1),
            sge.Lambda(
                this=self.cast(
                    sge.Struct(
                        expressions=[
                            sge.PropertyEQ(this=keep, expression=body),
                            sge.PropertyEQ(this=value, expression=param),
                        ]
                    ),
                    dt.Struct(
                        {
                            "keep": dt.boolean,
                            "value": op.arg.dtype.value_type,
                        }
                    ),
                ),
                expressions=[param, index],
            ),
        )

        # second, keep only the elements whose predicate returned true
        filtered = self.f.filter(
            # then, filter out elements that are null
            zipped,
            sge.Lambda(
                this=sge.Dot(this=placeholder, expression=keep),
                expressions=[placeholder],
            ),
        )

        # finally, extract the "value" field from the struct
        return self.f.transform(
            filtered,
            sge.Lambda(
                this=sge.Dot(this=placeholder, expression=value),
                expressions=[placeholder],
            ),
        )

    def visit_ArrayContains(self, op, *, arg, other):
        return self.if_(
            arg.is_(sg.not_(NULL)),
            self.f.coalesce(self.f.contains(arg, other), FALSE),
            NULL,
        )

    def visit_ArrayCollect(self, op, *, arg, where, order_by, include_null, distinct):
        if not include_null:
            cond = arg.is_(sg.not_(NULL, copy=False))
            where = cond if where is None else sge.And(this=cond, expression=where)
        if distinct:
            arg = sge.Distinct(expressions=[arg])
        return self.agg.array_agg(arg, where=where, order_by=order_by)

    def visit_JSONGetItem(self, op, *, arg, index):
        fmt = "%d" if op.index.dtype.is_integer() else '"%s"'
        return self.f.json_extract(arg, self.f.format(f"$[{fmt}]", index))

    def visit_UnwrapJSONString(self, op, *, arg):
        return self.f.json_value(
            self.f.json_format(arg), 'strict $?($.type() == "string")'
        )

    def visit_UnwrapJSONInt64(self, op, *, arg):
        value = self.f.json_value(
            self.f.json_format(arg), 'strict $?($.type() == "number")'
        )
        return self.cast(
            self.if_(self.f.regexp_like(value, r"^\d+$"), value, NULL), op.dtype
        )

    def visit_UnwrapJSONFloat64(self, op, *, arg):
        return self.cast(
            self.f.json_value(
                self.f.json_format(arg), 'strict $?($.type() == "number")'
            ),
            op.dtype,
        )

    def visit_UnwrapJSONBoolean(self, op, *, arg):
        return self.cast(
            self.f.json_value(
                self.f.json_format(arg), 'strict $?($.type() == "boolean")'
            ),
            op.dtype,
        )

    def visit_DayOfWeekIndex(self, op, *, arg):
        return self.cast(sge.paren(self.f.anon.dow(arg) + 6, copy=False) % 7, op.dtype)

    def visit_DayOfWeekName(self, op, *, arg):
        return self.f.date_format(arg, "%W")

    def visit_StrRight(self, op, *, arg, nchars):
        return self.f.substr(arg, -nchars)

    def visit_EndsWith(self, op, *, arg, end):
        return self.f.substr(arg, -self.f.length(end)).eq(end)

    def visit_Repeat(self, op, *, arg, times):
        return self.f.array_join(
            self.f.nullif(self.f.repeat(arg, times), self.f.array()), ""
        )

    def visit_DateTimestampTruncate(self, op, *, arg, unit):
        _truncate_precisions = {
            # ms unit is not yet officially documented but it works
            "ms": "millisecond",
            "s": "second",
            "m": "minute",
            "h": "hour",
            "D": "day",
            "W": "week",
            "M": "month",
            "Q": "quarter",
            "Y": "year",
        }

        if (precision := _truncate_precisions.get(unit.short)) is None:
            raise com.UnsupportedOperationError(
                f"Unsupported truncate unit {op.unit!r}"
            )
        return self.f.date_trunc(precision, arg)

    visit_DateTruncate = visit_TimestampTruncate = visit_DateTimestampTruncate

    def visit_DateFromYMD(self, op, *, year, month, day):
        return self.f.from_iso8601_date(
            self.f.format("%04d-%02d-%02d", year, month, day)
        )

    def visit_TimeFromHMS(self, op, *, hours, minutes, seconds):
        return self.cast(
            self.f.format("%02d:%02d:%02d", hours, minutes, seconds), dt.time
        )

    def visit_TimestampFromYMDHMS(
        self, op, *, year, month, day, hours, minutes, seconds
    ):
        return self.cast(
            self.f.from_iso8601_timestamp(
                self.f.format(
                    "%04d-%02d-%02dT%02d:%02d:%02d",
                    year,
                    month,
                    day,
                    hours,
                    minutes,
                    seconds,
                )
            ),
            dt.timestamp,
        )

    def visit_TimestampFromUNIX(self, op, *, arg, unit):
        short = unit.short
        if short == "ms":
            res = self.f.from_unixtime(self.f.floor(arg / 1_000))
        elif short == "s":
            res = self.f.from_unixtime(arg)
        elif short == "us":
            res = self.f.from_unixtime_nanos((arg - arg % 1_000_000) * 1_000)
        elif short == "ns":
            res = self.f.from_unixtime_nanos(arg - arg % 1_000_000_000)
        else:
            raise com.UnsupportedOperationError(f"{unit!r} unit is not supported")
        return self.cast(res, op.dtype)

    def visit_InSubquery(self, op, *, rel, needle):
        # cast the needle to the same type as the column being queried, since
        # trino is very strict about structs
        if op.needle.dtype.is_struct():
            needle = self.cast(
                sge.Struct.from_arg_list([needle]), op.rel.schema.as_struct()
            )

        return super().visit_InSubquery(op, rel=rel, needle=needle)

    def visit_StructColumn(self, op, *, names, values):
        return sge.TryCast(
            this=sge.Struct(expressions=values),
            to=self.type_mapper.from_ibis(op.dtype),
        )

    def visit_NonNullLiteral(self, op, *, value, dtype):
        if dtype.is_floating():
            if math.isfinite(value):
                return self.cast(value, dtype)
            return super().visit_NonNullLiteral(op, value=value, dtype=dtype)
        elif dtype.is_struct():
            return sge.TryCast(
                this=sge.Struct(
                    expressions=[
                        self.visit_Literal(
                            ops.Literal(v, field_dtype), value=v, dtype=field_dtype
                        )
                        for field_dtype, v in zip(dtype.types, value.values())
                    ]
                ),
                to=self.type_mapper.from_ibis(dtype),
            )
        elif dtype.is_timestamp():
            return self.cast(self.f.from_iso8601_timestamp(value.isoformat()), dtype)
        elif dtype.is_date():
            return self.f.from_iso8601_date(value.isoformat())
        elif dtype.is_time():
            return self.cast(value.isoformat(), dtype)
        elif dtype.is_interval():
            return self._make_interval(sge.convert(str(value)), dtype.unit)
        elif dtype.is_binary():
            return self.f.from_hex(value.hex())
        else:
            return None

    def visit_Log(self, op, *, arg, base):
        return self.f.log(base, arg)

    def visit_MapGet(self, op, *, arg, key, default):
        return self.f.coalesce(self.f.element_at(arg, key), default)

    def visit_MapContains(self, op, *, arg, key):
        return self.f.contains(self.f.map_keys(arg), key)

    def visit_ExtractFile(self, op, *, arg):
        return self.f.concat_ws(
            "?",
            self.f.nullif(self.f.url_extract_path(arg), ""),
            self.f.nullif(self.f.url_extract_query(arg), ""),
        )

    def visit_ExtractQuery(self, op, *, arg, key):
        if key is None:
            return self.f.url_extract_query(arg)
        return self.f.url_extract_parameter(arg, key)

    def visit_Cot(self, op, *, arg):
        return 1.0 / self.f.tan(arg)

    def visit_StringAscii(self, op, *, arg):
        return self.f.codepoint(
            sge.Cast(
                this=self.f.substr(arg, 1, 2),
                to=sge.DataType(
                    this=sge.DataType.Type.VARCHAR,
                    expressions=[sge.DataTypeParam(this=sge.convert(1))],
                ),
            )
        )

    def visit_ArrayStringJoin(self, op, *, sep, arg):
        return self.f.array_join(self.f.nullif(arg, self.f.array()), sep)

    def visit_First(self, op, *, arg, where, order_by, include_null):
        if not include_null:
            cond = arg.is_(sg.not_(NULL, copy=False))
            where = cond if where is None else sge.And(this=cond, expression=where)
        return self.f.element_at(
            self.agg.array_agg(arg, where=where, order_by=order_by), 1
        )

    def visit_Last(self, op, *, arg, where, order_by, include_null):
        if not include_null:
            cond = arg.is_(sg.not_(NULL, copy=False))
            where = cond if where is None else sge.And(this=cond, expression=where)
        return self.f.element_at(
            self.agg.array_agg(arg, where=where, order_by=order_by), -1
        )

    def visit_GroupConcat(self, op, *, arg, sep, where, order_by):
        cond = arg.is_(sg.not_(NULL, copy=False))
        where = cond if where is None else sge.And(this=cond, expression=where)
        array = self.agg.array_agg(
            self.cast(arg, dt.string), where=where, order_by=order_by
        )
        return self.f.array_join(self.f.nullif(array, self.f.array()), sep)

    def visit_ArrayZip(self, op, *, arg):
        max_zip_arguments = 5
        chunks = (
            (len(chunk), self.f.zip(*chunk) if len(chunk) > 1 else chunk[0])
            for chunk in toolz.partition_all(max_zip_arguments, arg)
        )

        def combine_zipped(left, right):
            left_n, left_chunk = left
            x, y = map(sg.to_identifier, "xy")

            lhs = list(map(x.__getitem__, range(left_n))) if left_n > 1 else [x]

            right_n, right_chunk = right
            rhs = list(map(y.__getitem__, range(right_n))) if right_n > 1 else [y]

            zipped_chunk = self.f.zip_with(
                left_chunk,
                right_chunk,
                sge.Lambda(this=self.f.row(*lhs, *rhs), expressions=[x, y]),
            )
            return left_n + right_n, zipped_chunk

        all_n, chunk = reduce(combine_zipped, chunks)
        assert all_n == len(op.dtype.value_type)
        return chunk

    def visit_ExtractMicrosecond(self, op, *, arg):
        # trino only seems to store milliseconds, but the result of formatting
        # always pads the right with 000
        return self.cast(self.f.date_format(arg, "%f"), dt.int32)

    def visit_TemporalDelta(self, op, *, part, left, right):
        # trino truncates _after_ the delta, whereas many other backends
        # truncate each operand
        return sge.DateDiff(
            this=self.f.date_trunc(part, left),
            expression=self.f.date_trunc(part, right),
            unit=part,
        )

    visit_TimeDelta = visit_DateDelta = visit_TimestampDelta = visit_TemporalDelta

    def _make_interval(self, arg, unit):
        short = unit.short
        if short in ("Q", "W"):
            raise com.UnsupportedOperationError(f"Interval unit {unit!r} not supported")

        if isinstance(arg, sge.Literal):
            # force strings in interval literals because trino requires it
            arg.args["is_string"] = True
            return super()._make_interval(arg, unit)

        elif short in ("Y", "M"):
            return arg * super()._make_interval(sge.convert("1"), unit)
        elif short in ("D", "h", "m", "s", "ms", "us"):
            return self.f.parse_duration(
                self.f.concat(self.cast(arg, dt.string), short.lower())
            )
        else:
            raise com.UnsupportedOperationError(
                f"Interval unit {unit.name!r} not supported"
            )

    def visit_Range(self, op, *, start, stop, step):
        def zero_value(dtype):
            if dtype.is_interval():
                # the unit doesn't matter here, because e.g. 0d = 0s
                return self.f.parse_duration("0s")
            return 0

        def interval_sign(v):
            zero = self.f.parse_duration("0s")
            return sge.Case(
                ifs=[
                    self.if_(v.eq(zero), 0),
                    self.if_(v < zero, -1),
                    self.if_(v > zero, 1),
                ]
            )

        def _sign(value, dtype):
            if dtype.is_interval():
                return interval_sign(value)
            return self.f.sign(value)

        step_dtype = op.step.dtype
        zero = zero_value(step_dtype)
        return self.if_(
            sg.and_(
                self.f.nullif(step, zero).is_(sg.not_(NULL)),
                _sign(step, step_dtype).eq(_sign(stop - start, step_dtype)),
            ),
            self.f.array_remove(self.f.sequence(start, stop, step), stop),
            self.f.array(),
        )

    visit_IntegerRange = visit_TimestampRange = visit_Range

    def visit_ArrayIndex(self, op, *, arg, index):
        return self.f.element_at(arg, index + 1)

    def visit_Cast(self, op, *, arg, to):
        from_ = op.arg.dtype
        if from_.is_numeric() and to.is_timestamp():
            tz = to.timezone or "UTC"
            if from_.is_integer():
                return self.f.from_unixtime(arg, tz)
            else:
                return self.f.from_unixtime_nanos(
                    self.cast(arg, dt.Decimal(38, 9)) * 1_000_000_000
                )
        return super().visit_Cast(op, arg=arg, to=to)

    def visit_CountDistinctStar(self, op, *, arg, where):
        make_col = partial(sg.column, table=arg.alias_or_name, quoted=self.quoted)
        row = self.f.row(*map(make_col, op.arg.schema.names))
        return self.agg.count(sge.Distinct(expressions=[row]), where=where)

    def visit_ArrayConcat(self, op, *, arg):
        return self.f.concat(*arg)

    def visit_StringContains(self, op, *, haystack, needle):
        return self.f.strpos(haystack, needle) > 0

    def visit_RegexExtract(self, op, *, arg, pattern, index):
        # sqlglot doesn't support the third `group` argument for trino so work
        # around that limitation using an anonymous function
        return self.f.anon.regexp_extract(arg, pattern, index)

    def visit_ToJSONMap(self, op, *, arg):
        return self.cast(
            self.f.json_parse(
                self.f.json_query(
                    self.f.json_format(arg), 'strict $?($.type() == "object")'
                )
            ),
            dt.Map(dt.string, dt.json),
        )

    def visit_ToJSONArray(self, op, *, arg):
        return self.cast(
            self.f.json_parse(
                self.f.json_query(
                    self.f.json_format(arg), 'strict $?($.type() == "array")'
                )
            ),
            dt.Array(dt.json),
        )

    def visit_TableUnnest(
        self,
        op,
        *,
        parent,
        column,
        column_name: str,
        offset: str | None,
        keep_empty: bool,
    ):
        quoted = self.quoted

        column_alias = sg.to_identifier(gen_name("table_unnest_column"), quoted=quoted)

        parent_schema = op.parent.schema
        overlaps_with_parent = column_name in parent_schema
        computed_column = column_alias.as_(column_name, quoted=quoted)

        parent_alias_or_name = parent.alias_or_name

        selcols = []

        if overlaps_with_parent:
            column_alias_or_name = column.alias_or_name
            selcols.extend(
                sg.column(col, table=parent_alias_or_name, quoted=quoted)
                if col != column_alias_or_name
                else computed_column
                for col in parent_schema.names
            )
        else:
            selcols.append(
                sge.Column(
                    this=STAR,
                    table=sg.to_identifier(parent_alias_or_name, quoted=quoted),
                )
            )
            selcols.append(computed_column)

        if offset is not None:
            offset_name = offset
            offset = sg.to_identifier(offset_name, quoted=quoted)
            selcols.append((offset - 1).as_(offset_name, quoted=quoted))

        unnest = sge.Unnest(
            expressions=[column],
            alias=sge.TableAlias(
                this=sg.to_identifier(gen_name("table_unnest"), quoted=quoted),
                columns=[column_alias],
            ),
            offset=offset,
        )
        return (
            sg.select(*selcols)
            .from_(parent)
            .join(
                unnest,
                on=None if not keep_empty else TRUE,
                join_type="CROSS" if not keep_empty else "LEFT",
            )
        )

    def visit_ArrayAny(self, op, *, arg):
        x = sg.to_identifier("x", quoted=self.quoted)
        identity = sge.Lambda(this=x, expressions=[x])
        is_not_null = sge.Lambda(this=x.is_(sg.not_(NULL)), expressions=[x])
        return self.f.any_match(
            self.f.nullif(self.f.filter(arg, is_not_null), self.f.array()), identity
        )

    def visit_ArrayAll(self, op, *, arg):
        x = sg.to_identifier("x", quoted=self.quoted)
        identity = sge.Lambda(this=x, expressions=[x])
        is_not_null = sge.Lambda(this=x.is_(sg.not_(NULL)), expressions=[x])
        return self.f.all_match(
            self.f.nullif(self.f.filter(arg, is_not_null), self.f.array()), identity
        )

    def visit_ArrayMin(self, op, *, arg):
        x = sg.to_identifier("x", quoted=self.quoted)
        func = sge.Lambda(this=x.is_(sg.not_(NULL)), expressions=[x])
        return self.f.array_min(self.f.filter(arg, func))

    def visit_ArrayMax(self, op, *, arg):
        x = sg.to_identifier("x", quoted=self.quoted)
        func = sge.Lambda(this=x.is_(sg.not_(NULL)), expressions=[x])
        return self.f.array_max(self.f.filter(arg, func))

    def visit_ArraySumAgg(self, op, *, arg, output):
        quoted = self.quoted
        dot = lambda a, f: sge.Dot.build((a, sge.to_identifier(f, quoted=quoted)))
        state_dtype = dt.Struct({"sum": op.dtype, "count": dt.int64})
        initial_state = self.cast(
            sge.Struct.from_arg_list([sge.convert(0), sge.convert(0)]), state_dtype
        )

        s = sg.to_identifier("s", quoted=quoted)
        x = sg.to_identifier("x", quoted=quoted)

        s_sum = dot(s, "sum")
        s_count = dot(s, "count")

        input_fn_body = self.cast(
            sge.Struct.from_arg_list(
                [
                    x + self.f.coalesce(s_sum, 0),
                    s_count + self.if_(x.is_(sg.not_(NULL)), 1, 0),
                ]
            ),
            state_dtype,
        )
        input_fn = sge.Lambda(this=input_fn_body, expressions=[s, x])

        output_fn_body = self.if_(s_count > 0, output(s_sum, s_count), NULL)
        return self.f.reduce(
            arg,
            initial_state,
            input_fn,
            sge.Lambda(this=output_fn_body, expressions=[s]),
        )

    def visit_ArraySum(self, op, *, arg):
        return self.visit_ArraySumAgg(op, arg=arg, output=lambda sum, _: sum)

    def visit_ArrayMean(self, op, *, arg):
        return self.visit_ArraySumAgg(op, arg=arg, output=operator.truediv)


compiler = TrinoCompiler()
