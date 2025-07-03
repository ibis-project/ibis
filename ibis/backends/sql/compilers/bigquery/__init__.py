"""Module to convert from Ibis expression to SQL string."""

from __future__ import annotations

import decimal
import math
import re
from typing import TYPE_CHECKING, Any

import sqlglot as sg
import sqlglot.expressions as sge
from sqlglot.dialects import BigQuery

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis import util
from ibis.backends.sql.compilers.base import NULL, STAR, AggGen, SQLGlotCompiler
from ibis.backends.sql.compilers.bigquery.udf.core import PythonToJavaScriptTranslator
from ibis.backends.sql.datatypes import BigQueryType, BigQueryUDFType
from ibis.backends.sql.rewrites import (
    FirstValue,
    LastValue,
    exclude_unsupported_window_frame_from_ops,
    exclude_unsupported_window_frame_from_rank,
    exclude_unsupported_window_frame_from_row_number,
    lower_sample,
    split_select_distinct_with_order_by,
)
from ibis.common.temporal import DateUnit, IntervalUnit, TimestampUnit, TimeUnit

if TYPE_CHECKING:
    from collections.abc import Mapping

    import ibis.expr.types as ir

_NAME_REGEX = re.compile(r'[^!"$()*,./;?@[\\\]^`{}~\n]+')


def _qualify_memtable(
    node: sge.Expression,
    *,
    dataset: str | None,
    project: str | None,
    memtable_names: frozenset[str],
) -> sge.Expression:
    """Add a BigQuery dataset and project to memtable references."""
    if isinstance(node, sge.Table) and node.name in memtable_names:
        node.args["db"] = dataset
        node.args["catalog"] = project
        # make sure to quote table location
        node = _force_quote_table(node)
    return node


def _remove_null_ordering_from_unsupported_window(
    node: sge.Expression,
) -> sge.Expression:
    """Remove null ordering in window frame clauses not supported by BigQuery.

    BigQuery has only partial support for NULL FIRST/LAST in RANGE windows so
    we remove it from any window frame clause that doesn't support it.

    Here's the support matrix:

    ✅ sum(x) over (order by y desc nulls last)
    🚫 sum(x) over (order by y asc nulls last)
    ✅ sum(x) over (order by y asc nulls first)
    🚫 sum(x) over (order by y desc nulls first)
    """
    if isinstance(node, sge.Window):
        order = node.args.get("order")
        if order is not None:
            for key in order.args["expressions"]:
                kargs = key.args
                if kargs.get("desc") is True and kargs.get("nulls_first", False):
                    kargs["nulls_first"] = False
                elif kargs.get("desc") is False and not kargs.setdefault(
                    "nulls_first", True
                ):
                    kargs["nulls_first"] = True
    return node


def _force_quote_table(table: sge.Table) -> sge.Table:
    """Force quote all the parts of a bigquery path.

    The BigQuery identifier quoting semantics are bonkers
    https://cloud.google.com/bigquery/docs/reference/standard-sql/lexical#identifiers

    my-table is OK, but not mydataset.my-table

    mytable-287 is OK, but not mytable-287a

    Just quote everything.
    """
    for key in ("this", "db", "catalog"):
        if (val := table.args[key]) is not None:
            if isinstance(val, sg.exp.Identifier) and not val.quoted:
                val.args["quoted"] = True
            else:
                table.args[key] = sg.to_identifier(val, quoted=True)
    return table


class BigQueryCompiler(SQLGlotCompiler):
    dialect = BigQuery
    type_mapper = BigQueryType
    udf_type_mapper = BigQueryUDFType

    agg = AggGen(supports_order_by=True)

    rewrites = (
        exclude_unsupported_window_frame_from_ops,
        exclude_unsupported_window_frame_from_row_number,
        exclude_unsupported_window_frame_from_rank,
        *SQLGlotCompiler.rewrites,
    )
    post_rewrites = (split_select_distinct_with_order_by,)

    supports_qualify = True

    LOWERED_OPS = {
        ops.Sample: lower_sample(
            supported_methods=("block",),
            supports_seed=False,
            physical_tables_only=True,
        ),
    }

    UNSUPPORTED_OPS = (
        ops.DateDiff,
        ops.ExtractAuthority,
        ops.ExtractUserInfo,
        ops.FindInSet,
        ops.Median,
        ops.RegexSplit,
        ops.RowID,
        ops.TimestampDiff,
        ops.Kurtosis,
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
        ops.Arbitrary: "any_value",
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
        ops.Levenshtein: "edit_distance",
        ops.Modulus: "mod",
        ops.RegexReplace: "regexp_replace",
        ops.RegexSearch: "regexp_contains",
        ops.Time: "time",
        ops.TimeFromHMS: "time_from_parts",
        ops.TimestampNow: "current_timestamp",
        ops.ExtractHost: "net.host",
        ops.RandomUUID: "generate_uuid",
    }

    def to_sqlglot(
        self,
        expr: ir.Expr,
        *,
        limit: str | None = None,
        params: Mapping[ir.Expr, Any] | None = None,
        session_dataset_id: str | None = None,
        session_project: str | None = None,
    ) -> Any:
        """Compile an Ibis expression.

        Parameters
        ----------
        expr
            Ibis expression
        limit
            For expressions yielding result sets; retrieve at most this number
            of values/rows. Overrides any limit already set on the expression.
        params
            Named unbound parameters
        session_dataset_id
            Optional dataset ID to qualify memtable references.
        session_project
            Optional project ID to qualify memtable references.

        Returns
        -------
        Any
            The output of compilation. The type of this value depends on the
            backend.

        """
        sql = super().to_sqlglot(expr, limit=limit, params=params)

        table_expr = expr.as_table()

        memtable_names = frozenset(
            op.name for op in table_expr.op().find(ops.InMemoryTable)
        )

        result = sql.transform(
            _qualify_memtable,
            dataset=session_dataset_id,
            project=session_project,
            memtable_names=memtable_names,
        ).transform(_remove_null_ordering_from_unsupported_window)

        sources = []

        for udf_node in table_expr.op().find(ops.ScalarUDF):
            compile_func = getattr(
                self, f"_compile_{udf_node.__input_type__.name.lower()}_udf"
            )
            if sql := compile_func(udf_node):
                sources.append(sql)

        if not sources:
            return result

        sources.append(result)
        return sources

    def _compile_python_udf(self, udf_node: ops.ScalarUDF) -> sge.Create:
        name = type(udf_node).__name__
        type_mapper = self.udf_type_mapper

        body = PythonToJavaScriptTranslator(udf_node.__func__).compile()
        config = udf_node.__config__
        libraries = config.get("libraries", [])

        signature = [
            sge.ColumnDef(
                this=sg.to_identifier(name, quoted=self.quoted),
                kind=type_mapper.from_ibis(param.annotation.pattern.dtype),
            )
            for name, param in udf_node.__signature__.parameters.items()
        ]

        lines = ['"""']

        if config.get("strict", True):
            lines.append('"use strict";')

        lines += [
            body,
            "",
            f"return {udf_node.__func_name__}({', '.join(udf_node.argnames)});",
            '"""',
        ]

        func = sge.Create(
            kind="FUNCTION",
            this=sge.UserDefinedFunction(
                this=sg.to_identifier(name), expressions=signature, wrapped=True
            ),
            # not exactly what I had in mind, but it works
            #
            # quoting is too simplistic to handle multiline strings
            expression=sge.Var(this="\n".join(lines)),
            exists=False,
            properties=sge.Properties(
                expressions=[
                    sge.TemporaryProperty(),
                    sge.ReturnsProperty(this=type_mapper.from_ibis(udf_node.dtype)),
                    sge.StabilityProperty(
                        this="IMMUTABLE" if config.get("determinism") else "VOLATILE"
                    ),
                    sge.LanguageProperty(this=sg.to_identifier("js")),
                ]
                + [
                    sge.Property(
                        this=sg.to_identifier("library"), value=self.f.array(*libraries)
                    )
                ]
                * bool(libraries)
            ),
        )

        return func

    @staticmethod
    def _minimize_spec(op, spec):
        # bigquery doesn't allow certain window functions to specify a window frame
        if (
            isinstance(func := op.func, ops.CountDistinct)
            and (spec.args["start"], spec.args["end"]) == ("UNBOUNDED", "UNBOUNDED")
        ) or (
            isinstance(func, ops.Analytic)
            and not isinstance(
                func, (ops.First, ops.Last, FirstValue, LastValue, ops.NthValue)
            )
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
        return sge.TimeDiff(this=left, expression=right, unit=self.v[part])

    def visit_DateDelta(self, op, *, left, right, part):
        return sge.DateDiff(this=left, expression=right, unit=self.v[part])

    def visit_TimestampDelta(self, op, *, left, right, part):
        left_tz = op.left.dtype.timezone
        right_tz = op.right.dtype.timezone

        if left_tz is None and right_tz is None:
            return sge.DatetimeDiff(this=left, expression=right, unit=self.v[part])
        elif left_tz is not None and right_tz is not None:
            return sge.TimestampDiff(this=left, expression=right, unit=self.v[part])

        raise com.UnsupportedOperationError(
            "timestamp difference with mixed timezone/timezoneless values is not implemented"
        )

    def visit_GroupConcat(self, op, *, arg, sep, where, order_by):
        if where is not None:
            arg = self.if_(where, arg, NULL)

        if order_by:
            arg = sge.Order(this=arg, expressions=order_by)

        return sge.GroupConcat(this=arg, separator=sep)

    def visit_ApproxQuantile(self, op, *, arg, quantile, where):
        if not isinstance(op.quantile, ops.Literal):
            raise com.UnsupportedOperationError(
                "quantile must be a literal in BigQuery"
            )

        # BigQuery syntax is `APPROX_QUANTILES(col, resolution)` to return
        # `resolution + 1` quantiles array. To handle this, we compute the
        # resolution ourselves then restructure the output array as needed.
        # To avoid excessive resolution we arbitrarily cap it at 100,000 -
        # since these are approximate quantiles anyway this seems fine.
        quantiles = util.promote_list(op.quantile.value)
        fracs = [decimal.Decimal(str(q)).as_integer_ratio() for q in quantiles]
        resolution = min(math.lcm(*(den for _, den in fracs)), 100_000)
        indices = [(num * resolution) // den for num, den in fracs]

        if where is not None:
            arg = self.if_(where, arg, NULL)

        if not op.arg.dtype.is_floating():
            arg = self.cast(arg, dt.float64)

        array = self.f.approx_quantiles(
            arg, sge.IgnoreNulls(this=sge.convert(resolution))
        )
        if isinstance(op, ops.ApproxQuantile):
            return array[indices[0]]

        if indices == list(range(resolution + 1)):
            return array
        else:
            return sge.Array(expressions=[array[i] for i in indices])

    visit_ApproxMultiQuantile = visit_ApproxQuantile

    def visit_FloorDivide(self, op, *, left, right):
        return self.cast(self.f.floor(self.f.ieee_divide(left, right)), op.dtype)

    def visit_Log2(self, op, *, arg):
        return self.f.log(arg, 2)

    def visit_Log(self, op, *, arg, base):
        if base is None:
            return self.f.ln(arg)
        return self.f.log(arg, base)

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

    def visit_ArrayCollect(self, op, *, arg, where, order_by, include_null, distinct):
        if where is not None:
            if include_null:
                raise com.UnsupportedOperationError(
                    "Combining `include_null=True` and `where` is not supported by bigquery"
                )
            if distinct:
                raise com.UnsupportedOperationError(
                    "Combining `distinct=True` and `where` is not supported by bigquery"
                )
            arg = compiler.if_(where, arg, NULL)
        if distinct:
            arg = sge.Distinct(expressions=[arg])
        if order_by:
            arg = sge.Order(this=arg, expressions=order_by)
        if not include_null:
            arg = sge.IgnoreNulls(this=arg)
        return self.f.array_agg(arg)

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

    def visit_TimestampFromYMDHMS(
        self, op, *, year, month, day, hours, minutes, seconds
    ):
        return self.f.anon.DATETIME(year, month, day, hours, minutes, seconds)

    def visit_NonNullLiteral(self, op, *, value, dtype):
        if dtype.is_inet() or dtype.is_macaddr():
            return sge.convert(str(value))
        elif dtype.is_timestamp():
            funcname = "DATETIME" if dtype.timezone is None else "TIMESTAMP"
            return self.f.anon[funcname](value.isoformat())
        elif dtype.is_date():
            return self.f.date_from_parts(value.year, value.month, value.day)
        elif dtype.is_time():
            time = self.f.time_from_parts(value.hour, value.minute, value.second)
            if micros := value.microsecond:
                # bigquery doesn't support `time(12, 34, 56.789101)`, AKA a
                # float seconds specifier, so add any non-zero micros to the
                # time value
                return sge.TimeAdd(
                    this=time, expression=sge.convert(micros), unit=self.v.MICROSECOND
                )
            return time
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
        elif from_.is_numeric() and to.is_timestamp():
            if from_.is_integer():
                return self.f.timestamp_seconds(arg)
            return self.f.timestamp_micros(self.cast(arg * 1_000_000, dt.int64))
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
        elif from_.is_floating() and to.is_integer():
            return self.cast(self.f.trunc(arg), dt.int64)
        return super().visit_Cast(op, arg=arg, to=to)

    def visit_JSONGetItem(self, op, *, arg, index):
        return arg[index]

    def visit_UnwrapJSONString(self, op, *, arg):
        return self.f.anon["safe.string"](arg)

    def visit_UnwrapJSONInt64(self, op, *, arg):
        return self.f.anon["safe.int64"](arg)

    def visit_UnwrapJSONFloat64(self, op, *, arg):
        return self.f.anon["safe.float64"](arg)

    def visit_UnwrapJSONBoolean(self, op, *, arg):
        return self.f.anon["safe.bool"](arg)

    def visit_ExtractEpochSeconds(self, op, *, arg):
        return self.f.unix_seconds(arg)

    def visit_ExtractWeekOfYear(self, op, *, arg):
        return self.f.extract(self.v.isoweek, arg)

    def visit_ExtractIsoYear(self, op, *, arg):
        return self.f.extract(self.v.isoyear, arg)

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
        return self.f.timestamp_trunc(arg, self.v[unit])

    def visit_DateTruncate(self, op, *, arg, unit):
        if unit == DateUnit.WEEK:
            unit = "WEEK(MONDAY)"
        else:
            unit = unit.name
        return self.f.date_trunc(arg, self.v[unit])

    def visit_TimeTruncate(self, op, *, arg, unit):
        if unit == TimeUnit.NANOSECOND:
            raise com.UnsupportedOperationError(
                f"BigQuery does not support truncating {op.arg.dtype} values to unit {unit!r}"
            )
        else:
            unit = unit.name
        return self.f.time_trunc(arg, self.v[unit])

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

    def visit_First(self, op, *, arg, where, order_by, include_null):
        if where is not None:
            arg = self.if_(where, arg, NULL)
            if include_null:
                raise com.UnsupportedOperationError(
                    "Combining `include_null=True` and `where` is not supported "
                    "by bigquery"
                )

        if order_by:
            arg = sge.Order(this=arg, expressions=order_by)

        if not include_null:
            arg = sge.IgnoreNulls(this=arg)

        array = self.f.array_agg(sge.Limit(this=arg, expression=sge.convert(1)))
        return array[self.f.safe_offset(0)]

    def visit_Last(self, op, *, arg, where, order_by, include_null):
        if where is not None:
            arg = self.if_(where, arg, NULL)
            if include_null:
                raise com.UnsupportedOperationError(
                    "Combining `include_null=True` and `where` is not supported "
                    "by bigquery"
                )

        if order_by:
            arg = sge.Order(this=arg, expressions=order_by)

        if not include_null:
            arg = sge.IgnoreNulls(this=arg)

        array = self.f.array_reverse(self.f.array_agg(arg))
        return array[self.f.safe_offset(0)]

    def visit_ArrayStringJoin(self, op, *, arg, sep):
        return self.if_(
            self.f.array_length(arg) > 0, self.f.array_to_string(arg, sep), NULL
        )

    def visit_ArrayFilter(self, op, *, arg, body, param, index):
        return self.f.array(
            sg.select(param)
            .from_(self._unnest(arg, as_=param, offset=index))
            .where(body)
        )

    def visit_ArrayMap(self, op, *, arg, body, param, index):
        return self.f.array(
            sg.select(body).from_(self._unnest(arg, as_=param, offset=index))
        )

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
        both_null = sg.and_(name.is_(NULL), other.is_(NULL))
        cond = sg.or_(name.neq(other), both_null)
        return self.f.array(sg.select(name).from_(unnest).where(cond))

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

    def visit_RegexExtract(self, op, *, arg, pattern, index):
        matches = self.f.regexp_contains(arg, pattern)
        nonzero_index_replace = self.f.regexp_replace(
            arg,
            self.f.concat(".*?", pattern, ".*"),
            self.f.concat("\\", self.cast(index, dt.string)),
        )
        zero_index_replace = self.f.regexp_replace(
            arg, self.f.concat(".*?", self.f.concat("(", pattern, ")"), ".*"), "\\1"
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
        return self._pudf("typeof", arg)

    def visit_Xor(self, op, *, left, right):
        return sg.or_(sg.and_(left, sg.not_(right)), sg.and_(sg.not_(left), right))

    def visit_HashBytes(self, op, *, arg, how):
        if how not in ("md5", "sha1", "sha256", "sha512"):
            raise NotImplementedError(how)
        return self.f[how](arg)

    @staticmethod
    def _gen_valid_name(name: str) -> str:
        candidate = "_".join(map(str.strip, _NAME_REGEX.findall(name))) or "tmp"
        # column names cannot be longer than 300 characters
        #
        # https://cloud.google.com/bigquery/docs/schemas#column_names
        #
        # it's easy to rename columns, so raise an exception telling the user
        # to do so
        #
        # we could potentially relax this and support arbitrary-length columns
        # by compressing the information using hashing, but there's no reason
        # to solve that problem until someone encounters this error and cannot
        # rename their columns
        limit = 300
        if len(candidate) > limit:
            raise com.IbisError(
                f"BigQuery does not allow column names longer than {limit:d} characters. "
                "Please rename your columns to have fewer characters."
            )
        return candidate

    def visit_CountStar(self, op, *, arg, where):
        if where is not None:
            return self.f.countif(where)
        return self.f.count(STAR)

    def visit_CountDistinctStar(self, op, *, where, arg):
        # Bigquery does not support count(distinct a,b,c) or count(distinct (a, b, c))
        # as expressions must be "groupable":
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax#group_by_grouping_item
        #
        # Instead, convert the entire expression to a string
        # SELECT COUNT(DISTINCT concat(to_json_string(a), to_json_string(b)))
        # This works with an array of datatypes which generates a unique string
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/json_functions#json_encodings
        row = sge.Concat(
            expressions=[
                self.f.to_json_string(sg.column(x, quoted=self.quoted))
                for x in op.arg.schema.keys()
            ]
        )
        if where is not None:
            row = self.if_(where, row, NULL)
        return self.f.count(sge.Distinct(expressions=[row]))

    def visit_Degrees(self, op, *, arg):
        return self._pudf("degrees", arg)

    def visit_Radians(self, op, *, arg):
        return self._pudf("radians", arg)

    def visit_CountDistinct(self, op, *, arg, where):
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return self.f.count(sge.Distinct(expressions=[arg]))

    def visit_ExtractFile(self, op, *, arg):
        return self._pudf("cw_url_extract_file", arg)

    def visit_ExtractFragment(self, op, *, arg):
        return self._pudf("cw_url_extract_fragment", arg)

    def visit_ExtractPath(self, op, *, arg):
        return self._pudf("cw_url_extract_path", arg)

    def visit_ExtractProtocol(self, op, *, arg):
        return self._pudf("cw_url_extract_protocol", arg)

    def visit_ExtractQuery(self, op, *, arg, key):
        if key is not None:
            return self._pudf("cw_url_extract_parameter", arg, key)
        else:
            return self._pudf("cw_url_extract_query", arg)

    def _pudf(self, name, *args):
        name = sg.table(name, db="persistent_udfs", catalog="bigquery-public-data").sql(
            self.dialect
        )
        return self.f[name](*args)

    def visit_DropColumns(self, op, *, parent, columns_to_drop):
        quoted = self.quoted
        excludes = [sg.column(column, quoted=quoted) for column in columns_to_drop]
        star = sge.Star(**{"except": excludes})
        table = sg.to_identifier(parent.alias_or_name, quoted=quoted)
        column = sge.Column(this=star, table=table)
        return sg.select(column).from_(parent)

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

        column_alias = sg.to_identifier(
            util.gen_name("table_unnest_column"), quoted=quoted
        )

        selcols = []

        table = sg.to_identifier(parent.alias_or_name, quoted=quoted)

        overlaps_with_parent = column_name in op.parent.schema
        computed_column = column_alias.as_(column_name, quoted=quoted)

        # replace the existing column if the unnested column hasn't been
        # renamed
        #
        # e.g., table.unnest("x")
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

        unnest = sge.Unnest(
            expressions=[column],
            alias=sge.TableAlias(columns=[column_alias]),
            offset=offset,
        )
        return (
            sg.select(*selcols)
            .from_(parent)
            .join(unnest, join_type="CROSS" if not keep_empty else "LEFT")
        )

    def visit_TimestampBucket(self, op, *, arg, interval, offset):
        arg_dtype = op.arg.dtype
        if arg_dtype.timezone is not None:
            funcname = "timestamp"
        else:
            funcname = "datetime"

        func = self.f[f"{funcname}_bucket"]

        origin = sge.convert("1970-01-01")
        if offset is not None:
            origin = self.f.anon[f"{funcname}_add"](origin, offset)

        return func(arg, interval, origin)

    def _array_reduction(self, *, arg, reduction):
        name = sg.to_identifier(util.gen_name(f"bq_arr_{reduction}"))
        return (
            sg.select(self.f[reduction](name))
            .from_(self._unnest(arg, as_=name))
            .subquery()
        )

    def visit_ArrayMin(self, op, *, arg):
        return self._array_reduction(arg=arg, reduction="min")

    def visit_ArrayMax(self, op, *, arg):
        return self._array_reduction(arg=arg, reduction="max")

    def visit_ArraySum(self, op, *, arg):
        return self._array_reduction(arg=arg, reduction="sum")

    def visit_ArrayMean(self, op, *, arg):
        return self._array_reduction(arg=arg, reduction="avg")

    def visit_ArrayAny(self, op, *, arg):
        return self._array_reduction(arg=arg, reduction="logical_or")

    def visit_ArrayAll(self, op, *, arg):
        return self._array_reduction(arg=arg, reduction="logical_and")


compiler = BigQueryCompiler()
