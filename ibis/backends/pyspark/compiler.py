from __future__ import annotations

import calendar
import itertools
import re

import sqlglot as sg
import sqlglot.expressions as sge
from public import public

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.sql.compiler import FALSE, NULL, STAR, TRUE, SQLGlotCompiler
from ibis.backends.sql.datatypes import PySparkType
from ibis.backends.sql.dialects import PySpark
from ibis.backends.sql.rewrites import p
from ibis.common.patterns import replace
from ibis.config import options
from ibis.util import gen_name


@replace(p.Limit)
def offset_to_filter(_):
    # spark doesn't support dynamic limit, so raise an error if either limit or
    # offset is not a literal expression
    if isinstance(_.n, ops.Value) and _.n.find(ops.Relation):
        raise com.UnsupportedOperationError(
            "PySpark backend does not support dynamic limit."
        )
    if isinstance(_.offset, ops.Value) and _.offset.find(ops.Relation):
        raise com.UnsupportedOperationError(
            "PySpark backend does not support dynamic offset."
        )
    if _.offset == 0:
        return _
    # spark doesn't support offset by default, so we need to emulate it by first
    # generating row numbers and then filtering out the first N rows
    field_name = gen_name("ibis_row_number")
    rel = _.parent.to_expr()
    rel = rel.mutate(ibis.row_number().name(field_name))
    rel = rel.filter(rel[field_name] > _.offset)
    return _.copy(parent=rel, offset=0)


@public
class PySparkCompiler(SQLGlotCompiler):
    __slots__ = ()

    dialect = PySpark
    type_mapper = PySparkType
    rewrites = (offset_to_filter, *SQLGlotCompiler.rewrites)

    UNSUPPORTED_OPERATIONS = frozenset(
        (
            ops.RowID,
            ops.TimestampBucket,
        )
    )

    SIMPLE_OPS = {
        ops.ArrayDistinct: "array_distinct",
        ops.ArrayFlatten: "flatten",
        ops.ArrayIntersect: "array_intersect",
        ops.ArrayRemove: "array_remove",
        ops.ArraySort: "array_sort",
        ops.ArrayUnion: "array_union",
        ops.EndsWith: "endswith",
        ops.Hash: "hash",
        ops.Log10: "log10",
        ops.LStrip: "ltrim",
        ops.RStrip: "rtrim",
        ops.MapLength: "size",
        ops.MapContains: "map_contains_key",
        ops.MapMerge: "map_concat",
        ops.MapKeys: "map_keys",
        ops.MapValues: "map_values",
    }

    def _aggregate(self, funcname: str, *args, where):
        func = self.f[funcname]
        if where is not None:
            args = tuple(self.if_(where, arg, NULL) for arg in args)
        return func(*args)

    def visit_NonNullLiteral(self, op, *, value, dtype):
        if dtype.is_floating():
            result = super().visit_NonNullLiteral(op, value=value, dtype=dtype)
            if options.pyspark.treat_nan_as_null:
                return self.f.nanvl(result, NULL)
            else:
                return result
        elif dtype.is_string():
            value = value.replace("\\", "\\\\")
            return sge.convert(value)
        elif dtype.is_binary():
            return self.f.unhex(value.hex())
        elif dtype.is_decimal():
            if value.is_finite():
                return self.cast(str(value), dtype)
            else:
                return self.cast(str(value), dt.float64)
        elif dtype.is_uuid():
            return sge.convert(str(value))
        else:
            return None

    def visit_Field(self, op, *, rel, name):
        result = super().visit_Field(op, rel=rel, name=name)
        if op.dtype.is_floating() and options.pyspark.treat_nan_as_null:
            return self.f.nanvl(result, NULL)
        else:
            return result

    def visit_Cast(self, op, *, arg, to):
        if to.is_json():
            if op.arg.dtype.is_string():
                return arg
            else:
                return self.f.to_json(arg)
        else:
            return self.cast(arg, to)

    def visit_IsNull(self, op, *, arg):
        is_null = arg.is_(NULL)
        is_nan = self.f.isnan(arg)
        if op.arg.dtype.is_floating():
            return sg.or_(is_null, is_nan)
        else:
            return is_null

    def visit_NotNull(self, op, *, arg):
        is_not_null = arg.is_(sg.not_(NULL))
        is_not_nan = sg.not_(self.f.isnan(arg))
        if op.arg.dtype.is_floating():
            return sg.and_(is_not_null, is_not_nan)
        else:
            return is_not_null

    def visit_IsInf(self, op, *, arg):
        if op.arg.dtype.is_floating():
            return sg.or_(arg == self.POS_INF, arg == self.NEG_INF)
        return FALSE

    def visit_Xor(self, op, left, right):
        return (left | right) & ~(left & right)

    def visit_Time(self, op, *, arg):
        return arg - self.f.anon.date_trunc("day", arg)

    def visit_IntervalFromInteger(self, op, *, arg, unit):
        arg = self.f.concat(arg, sge.convert(f" {unit.plural}"))
        typ = sge.DataType(this=sge.DataType.Type.INTERVAL)
        return sg.cast(sge.convert(arg), to=typ)

    def visit_DayOfWeekIndex(self, op, *, arg):
        return (self.f.dayofweek(arg) + 5) % 7

    def visit_DayOfWeekName(self, op, *, arg):
        return sge.Case(
            this=(self.f.dayofweek(arg) + 5) % 7,
            ifs=list(itertools.starmap(self.if_, enumerate(calendar.day_name))),
        )

    def visit_ExtractDayOfYear(self, op, *, arg):
        return self.cast(self.f.dayofyear(arg), op.dtype)

    def visit_ExtractMillisecond(self, op, *, arg):
        return self.cast(self.f.date_format(arg, "SSS"), op.dtype)

    def visit_ExtractMicrosecond(self, op, *, arg):
        raise com.UnsupportedOperationError(
            "PySpark backend does not support extracting microseconds."
        )

    def visit_ExtractEpochSeconds(self, op, *, arg):
        return self.f.unix_timestamp(self.cast(arg, dt.timestamp))

    def visit_TimestampFromUNIX(self, op, *, arg, unit):
        if not op.unit:
            return self.f.to_timestamp(self.f.from_unixtime(arg))
        elif op.unit.short == "s":
            fmt = "yyyy-MM-dd HH:mm:ss"
            return self.f.to_timestamp(self.f.from_unixtime(arg, fmt), fmt)
        else:
            raise com.UnsupportedArgumentError(
                "PySpark backend does not support timestamp from unix time with "
                f"unit {op.unit.short}. Supported unit is s."
            )

    def visit_TimestampTruncate(self, op, *, arg, unit):
        if unit.short == "ns":
            raise com.UnsupportedOperationError(
                f"{unit!r} unit is not supported in timestamp {type(op)}"
            )
        return self.f.anon.date_trunc(unit.singular, arg)

    visit_TimeTruncate = visit_DateTruncate = visit_TimestampTruncate

    def visit_CountStar(self, op, *, arg, where):
        if where is not None:
            return self.f.sum(self.cast(where, op.dtype))
        return self.f.count(STAR)

    def visit_CountDistinct(self, op, *, arg, where):
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return self.f.count(sge.Distinct(expressions=[arg]))

    def visit_CountDistinctStar(self, op, *, arg, where):
        if where is None:
            return self.f.count(sge.Distinct(expressions=[STAR]))

        cols = [
            self.if_(
                where,
                sg.column(name, table=arg.alias_or_name, quoted=self.quoted),
                NULL,
            )
            for name in op.arg.schema
        ]
        return self.f.count(sge.Distinct(expressions=cols))

    def visit_First(self, op, *, arg, where):
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return self.f.first(arg, TRUE)

    def visit_Last(self, op, *, arg, where):
        if where is not None:
            arg = self.if_(where, arg, NULL)
        return self.f.last(arg, TRUE)

    def visit_Arbitrary(self, op, *, arg, how, where):
        if where is not None:
            arg = self.if_(where, arg, NULL)
        if how == "first":
            return self.f.first(arg, TRUE)
        elif how == "last":
            return self.f.last(arg, TRUE)
        else:
            raise com.UnsupportedOperationError(
                f"PySpark backend does not support arbitrary with how={how}. "
                "Supported values are `first` and `last`."
            )

    def visit_Median(self, op, *, arg, where):
        return self.agg.percentile(arg, 0.5, where=where)

    def visit_GroupConcat(self, op, *, arg, sep, where):
        if where is not None:
            arg = self.if_(where, arg, NULL)
        collected = self.f.collect_list(arg)
        collected = self.if_(self.f.size(collected).eq(0), NULL, collected)
        return self.f.array_join(collected, sep)

    def visit_Correlation(self, op, *, left, right, how, where):
        if (left_type := op.left.dtype).is_boolean():
            left = self.cast(left, dt.Int32(nullable=left_type.nullable))
        if (right_type := op.right.dtype).is_boolean():
            right = self.cast(right, dt.Int32(nullable=right_type.nullable))
        return self.agg.corr(left, right, where=where)

    def _build_sequence(self, start, stop, step, zero):
        seq = self.f.sequence(start, stop, step)
        length = self.f.size(seq)
        last_element = self.f.element_at(seq, length)
        # slice off the last element if we'd be inclusive on the right
        seq = self.if_(last_element.eq(stop), self.f.slice(seq, 1, length - 1), seq)
        return self.if_(
            step.neq(zero) & self.f.signum(step).eq(self.f.signum(stop - start)),
            seq,
            self.f.array(),
        )

    def visit_IntegerRange(self, op, *, start, stop, step):
        zero = sge.convert(0)
        return self._build_sequence(start, stop, step, zero)

    def visit_TimestampRange(self, op, *, start, stop, step):
        unit = op.step.dtype.resolution
        zero = sge.Interval(this=sge.convert(0), unit=unit)
        return self._build_sequence(start, stop, step, zero)

    def visit_Sample(
        self, op, *, parent, fraction: float, method: str, seed: int | None, **_
    ):
        if seed is not None:
            raise com.UnsupportedOperationError(
                "PySpark backend does not support sampling with seed."
            )
        sample = sge.TableSample(
            this=parent,
            percent=sge.convert(fraction * 100.0),
        )
        return sg.select(STAR).from_(sample)

    def visit_WindowBoundary(self, op, *, value, preceding):
        if isinstance(op.value, ops.Literal) and op.value.value == 0:
            value = "CURRENT ROW"
            side = None
        else:
            side = "PRECEDING" if preceding else "FOLLOWING"
        return {"value": value, "side": side}

    def __sql_name__(self, op) -> str:
        if isinstance(op, (ops.ScalarUDF, ops.AggUDF)):
            func = op.__func__
            name = op.__func_name__
        elif isinstance(op, (ops.ElementWiseVectorizedUDF, ops.ReductionVectorizedUDF)):
            func = op.func
            name = op.func.__name__
        else:
            raise TypeError(f"Cannot get SQL name for {type(op).__name__}")

        if not name.isidentifier():
            # replace invalid characters with underscores
            name = re.sub("[^0-9a-zA-Z_]", "", name)

        # generate unique name for all functions; this is necessary because
        # of lambda functions and because kwargs passed to VectorizedUDF nodes
        # are encoded as part of the closure
        name = f"{name}_{hash(func):X}"

        return f"ibis_udf_{name}"

    def visit_VectorizedUDF(self, op, *, func, func_args, input_type, return_type):
        return self.f[self.__sql_name__(op)](*func_args)

    visit_ElementWiseVectorizedUDF = visit_ReductionVectorizedUDF = visit_VectorizedUDF

    def visit_MapGet(self, op, *, arg, key, default):
        if default is None:
            return arg[key]
        else:
            return self.if_(self.f.map_contains_key(arg, key), arg[key], default)

    def visit_ArrayZip(self, op, *, arg):
        return self.f.arrays_zip(*arg)

    def visit_ArrayMap(self, op, *, arg, body, param):
        param = sge.Identifier(this=param)
        func = sge.Lambda(this=body, expressions=[param])
        return self.f.transform(arg, func)

    def visit_ArrayFilter(self, op, *, arg, body, param):
        param = sge.Identifier(this=param)
        func = sge.Lambda(this=self.if_(body, param, NULL), expressions=[param])
        transform = self.f.transform(arg, func)
        func = sge.Lambda(this=param.is_(sg.not_(NULL)), expressions=[param])
        return self.f.filter(transform, func)

    def visit_ArrayIndex(self, op, *, arg, index):
        return self.f.element_at(arg, index + 1)

    def visit_ArrayPosition(self, op, *, arg, other):
        return self.f.array_position(arg, other)

    def visit_ArrayRepeat(self, op, *, arg, times):
        return self.f.flatten(self.f.array_repeat(arg, times))

    def visit_ArraySlice(self, op, *, arg, start, stop):
        size = self.f.array_size(arg)
        start = self.if_(start < 0, self.if_(start < -size, 0, size + start), start)
        if stop is None:
            stop = size
        else:
            stop = self.if_(stop < 0, self.if_(stop < -size, 0, size + stop), stop)

        length = self.if_(stop < start, 0, stop - start)
        return self.f.slice(arg, start + 1, length)

    def visit_ArrayContains(self, op, *, arg, other):
        return self.if_(
            arg.is_(NULL),
            NULL,
            self.f.coalesce(self.f.array_contains(arg, other), FALSE),
        )

    def visit_ArrayStringJoin(self, op, *, arg, sep):
        return self.f.concat_ws(sep, arg)

    def visit_StringFind(self, op, *, arg, substr, start, end):
        if end is not None:
            raise com.UnsupportedOperationError(
                "String find doesn't support `end` argument"
            )

        if start is not None:
            arg = self.f.substr(arg, start + 1)
            pos = self.f.instr(arg, substr)
            return self.if_(pos > 0, pos + start, 0)

        return self.f.instr(arg, substr)

    def visit_RegexReplace(self, op, *, arg, pattern, replacement):
        return self.f.regexp_replace(arg, pattern, replacement)

    def visit_JSONGetItem(self, op, *, arg, index):
        if op.index.dtype.is_integer():
            fmt = "$[%s]"
        else:
            fmt = "$.%s"
        path = self.f.format_string(fmt, index)
        return self.f.get_json_object(arg, path)

    def visit_Window(self, op, *, func, group_by, order_by, **kwargs):
        if isinstance(op.func, ops.Analytic):
            # spark disallows specifying boundaries for lead/lag
            if order_by:
                order = sge.Order(expressions=order_by)
            else:
                # pyspark requires an order by clause for lag/lead
                order = sge.Order(expressions=[NULL])
            return sge.Window(this=func, partition_by=group_by, order=order)
        else:
            return super().visit_Window(
                op, func=func, group_by=group_by, order_by=order_by, **kwargs
            )

    def visit_JoinLink(self, op, **kwargs):
        if op.how == "asof":
            raise com.UnsupportedOperationError(
                "ASOF joins are not supported by Spark SQL yet and LATERAL joins "
                "raise an analysis error if the lateral subquery is limited which "
                "would be necessary to emulate ASOF joins. Once this is fixed "
                "upstream, we can add support for ASOF joins."
            )
        return super().visit_JoinLink(op, **kwargs)

    def visit_HexDigest(self, op, *, arg, how):
        if how == "md5":
            return self.f.md5(arg)
        elif how == "sha1":
            return self.f.sha1(arg)
        elif how in ("sha256", "sha512"):
            return self.f.sha2(arg, int(how[-3:]))
        else:
            raise NotImplementedError(f"No available hashing function for {how}")
