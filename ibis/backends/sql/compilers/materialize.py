"""Materialize SQL compiler.

This module provides Materialize-specific SQL compilation overrides.
Materialize uses PostgreSQL wire protocol but has different function implementations.

For best practices and recommended patterns when writing Materialize SQL, see:
https://materialize.com/docs/transform-data/idiomatic-materialize-sql/
"""

from __future__ import annotations

import sqlglot as sg
import sqlglot.expressions as sge

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.sql.compilers.base import NULL, STAR
from ibis.backends.sql.compilers.postgres import PostgresCompiler
from ibis.backends.sql.datatypes import MaterializeType


class MaterializeCompiler(PostgresCompiler):
    """Materialize SQL compiler with custom function translations.

    Materialize is based on PostgreSQL but doesn't support all PostgreSQL functions.
    This compiler provides Materialize-specific implementations.

    Special handling:
    - First/Last aggregates: Materialize doesn't support FIRST()/LAST() aggregate functions.
      We attempt to rewrite aggregates using only First() into DISTINCT ON queries where possible.
    """

    __slots__ = ()

    type_mapper = MaterializeType

    UNSUPPORTED_OPS = (
        *PostgresCompiler.UNSUPPORTED_OPS,
        ops.TimeDelta,
        # Materialize doesn't support percentile/quantile functions
        ops.Median,
        ops.ApproxMedian,
        ops.Quantile,
        ops.ApproxQuantile,
        ops.MultiQuantile,
        ops.ApproxMultiQuantile,
        # Materialize doesn't support bitwise aggregate functions
        ops.BitAnd,
        ops.BitOr,
        ops.BitXor,
        # Materialize doesn't support correlation/covariance functions
        ops.Correlation,
        ops.Covariance,
        # Materialize doesn't support statistical aggregate functions
        ops.Mode,
        ops.Kurtosis,
        ops.Arbitrary,
        # Materialize doesn't support first/last aggregate functions
        # Note: We handle First specially in visit_Aggregate by rewriting to DISTINCT ON
        # ops.First is NOT in UNSUPPORTED_OPS - we handle it
        ops.Last,  # Last is still unsupported
        # Materialize doesn't support certain window functions
        ops.PercentRank,
        ops.CumeDist,
        ops.NTile,
        ops.NthValue,
        # Materialize doesn't support geospatial operations
        ops.GeoDistance,
        ops.GeoAsText,
        ops.GeoUnaryUnion,
        # Materialize doesn't support impure/non-deterministic functions
        ops.RandomScalar,
        ops.RandomUUID,
        # Materialize doesn't support rowid
        ops.RowID,
    )

    def visit_ArrayLength(self, op, *, arg):
        """Compile ArrayLength operation.

        Materialize uses array_length(array, dimension) instead of cardinality().
        For 1-dimensional arrays, we use dimension 1.
        """
        # Use array_length with dimension 1 for standard arrays
        # Use Anonymous to prevent sqlglot from transforming the function name
        return sge.Anonymous(this="array_length", expressions=[arg, sge.convert(1)])

    def visit_ArrayRemove(self, op, *, arg, other):
        """Compile ArrayRemove operation.

        Materialize doesn't support array_remove().
        """
        raise NotImplementedError("array_remove is not available in Materialize")

    def visit_ArrayRepeat(self, op, *, arg, times):
        """Compile ArrayRepeat operation.

        Override PostgreSQL to use array_length instead of cardinality.
        Materialize's generate_series returns results unordered, so we need ORDER BY.
        """
        i = sg.to_identifier("i")
        # Use array_length(array, 1) instead of cardinality
        length = sge.Anonymous(this="array_length", expressions=[arg, sge.convert(1)])
        return self.f.array(
            sg.select(arg[i % length + 1])
            .from_(self.f.generate_series(0, length * times - 1).as_(i.name))
            .order_by(i)
        )

    def visit_ArrayDistinct(self, op, *, arg):
        """Compile ArrayDistinct operation.

        Use unnest and array_agg with DISTINCT to get unique elements.
        """
        # Materialize supports ARRAY(SELECT DISTINCT UNNEST(array))
        return self.f.array(
            sge.Select(expressions=[sge.Distinct(expressions=[self.f.unnest(arg)])])
        )

    def visit_ArrayUnion(self, op, *, left, right):
        """Compile ArrayUnion operation.

        Concatenate arrays and remove duplicates.
        """
        # Use array_cat to concatenate, then get distinct elements
        concatenated = self.f.array_cat(left, right)
        return self.f.array(
            sge.Select(
                expressions=[sge.Distinct(expressions=[self.f.unnest(concatenated)])]
            )
        )

    def visit_ArrayIndex(self, op, *, arg, index):
        """Compile ArrayIndex operation.

        Override PostgreSQL to use array_length instead of cardinality.
        """
        # Use array_length(array, 1) instead of cardinality
        arg_length = sge.Anonymous(
            this="array_length", expressions=[arg, sge.convert(1)]
        )
        index = self.if_(index < 0, arg_length + index, index)
        return sge.paren(arg, copy=False)[index]

    def visit_ArraySlice(self, op, *, arg, start, stop):
        """Compile ArraySlice operation.

        Override PostgreSQL to use array_length instead of cardinality.
        """
        neg_to_pos_index = lambda n, index: self.if_(index < 0, n + index, index)

        # Use array_length(array, 1) instead of cardinality
        arg_length = sge.Anonymous(
            this="array_length", expressions=[arg, sge.convert(1)]
        )

        if start is None:
            start = 0
        else:
            start = self.f.least(arg_length, neg_to_pos_index(arg_length, start))

        if stop is None:
            stop = arg_length
        else:
            stop = neg_to_pos_index(arg_length, stop)

        slice_expr = sge.Slice(this=start + 1, expression=stop)
        return sge.paren(arg, copy=False)[slice_expr]

    def visit_Sign(self, op, *, arg):
        """Compile Sign operation.

        Materialize doesn't have a sign() function, so use CASE WHEN logic.
        Returns -1 for negative, 0 for zero, 1 for positive.
        """
        return self.if_(arg.eq(0), 0, self.if_(arg > 0, 1, -1))

    def visit_Range(self, op, *, start, stop, step):
        """Compile Range operation.

        Override PostgreSQL's visit_Range to avoid using sign() function
        which doesn't exist in Materialize.
        """

        def zero_value(dtype):
            if dtype.is_interval():
                # Use CAST('0 seconds' AS INTERVAL) for zero interval
                return sge.Cast(
                    this=sge.convert("0 seconds"),
                    to=sge.DataType(this=sge.DataType.Type.INTERVAL),
                )
            return 0

        def interval_sign(v):
            # Use CAST('0 seconds' AS INTERVAL) for zero interval
            zero = sge.Cast(
                this=sge.convert("0 seconds"),
                to=sge.DataType(this=sge.DataType.Type.INTERVAL),
            )
            return sge.Case(
                ifs=[
                    self.if_(v.eq(zero), 0),
                    self.if_(v < zero, -1),
                    self.if_(v > zero, 1),
                ],
                default=NULL,
            )

        def _sign(value, dtype):
            """Custom sign implementation without using sign() function."""
            if dtype.is_interval():
                return interval_sign(value)
            # Use CASE WHEN instead of sign() function
            return self.if_(value.eq(0), 0, self.if_(value > 0, 1, -1))

        step_dtype = op.step.dtype
        # Build the array result from generate_series
        # Don't use array_remove since it's not available in Materialize
        # Instead, filter the results to exclude the stop value
        # Materialize's generate_series returns results unordered, so we need ORDER BY
        # Order by value * sign(step) to get correct direction (ASC for positive, DESC for negative)
        series_alias = sg.to_identifier("gs")

        # For timestamps, we need to extract epoch before multiplying by sign
        # For numeric types, we can multiply directly
        if op.start.dtype.is_timestamp():
            order_expr = self.f.extract("EPOCH", series_alias) * _sign(step, step_dtype)
        else:
            order_expr = series_alias * _sign(step, step_dtype)

        array_result = self.f.array(
            sg.select(series_alias)
            .from_(self.f.generate_series(start, stop, step).as_(series_alias.name))
            .where(series_alias.neq(stop))
            .order_by(order_expr)
        )
        # Cast both branches to the same type to avoid CASE type mismatch
        return self.if_(
            sg.and_(
                self.f.nullif(step, zero_value(step_dtype)).is_(sg.not_(NULL)),
                _sign(step, step_dtype).eq(_sign(stop - start, step_dtype)),
            ),
            self.cast(array_result, op.dtype),
            self.cast(self.f.array(), op.dtype),
        )

    visit_IntegerRange = visit_TimestampRange = visit_Range

    def visit_TimestampFromUNIX(self, op, *, arg, unit):
        """Compile TimestampFromUNIX operation.

        Materialize uses to_timestamp(double precision) which expects seconds.
        Convert other units to seconds before calling to_timestamp.
        """
        unit_short = unit.short
        if unit_short == "s":
            # Already in seconds
            timestamp_arg = arg
        elif unit_short == "ms":
            # Convert milliseconds to seconds
            timestamp_arg = arg / 1e3
        elif unit_short == "us":
            # Convert microseconds to seconds
            timestamp_arg = arg / 1e6
        elif unit_short == "ns":
            # Convert nanoseconds to seconds
            timestamp_arg = arg / 1e9
        else:
            raise com.UnsupportedOperationError(
                f"Unit {unit_short!r} is not supported for TimestampFromUNIX"
            )

        result = self.f.to_timestamp(timestamp_arg)

        # Apply timezone if specified
        if (timezone := op.dtype.timezone) is not None:
            result = self.f.timezone(timezone, result)

        return result

    def visit_DateFromYMD(self, op, *, year, month, day):
        """Compile DateFromYMD operation.

        Materialize doesn't have make_date(), so construct a date string and cast it.
        Format: 'YYYY-MM-DD'
        """
        # Use lpad to ensure proper formatting with leading zeros
        year_str = self.cast(year, dt.string)
        month_str = self.f.lpad(self.cast(month, dt.string), 2, "0")
        day_str = self.f.lpad(self.cast(day, dt.string), 2, "0")

        # Concatenate into 'YYYY-MM-DD' format
        date_str = self.f.concat(
            year_str, sge.convert("-"), month_str, sge.convert("-"), day_str
        )

        # Cast to date
        return self.cast(date_str, dt.date)

    def visit_NonNullLiteral(self, op, *, value, dtype):
        """Override date literal handling.

        Materialize doesn't have make_date(), so we need to handle date literals
        by casting from ISO format strings instead of using datefromparts.
        """
        if dtype.is_date():
            # Use ISO format string and cast to date
            return self.cast(value.isoformat(), dtype)

        # Delegate to parent for other types
        return super().visit_NonNullLiteral(op, value=value, dtype=dtype)

    def _make_interval(self, arg, unit):
        """Override PostgreSQL's _make_interval to use INTERVAL string literals.

        Materialize doesn't support make_interval() function.
        Instead, use CAST(arg || ' unit' AS INTERVAL) syntax.
        """
        plural = unit.plural

        # Materialize doesn't support 'weeks', convert to days
        if plural == "weeks":
            arg = arg * 7
            plural = "days"

        # Map plural to Materialize interval unit names
        unit_map = {
            "years": "year",
            "months": "month",
            "days": "day",
            "hours": "hour",
            "minutes": "minute",
            "seconds": "second",
            "milliseconds": "millisecond",
            "microseconds": "microsecond",
            "nanoseconds": "nanosecond",
        }

        unit_str = unit_map.get(plural, plural.rstrip("s"))

        # Convert arg to string and concatenate with unit
        # CAST(arg::text || ' days' AS INTERVAL)
        arg_str = self.cast(arg, dt.string)
        interval_str = self.f.concat(arg_str, sge.convert(f" {unit_str}s"))

        # Cast to INTERVAL without unit specification (Materialize doesn't support INTERVAL DAY syntax)
        # Use sqlglot DataType directly to avoid ibis validation
        return sge.Cast(
            this=interval_str, to=sge.DataType(this=sge.DataType.Type.INTERVAL)
        )

    # JSON operations
    def json_extract_path_text(self, op, arg, *rest):
        """Extract JSON value as text using Materialize operators.

        Materialize doesn't have json_extract_path_text or jsonb_extract_path_text.
        Use the #>> operator instead which extracts as text by path.
        """
        if not rest:
            # No path specified, just return the arg cast to text
            return self.cast(arg, dt.string)

        # Build array of path elements for #>> operator
        # The #>> operator takes a path as TEXT[] array
        path_array = self.f.array(*rest)
        return sge.JSONExtractScalar(this=arg, expression=path_array)

    def visit_DateNow(self, op):
        """Compile DateNow operation.

        Materialize doesn't support CURRENT_DATE, use NOW()::date instead.
        """
        return self.cast(self.f.now(), dt.date)

    def json_typeof(self, op, arg):
        """Get JSON type using Materialize's jsonb_typeof function.

        Materialize only has jsonb_typeof, not json_typeof.
        """
        # Always use jsonb_typeof regardless of binary flag
        return self.f.jsonb_typeof(arg)

    def visit_MzNow(self, op):
        """Compile MzNow operation to mz_now() function call.

        Returns Materialize's logical timestamp, which is used for temporal
        filters in materialized views and streaming queries.

        Note: We cast mz_now() to TIMESTAMPTZ because mz_timestamp is an opaque
        internal type that's represented as microseconds since Unix epoch.
        Casting it makes it compatible with standard timestamp operations.

        For best performance in streaming queries, mz_now() should be isolated
        on one side of temporal filter comparisons. For example:
            Good: mz_now() > created_at + INTERVAL '1 day'
            Bad:  mz_now() - created_at > INTERVAL '1 day'

        References
        ----------
        - Function: https://materialize.com/docs/sql/functions/now_and_mz_now/
        - Type: https://materialize.com/docs/sql/types/mz_timestamp/
        - Idiomatic patterns: https://materialize.com/docs/transform-data/idiomatic-materialize-sql/#temporal-filters
        """
        # Cast mz_now() to TIMESTAMPTZ for compatibility with standard timestamp operations
        return self.cast(self.f.mz_now(), dt.Timestamp(timezone="UTC"))

    def visit_Aggregate(self, op, *, parent, groups, metrics):
        """Compile aggregate with special handling for First aggregates.

        Materialize doesn't support FIRST()/LAST() aggregate functions.
        When ALL metrics are First() aggregates, we rewrite to DISTINCT ON.

        Example transformation:
          SELECT category, FIRST(value), FIRST(name)
          FROM table
          GROUP BY category
        Becomes:
          SELECT DISTINCT ON (category) category, value, name
          FROM table
          ORDER BY category
        """
        # Check if all metrics are First operations
        # Access the original operation objects from op.metrics
        all_first = all(
            isinstance(metric_val, ops.First) for metric_val in op.metrics.values()
        )

        if all_first and groups:
            # Rewrite to DISTINCT ON
            # The metrics dict already contains column references (from visit_First)
            # which is exactly what we need for DISTINCT ON
            sel = sg.select(
                *self._cleanup_names(groups), *self._cleanup_names(metrics), copy=False
            ).from_(parent, copy=False)

            # Add DISTINCT ON clause
            group_keys = list(groups.values())
            sel = sel.distinct(*group_keys, copy=False)

            # Add ORDER BY for the group keys to make DISTINCT ON deterministic
            # DISTINCT ON requires ORDER BY to start with the DISTINCT ON columns
            order_exprs = list(group_keys)

            # Add ORDER BY from First operations if specified
            order_exprs.extend(
                self.visit(sort_key, {})
                for metric_val in op.metrics.values()
                if isinstance(metric_val, ops.First) and metric_val.order_by
                for sort_key in metric_val.order_by
            )

            if order_exprs:
                sel = sel.order_by(*order_exprs, copy=False)

            return sel

        # Check if any metrics contain First (mixed case - not supported)
        has_first = any(
            isinstance(metric_val, ops.First) for metric_val in op.metrics.values()
        )

        if has_first:
            # Mixed aggregates with First - not supported
            raise com.UnsupportedOperationError(
                "Materialize doesn't support FIRST() aggregate function. "
                "Use distinct(on=..., keep='first') for Top-1 queries where ALL "
                "aggregates are first(), or use window functions instead."
            )

        # Fall back to default behavior (will fail for First/Last)
        return super().visit_Aggregate(
            op, parent=parent, groups=groups, metrics=metrics
        )

    def visit_First(self, op, *, arg, where, order_by, include_null):
        """Compile First operation.

        Materialize doesn't support the FIRST() aggregate function.
        However, visit_Aggregate will rewrite aggregations with ONLY First() into DISTINCT ON.

        For DISTINCT ON to work, we return just the column reference here.
        If First is used incorrectly (mixed with other aggregates), visit_Aggregate will catch it.
        """
        if include_null:
            raise com.UnsupportedOperationError(
                "`include_null=True` is not supported by Materialize"
            )
        if where is not None:
            raise com.UnsupportedOperationError(
                "First() with WHERE clause is not supported by Materialize. "
                "Use filter() before distinct() instead."
            )

        # Return the column reference
        # visit_Aggregate will either:
        # 1. Use it in DISTINCT ON (if all metrics are First), OR
        # 2. Raise an error (if mixed with other aggregates)
        return arg

    def visit_JSONGetItem(self, op, *, arg, index):
        """Compile JSON get item operation using -> operator.

        Materialize supports -> operator for JSON access.
        Materialize doesn't have jsonb_extract_path, use -> operator instead.
        """
        # Use Python bracket notation like BigQuery
        # sqlglot will convert this to the appropriate operator for the dialect
        return arg[index]

    def visit_ToJSONArray(self, op, *, arg):
        """Convert JSON array to SQL array using jsonb_array_elements.

        Materialize doesn't support the same array construction as PostgreSQL
        for JSON arrays. Use jsonb_array_elements.
        """
        return self.if_(
            self.f.jsonb_typeof(arg).eq(sge.convert("array")),
            self.f.array(sg.select(STAR).from_(self.f.jsonb_array_elements(arg))),
            NULL,
        )


compiler = MaterializeCompiler()
