"""Beam SQL Ibis expression to SQL compiler."""

from __future__ import annotations

import sqlglot as sg
import sqlglot.expressions as sge

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.sql.compilers.base import NULL, STAR, AggGen, SQLGlotCompiler
from ibis.backends.sql.datatypes import BeamType
from ibis.backends.sql.dialects import Beam
from ibis.backends.sql.rewrites import (
    exclude_unsupported_window_frame_from_ops,
    exclude_unsupported_window_frame_from_rank,
    exclude_unsupported_window_frame_from_row_number,
    split_select_distinct_with_order_by,
)


class BeamAggGen(AggGen):
    """Aggregate function generator for Beam SQL."""
    
    def aggregate(self, compiler, name, *args, where=None, order_by=()):
        """Generate aggregate function calls."""
        if order_by:
            raise com.UnsupportedOperationError(
                "ordering of order-sensitive aggregations via `order_by` is "
                "not supported for this backend"
            )

        func = compiler.f[name]
        if where is not None:
            # Filter the aggregation
            filtered_args = []
            for arg in args:
                if hasattr(arg, 'op') and isinstance(arg.op(), ops.Value):
                    # Apply WHERE clause to the argument
                    filtered_arg = compiler.visit_Filter(
                        ops.Filter(table=arg, predicates=[where])
                    )
                    filtered_args.append(filtered_arg)
                else:
                    filtered_args.append(arg)
            args = filtered_args

        return func(*args)


class BeamCompiler(SQLGlotCompiler):
    """Beam SQL compiler."""
    
    quoted = True
    dialect = Beam
    type_mapper = BeamType

    agg = BeamAggGen()

    rewrites = (
        exclude_unsupported_window_frame_from_row_number,
        exclude_unsupported_window_frame_from_ops,
        exclude_unsupported_window_frame_from_rank,
        *SQLGlotCompiler.rewrites,
    )
    post_rewrites = (split_select_distinct_with_order_by,)

    UNSUPPORTED_OPS = (
        ops.AnalyticVectorizedUDF,
        ops.ApproxMedian,
        ops.ArrayFlatten,
        ops.ArrayStringJoin,
        ops.ArgMax,
        ops.ArgMin,
        ops.Correlation,
        ops.CountDistinctStar,
        ops.Covariance,
        ops.DateDiff,
        ops.FindInSet,
        ops.IsInf,
        ops.IsNan,
        ops.Levenshtein,
        ops.Median,
        ops.NthValue,
        ops.ReductionVectorizedUDF,
        ops.RegexSplit,
        ops.RowID,
        ops.Translate,
        ops.StringToTime,
        ops.Kurtosis,
    )

    SIMPLE_OPS = {
        ops.All: "min",
        ops.Any: "max",
        ops.ApproxCountDistinct: "approx_count_distinct",
        ops.ArrayDistinct: "array_distinct",
        ops.ArrayLength: "cardinality",
        ops.ArrayPosition: "array_position",
        ops.ArrayRemove: "array_remove",
        ops.ArraySort: "array_sort",
        ops.ArrayUnion: "array_union",
        ops.ExtractDayOfYear: "dayofyear",
        ops.MapKeys: "map_keys",
        ops.MapValues: "map_values",
        ops.Power: "power",
        ops.RegexSearch: "regexp",
        ops.StrRight: "right",
        ops.StringLength: "char_length",
        ops.StringToDate: "to_date",
        ops.StringToTimestamp: "to_timestamp",
        ops.TypeOf: "typeof",
    }

    @property
    def NAN(self):
        """Beam SQL doesn't support NaN."""
        raise NotImplementedError("Beam SQL does not support NaN")

    @property
    def POS_INF(self):
        """Beam SQL doesn't support Infinity."""
        raise NotImplementedError("Beam SQL does not support Infinity")

    NEG_INF = POS_INF

    @staticmethod
    def _generate_groups(groups):
        """Generate GROUP BY clauses."""
        return groups

    def visit_ArrayCollect(self, op, *, arg, where, order_by, include_null, distinct):
        """Visit ArrayCollect operation."""
        if order_by:
            raise com.UnsupportedOperationError(
                "ArrayCollect with order_by is not supported in Beam SQL"
            )
        
        if distinct:
            func_name = "array_agg_distinct"
        else:
            func_name = "array_agg"
        
        if where is not None:
            # Apply filter to the argument
            filtered_arg = self.visit_Filter(
                ops.Filter(table=arg, predicates=[where])
            )
            return self.f[func_name](filtered_arg)
        
        return self.f[func_name](arg)

    def visit_ArrayConcat(self, op, *, arg):
        """Visit ArrayConcat operation."""
        return self.f.array_concat(*arg)

    def visit_ArrayContains(self, op, *, arg, other):
        """Visit ArrayContains operation."""
        return self.f.array_contains(arg, other)

    def visit_ArrayIndex(self, op, *, arg, index):
        """Visit ArrayIndex operation."""
        return self.f.array_get(arg, index + 1)  # Beam SQL is 1-indexed

    def visit_ArrayRepeat(self, op, *, arg, times):
        """Visit ArrayRepeat operation."""
        return self.f.array_repeat(arg, times)

    def visit_ArraySlice(self, op, *, arg, start, stop):
        """Visit ArraySlice operation."""
        return self.f.array_slice(arg, start + 1, stop)  # Beam SQL is 1-indexed

    def visit_ArraySort(self, op, *, arg, key=None):
        """Visit ArraySort operation."""
        if key is not None:
            raise com.UnsupportedOperationError(
                "ArraySort with key function is not supported in Beam SQL"
            )
        return self.f.array_sort(arg)

    def visit_ArrayUnion(self, op, *, left, right):
        """Visit ArrayUnion operation."""
        return self.f.array_union(left, right)

    def visit_ArrayDistinct(self, op, *, arg):
        """Visit ArrayDistinct operation."""
        return self.f.array_distinct(arg)

    def visit_ArrayLength(self, op, *, arg):
        """Visit ArrayLength operation."""
        return self.f.cardinality(arg)

    def visit_ArrayPosition(self, op, *, arg, other):
        """Visit ArrayPosition operation."""
        return self.f.array_position(arg, other)

    def visit_ArrayRemove(self, op, *, arg, other):
        """Visit ArrayRemove operation."""
        return self.f.array_remove(arg, other)

    def visit_ArrayFlatten(self, op, *, arg):
        """Visit ArrayFlatten operation."""
        raise com.UnsupportedOperationError(
            "ArrayFlatten is not supported in Beam SQL"
        )

    def visit_ArrayStringJoin(self, op, *, arg, sep):
        """Visit ArrayStringJoin operation."""
        raise com.UnsupportedOperationError(
            "ArrayStringJoin is not supported in Beam SQL"
        )

    def visit_MapKeys(self, op, *, arg):
        """Visit MapKeys operation."""
        return self.f.map_keys(arg)

    def visit_MapValues(self, op, *, arg):
        """Visit MapValues operation."""
        return self.f.map_values(arg)

    def visit_MapGet(self, op, *, arg, key, default=None):
        """Visit MapGet operation."""
        if default is not None:
            return self.f.coalesce(self.f.map_get(arg, key), default)
        return self.f.map_get(arg, key)

    def visit_MapContains(self, op, *, arg, key):
        """Visit MapContains operation."""
        return self.f.map_contains(arg, key)

    def visit_MapMerge(self, op, *, left, right):
        """Visit MapMerge operation."""
        return self.f.map_merge(left, right)

    def visit_MapConcat(self, op, *, arg):
        """Visit MapConcat operation."""
        return self.f.map_concat(*arg)

    def visit_MapFromArrays(self, op, *, keys, values):
        """Visit MapFromArrays operation."""
        return self.f.map_from_arrays(keys, values)

    def visit_MapFromEntries(self, op, *, arg):
        """Visit MapFromEntries operation."""
        return self.f.map_from_entries(arg)

    def visit_MapEntries(self, op, *, arg):
        """Visit MapEntries operation."""
        return self.f.map_entries(arg)

    def visit_MapKeys(self, op, *, arg):
        """Visit MapKeys operation."""
        return self.f.map_keys(arg)

    def visit_MapValues(self, op, *, arg):
        """Visit MapValues operation."""
        return self.f.map_values(arg)

    def visit_MapGet(self, op, *, arg, key, default=None):
        """Visit MapGet operation."""
        if default is not None:
            return self.f.coalesce(self.f.map_get(arg, key), default)
        return self.f.map_get(arg, key)

    def visit_MapContains(self, op, *, arg, key):
        """Visit MapContains operation."""
        return self.f.map_contains(arg, key)

    def visit_MapMerge(self, op, *, left, right):
        """Visit MapMerge operation."""
        return self.f.map_merge(left, right)

    def visit_MapConcat(self, op, *, arg):
        """Visit MapConcat operation."""
        return self.f.map_concat(*arg)

    def visit_MapFromArrays(self, op, *, keys, values):
        """Visit MapFromArrays operation."""
        return self.f.map_from_arrays(keys, values)

    def visit_MapFromEntries(self, op, *, arg):
        """Visit MapFromEntries operation."""
        return self.f.map_from_entries(arg)

    def visit_MapEntries(self, op, *, arg):
        """Visit MapEntries operation."""
        return self.f.map_entries(arg)

    def visit_Strip(self, op, *, arg):
        """Visit Strip operation."""
        # Beam SQL doesn't have BTRIM, so we use a combination of left and right trim
        return self.visit_RStrip(op, arg=self.visit_LStrip(op, arg=arg))


compiler = BeamCompiler()
