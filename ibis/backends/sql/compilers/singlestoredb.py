from __future__ import annotations

import sqlglot.expressions as sge
from sqlglot.dialects.singlestore import SingleStore

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.singlestoredb.datatypes import SingleStoreDBType
from ibis.backends.sql.compilers.mysql import MySQLCompiler
from ibis.backends.sql.rewrites import (
    exclude_unsupported_window_frame_from_ops,
    exclude_unsupported_window_frame_from_rank,
    exclude_unsupported_window_frame_from_row_number,
    rewrite_empty_order_by_window,
)
from ibis.common.patterns import replace
from ibis.expr.rewrites import p


@replace(p.Limit)
def rewrite_limit(_, **kwargs):
    """Rewrite limit for SingleStoreDB to include a large upper bound.

    SingleStoreDB uses the MySQL protocol, so this follows the same pattern.
    """
    if _.n is None and _.offset is not None:
        some_large_number = (1 << 64) - 1
        return _.copy(n=some_large_number)
    return _


class SingleStoreDBCompiler(MySQLCompiler):
    """SQL compiler for SingleStoreDB.

    SingleStoreDB is MySQL protocol compatible, so we inherit most functionality
    from MySQLCompiler and override only SingleStoreDB-specific behaviors.
    """

    __slots__ = ()

    dialect = SingleStore  # SingleStoreDB uses SingleStore dialect in SQLGlot
    type_mapper = SingleStoreDBType  # Use SingleStoreDB-specific type mapper
    rewrites = (
        rewrite_limit,
        exclude_unsupported_window_frame_from_ops,
        exclude_unsupported_window_frame_from_rank,
        exclude_unsupported_window_frame_from_row_number,
        rewrite_empty_order_by_window,
        *MySQLCompiler.rewrites,
    )

    # SingleStoreDB has some differences from MySQL in supported operations
    UNSUPPORTED_OPS = (
        # Inherit MySQL unsupported ops
        *MySQLCompiler.UNSUPPORTED_OPS,
        # Add SingleStoreDB-specific unsupported operations
        ops.HexDigest,  # HexDigest not supported in SingleStoreDB
        ops.Hash,  # Hash function not available
        ops.First,  # First aggregate not supported
        ops.Last,  # Last aggregate not supported
    )

    # SingleStoreDB supports most MySQL simple operations
    # Override here if there are SingleStoreDB-specific function names
    SIMPLE_OPS = {
        **MySQLCompiler.SIMPLE_OPS,
        # Add SingleStoreDB-specific function mappings here
        # For example, if SingleStoreDB has different function names:
        # ops.SomeOperation: "singlestoredb_function_name",
    }

    @property
    def NAN(self):
        raise NotImplementedError("SingleStoreDB does not support NaN")

    @property
    def POS_INF(self):
        raise NotImplementedError("SingleStoreDB does not support Infinity")

    NEG_INF = POS_INF

    def visit_Date(self, op, *, arg):
        """Extract the date part from a timestamp or date value."""
        # Use DATE() function for SingleStoreDB, which is MySQL-compatible
        # Create an anonymous function call since SQLGlot's f.date creates a cast
        return sge.Anonymous(this="DATE", expressions=[arg])

    def visit_Cast(self, op, *, arg, to):
        """Handle casting operations in SingleStoreDB.

        Includes support for SingleStoreDB-specific types like VECTOR and enhanced JSON.
        """
        from_ = op.arg.dtype

        # JSON casting - SingleStoreDB has enhanced JSON support
        if from_.is_json() and to.is_json():
            # JSON to JSON cast is a no-op
            return arg
        elif from_.is_string() and to.is_json():
            # Cast string to JSON with validation
            return self.cast(arg, to)

        # Timestamp casting
        elif from_.is_numeric() and to.is_timestamp():
            return self.if_(
                arg.eq(0),
                sge.Anonymous(this="TIMESTAMP", expressions=["1970-01-01 00:00:00"]),
                self.f.from_unixtime(arg),
            )

        # Binary casting (includes VECTOR type support)
        elif from_.is_string() and to.is_binary():
            # Cast string to binary/VECTOR - useful for VECTOR type data
            return sge.Anonymous(this="UNHEX", expressions=[arg])
        elif from_.is_binary() and to.is_string():
            # Cast binary/VECTOR to string representation
            return sge.Anonymous(this="HEX", expressions=[arg])

        # Geometry casting
        elif to.is_geospatial():
            # SingleStoreDB GEOMETRY type casting
            return sge.Anonymous(
                this="ST_GEOMFROMTEXT", expressions=[self.cast(arg, dt.string)]
            )
        elif from_.is_geospatial() and to.is_string():
            return sge.Anonymous(this="ST_ASTEXT", expressions=[arg])

        return super().visit_Cast(op, arg=arg, to=to)

    def visit_NonNullLiteral(self, op, *, value, dtype):
        """Handle non-null literal values for SingleStoreDB."""
        if dtype.is_decimal() and not value.is_finite():
            raise com.UnsupportedOperationError(
                "SingleStoreDB does not support NaN or infinity"
            )
        elif dtype.is_binary():
            return self.f.unhex(value.hex())
        elif dtype.is_date():
            return sge.Anonymous(this="DATE", expressions=[value.isoformat()])
        elif dtype.is_timestamp():
            return sge.Anonymous(this="TIMESTAMP", expressions=[value.isoformat()])
        elif dtype.is_time():
            return sge.Anonymous(
                this="MAKETIME",
                expressions=[
                    value.hour,
                    value.minute,
                    value.second + value.microsecond / 1e6,
                ],
            )
        elif dtype.is_array() or dtype.is_struct() or dtype.is_map():
            # SingleStoreDB has some JSON support for these types
            # For now, treat them as unsupported like MySQL
            raise com.UnsupportedBackendType(
                "SingleStoreDB does not fully support arrays, structs or maps yet"
            )
        return None

    # SingleStoreDB-specific methods can be added here
    def visit_SingleStoreDBSpecificOp(self, op, **kwargs):
        """Example of a SingleStoreDB-specific operation handler.

        This would be used for operations that are unique to SingleStoreDB,
        such as distributed query hints, shard key operations, etc.
        """
        raise NotImplementedError(
            "SingleStoreDB-specific operations not yet implemented"
        )

    # JSON operations - SingleStoreDB may have enhanced JSON support
    def visit_JSONGetItem(self, op, *, arg, index):
        """Handle JSON path extraction in SingleStoreDB using JSON_EXTRACT_JSON."""
        if op.index.dtype.is_integer():
            # For array indices, SingleStoreDB JSON_EXTRACT_JSON expects just the number
            path = index
        else:
            # For object keys, SingleStoreDB JSON_EXTRACT_JSON expects just the key name
            path = index
        # Use JSON_EXTRACT_JSON function (SingleStoreDB-specific)
        return sge.Anonymous(this="JSON_EXTRACT_JSON", expressions=[arg, path])

    def visit_Sign(self, op, *, arg):
        """Handle SIGN function to ensure consistent return type."""
        # SingleStoreDB's SIGN function returns DECIMAL, but tests expect FLOAT
        # Cast to DOUBLE to match NumPy's float64 behavior
        sign_func = sge.Anonymous(this="SIGN", expressions=[arg])
        return self.cast(sign_func, dt.Float64())

    # Window functions - SingleStoreDB may have better support than MySQL
    @staticmethod
    def _minimize_spec(op, spec):
        """Handle window function specifications for SingleStoreDB."""
        if isinstance(
            op.func, (ops.RankBase, ops.CumeDist, ops.NTile, ops.PercentRank)
        ):
            return None
        return spec

    # String operations - SingleStoreDB follows MySQL pattern
    def visit_StringFind(self, op, *, arg, substr, start, end):
        """Handle string find operations in SingleStoreDB."""
        if end is not None:
            raise NotImplementedError(
                "`end` argument is not implemented for SingleStoreDB `StringValue.find`"
            )
        substr = sge.Cast(this=substr, to=sge.DataType(this=sge.DataType.Type.BINARY))

        if start is not None:
            return sge.Anonymous(this="LOCATE", expressions=[substr, arg, start + 1])
        return sge.Anonymous(this="LOCATE", expressions=[substr, arg])

    # Distributed query features - SingleStoreDB specific
    def _add_shard_key_hint(self, query, shard_key=None):
        """Add SingleStore shard key hints for distributed queries."""
        if shard_key is None:
            return query

        # For SingleStore, we can add hints as SQL comments for optimization
        # This adds a query hint for shard key optimization
        hint = f"/*+ SHARD_KEY({shard_key}) */"

        # Convert query to string if it's a SQLGlot object
        query_str = query.sql(self.dialect) if hasattr(query, "sql") else str(query)

        # Insert hint after SELECT keyword
        if query_str.strip().upper().startswith("SELECT"):
            parts = query_str.split(" ", 1)
            if len(parts) >= 2:
                return f"{parts[0]} {hint} {parts[1]}"
            else:
                return f"{parts[0]} {hint}"

        return query_str

    def _optimize_for_columnstore(self, query):
        """Optimize queries for SingleStore columnstore tables."""
        # Convert query to string if it's a SQLGlot object
        query_str = query.sql(self.dialect) if hasattr(query, "sql") else str(query)

        # Add hints for columnstore optimization
        hint = "/*+ USE_COLUMNSTORE_STRATEGY */"

        # Insert hint after SELECT keyword
        if query_str.strip().upper().startswith("SELECT"):
            parts = query_str.split(" ", 1)
            if len(parts) >= 2:
                return f"{parts[0]} {hint} {parts[1]}"
            else:
                return f"{parts[0]} {hint}"

        return query_str


# Create the compiler instance
compiler = SingleStoreDBCompiler()
