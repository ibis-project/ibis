from __future__ import annotations

import sqlglot.expressions as sge

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.sql.compilers.mysql import MySQLCompiler
from ibis.backends.sql.dialects import MySQL
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

    dialect = MySQL  # SingleStoreDB uses MySQL dialect
    type_mapper = MySQLCompiler.type_mapper  # Use MySQL type mapper for now
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
        # Add any SingleStoreDB-specific unsupported operations here
        # Note: SingleStoreDB may support some operations that MySQL doesn't
        # and vice versa, but for now we use the MySQL set as baseline
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

    def visit_Cast(self, op, *, arg, to):
        """Handle casting operations in SingleStoreDB."""
        from_ = op.arg.dtype
        if (from_.is_json() or from_.is_string()) and to.is_json():
            # SingleStoreDB handles JSON casting similarly to MySQL/MariaDB
            return arg
        elif from_.is_numeric() and to.is_timestamp():
            return self.if_(
                arg.eq(0),
                self.f.timestamp("1970-01-01 00:00:00"),
                self.f.from_unixtime(arg),
            )
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
            return self.f.date(value.isoformat())
        elif dtype.is_timestamp():
            return self.f.timestamp(value.isoformat())
        elif dtype.is_time():
            return self.f.maketime(
                value.hour, value.minute, value.second + value.microsecond / 1e6
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
        """Handle JSON path extraction in SingleStoreDB."""
        if op.index.dtype.is_integer():
            path = self.f.concat("$[", self.cast(index, dt.string), "]")
        else:
            path = self.f.concat("$.", index)
        return self.f.json_extract(arg, path)

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
            return self.f.locate(substr, arg, start + 1)
        return self.f.locate(substr, arg)

    # Distributed query features - SingleStoreDB specific
    def _add_shard_key_hint(self, query, shard_key=None):
        """Add SingleStoreDB shard key hints for distributed queries.

        This is a placeholder for future SingleStoreDB-specific optimization.
        """
        # Implementation would depend on SingleStoreDB's distributed query syntax
        return query

    def _optimize_for_columnstore(self, query):
        """Optimize queries for SingleStoreDB columnstore tables.

        This is a placeholder for future SingleStoreDB-specific optimization.
        """
        # Implementation would depend on SingleStoreDB's columnstore optimizations
        return query


# Create the compiler instance
compiler = SingleStoreDBCompiler()
