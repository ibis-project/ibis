from __future__ import annotations

import sqlglot as sg
import sqlglot.expressions as sge

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.sql.compilers.pyspark import PySparkCompiler
from ibis.backends.sql.dialects import Databricks


class DatabricksCompiler(PySparkCompiler):
    __slots__ = ()
    dialect = Databricks

    SIMPLE_OPS = PySparkCompiler.SIMPLE_OPS | {
        ops.Divide: "try_divide",
        ops.Mode: "mode",
        ops.BitAnd: "bit_and",
        ops.BitOr: "bit_or",
        ops.BitXor: "bit_xor",
        ops.TypeOf: "typeof",
        ops.RandomUUID: "uuid",
        ops.StringSplit: "split",
    }

    UNSUPPORTED_OPS = (
        ops.ElementWiseVectorizedUDF,
        ops.AnalyticVectorizedUDF,
        ops.ReductionVectorizedUDF,
        ops.RowID,
        ops.TimestampBucket,
    )

    def visit_NonNullLiteral(self, op, *, value, dtype):
        if dtype.is_binary():
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
        return sg.column(
            self._gen_valid_name(name), table=rel.alias_or_name, quoted=self.quoted
        )


compiler = DatabricksCompiler()
