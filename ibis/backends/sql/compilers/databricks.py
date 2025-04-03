from __future__ import annotations

import sqlglot as sg
import sqlglot.expressions as sge

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.sql.compilers.base import NULL
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
        ops.JSONGetItem: "json_extract",
    }
    del (
        SIMPLE_OPS[ops.UnwrapJSONString],
        SIMPLE_OPS[ops.UnwrapJSONInt64],
        SIMPLE_OPS[ops.UnwrapJSONFloat64],
        SIMPLE_OPS[ops.UnwrapJSONBoolean],
    )

    UNSUPPORTED_OPS = (
        ops.ElementWiseVectorizedUDF,
        ops.AnalyticVectorizedUDF,
        ops.ReductionVectorizedUDF,
        ops.RowID,
        ops.TimestampBucket,
    )

    def visit_ToJSONArray(self, op, *, arg):
        return self.f.try_variant_get(arg, "$", "ARRAY<VARIANT>")

    def visit_ToJSONMap(self, op, *, arg):
        return self.f.try_variant_get(arg, "$", "MAP<STRING, VARIANT>")

    def visit_UnwrapJSONString(self, op, *, arg):
        return self.if_(
            self.f.schema_of_variant(arg).eq(sge.convert("STRING")),
            self.f.try_variant_get(arg, "$", "STRING"),
            NULL,
        )

    def visit_UnwrapJSONInt64(self, op, *, arg):
        return self.if_(
            self.f.schema_of_variant(arg).eq(sge.convert("BIGINT")),
            self.f.try_variant_get(arg, "$", "BIGINT"),
            NULL,
        )

    def visit_UnwrapJSONFloat64(self, op, *, arg):
        return self.if_(
            self.f.schema_of_variant(arg).isin(
                sge.convert("STRING"), sge.convert("BOOLEAN")
            ),
            NULL,
            self.f.try_variant_get(arg, "$", "DOUBLE"),
        )

    def visit_UnwrapJSONBoolean(self, op, *, arg):
        return self.if_(
            self.f.schema_of_variant(arg).eq(sge.convert("BOOLEAN")),
            self.f.try_variant_get(arg, "$", "BOOLEAN"),
            NULL,
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
