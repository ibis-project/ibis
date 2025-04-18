from __future__ import annotations

from sqlglot.dialects import Athena

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.sql.compilers.trino import TrinoCompiler
from ibis.backends.sql.datatypes import AthenaType


class AthenaCompiler(TrinoCompiler):
    __slots__ = ()
    dialect = Athena
    type_mapper = AthenaType

    UNSUPPORTED_OPS = (
        *TrinoCompiler.UNSUPPORTED_OPS,
        ops.AnalyticVectorizedUDF,
        ops.ArrayFilter,
        ops.ArrayMap,
        ops.ArrayRepeat,
        ops.BitXor,  # athena is on an older version of trino and doesn't yet support bitwise_xor_agg
        ops.ElementWiseVectorizedUDF,
        ops.Median,
        ops.Mode,
        ops.ReductionVectorizedUDF,
        ops.RowID,
        ops.TimestampBucket,
        ops.TimestampRange,
    )

    @staticmethod
    def _gen_valid_name(name: str) -> str:
        return name.replace(",", ";")

    def visit_Cast(self, op, *, arg, to):
        from_ = op.arg.dtype
        if from_.is_numeric() and to.is_timestamp():
            if from_.is_integer():
                return self.f.from_unixtime(arg)
            elif from_.is_floating():
                return self.f.from_unixtime(self.cast(arg, dt.Decimal(38, 9)))
            else:
                return self.f.from_unixtime(arg)
        return super().visit_Cast(op, arg=arg, to=to)


compiler = AthenaCompiler()
