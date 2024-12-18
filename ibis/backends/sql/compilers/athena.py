from __future__ import annotations

import re

from sqlglot.dialects import Athena

import ibis.expr.operations as ops
from ibis.backends.sql.compilers.trino import TrinoCompiler
from ibis.backends.sql.datatypes import AthenaType

_NAME_REGEX = re.compile(r'[^!"$()*,./;?@[\\\]^`{}~\n]+')


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
        return "_".join(map(str.strip, _NAME_REGEX.findall(name))) or "tmp"


compiler = AthenaCompiler()
