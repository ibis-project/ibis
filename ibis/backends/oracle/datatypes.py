from __future__ import annotations

import parsy
import sqlalchemy as sa
from sqlalchemy.dialects import oracle
from sqlalchemy.dialects.oracle.base import OracleDialect
from sqlalchemy.ext.compiler import compiles

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.common.parsing import spaceless_string


@dt.dtype.register(OracleDialect, oracle.ROWID)
def sa_oracle_rowid(_, satype, nullable=False):
    return dt.String(nullable=nullable)


# def parse(text: str) -> dt.DataType:
#     """Parse a Druid type into an ibis data type."""
#     primitive = (
#         spaceless_string("varchar2").result(dt.string)
#         | spaceless_string("binarydouble").result(dt.float64)
#         | spaceless_string("binaryfloat").result(dt.float32)
#         | spaceless_string("long").result(dt.int64)
#         | spaceless_string("rowid").result(dt.string)
#     )

#     ty = parsy.forward_declaration()

#     ty.become(primitive)
#     return ty.parse(text)
