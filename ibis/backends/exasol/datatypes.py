from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy.types as sa_types

from ibis.backends.base.sql.alchemy.datatypes import AlchemyType

if TYPE_CHECKING:
    import ibis.expr.datatypes as dt


class ExasolSQLType(AlchemyType):
    dialect = "exa.websocket"

    @classmethod
    def from_ibis(cls, dtype: dt.DataType) -> sa_types.TypeEngine:
        if dtype.is_string():
            # see also: https://docs.exasol.com/db/latest/sql_references/data_types/datatypesoverview.htm
            MAX_VARCHAR_SIZE = 2_000_000
            return sa_types.VARCHAR(MAX_VARCHAR_SIZE)
        return super().from_ibis(dtype)

    @classmethod
    def to_ibis(cls, typ: sa_types.TypeEngine, nullable: bool = True) -> dt.DataType:
        return super().to_ibis(typ, nullable=nullable)
