from __future__ import annotations

import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.schema as sch
from ibis.backends.base import Database


class AlchemyDatabase(Database):
    """SQLAlchemy-based database class."""

    def table(self, name, schema=None):
        return self.client.table(name, schema=schema)


class AlchemyTable(ops.DatabaseTable):
    sqla_table = rlz.instance_of(object)
    name = rlz.optional(rlz.instance_of(str), default=None)
    schema = rlz.optional(rlz.instance_of(sch.Schema), default=None)

    def __init__(self, source, sqla_table, name, schema):
        if name is None:
            name = sqla_table.name
        if schema is None:
            schema = sch.infer(sqla_table, schema=schema, dialect=source.con.dialect)
        super().__init__(name=name, schema=schema, sqla_table=sqla_table, source=source)

    # TODO(kszucs): remove this
    def __equals__(self, other: AlchemyTable) -> bool:
        # override the default implementation to not compare
        # sqla_table instances
        return (
            self.name == other.name
            and self.source == other.source
            and self.schema.equals(other.schema)
        )
