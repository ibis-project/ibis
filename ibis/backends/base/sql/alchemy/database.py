import ibis.expr.operations as ops
import ibis.expr.schema as sch
from ibis.backends.base import Database


class AlchemyDatabase(Database):
    """

    Attributes
    ----------
    client : AlchemyClient

    """

    def table(self, name, schema=None):
        return self.client.table(name, schema=schema)


class AlchemyTable(ops.DatabaseTable):
    def __init__(self, table, source, schema=None):
        schema = sch.infer(table, schema=schema)
        super().__init__(table.name, schema, source)
        self.sqla_table = table
