import ibis.expr.operations as ops
import ibis.expr.schema as sch
from ibis.backends.base import Database


class AlchemyDatabaseSchema(Database):
    def __init__(self, name, database):
        """

        Parameters
        ----------
        name : str
        database : AlchemyDatabase
        """
        self.name = name
        self.database = database
        self.client = database.client

    def __repr__(self):
        return "Schema({!r})".format(self.name)

    def drop(self, force=False):
        """
        Drop the schema

        Parameters
        ----------
        force : boolean, default False
          Drop any objects if they exist, and do not fail if the schema does
          not exist.
        """
        raise NotImplementedError(
            "Drop is not implemented yet for sqlalchemy schemas"
        )

    def table(self, name):
        """
        Return a table expression referencing a table in this schema

        Returns
        -------
        table : TableExpr
        """
        qualified_name = self._qualify(name)
        return self.database.table(qualified_name, self.name)

    def list_tables(self, like=None):
        return self.database.list_tables(like, database=self.name)


class AlchemyDatabase(Database):
    """

    Attributes
    ----------
    client : AlchemyClient

    """

    schema_class = AlchemyDatabaseSchema

    def table(self, name, schema=None):
        return self.client.table(name, schema=schema)

    def schema(self, name):
        return self.schema_class(name, self)


class AlchemyTable(ops.DatabaseTable):
    def __init__(self, table, source, schema=None):
        schema = sch.infer(table, schema=schema)
        super().__init__(table.name, schema, source)
        self.sqla_table = table
