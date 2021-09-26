import sqlalchemy as sa

from ibis.backends.base.sql.alchemy import AlchemyClient

from . import udf


class SQLiteClient(AlchemyClient):
    """The Ibis SQLite client class."""

    def __init__(self, backend, path=None, create=False):
        self.backend = backend
        super().__init__(sa.create_engine("sqlite://"))
        # self.backend.name = path
        self.backend.database_name = "base"

        if path is not None:
            self.backend.attach(
                self.backend.database_name, path, create=create
            )

        udf.register_all(self.backend.con)
