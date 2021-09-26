import getpass
from typing import Optional

import psycopg2  # NOQA fail early if the driver is missing
import sqlalchemy as sa

from ibis.backends.base.sql.alchemy import AlchemyClient


class PostgreSQLClient(AlchemyClient):

    """The Ibis PostgreSQL client class

    Attributes
    ----------
    con : sqlalchemy.engine.Engine
    """

    def __init__(
        self,
        backend,
        host: str = 'localhost',
        user: str = getpass.getuser(),
        password: Optional[str] = None,
        port: int = 5432,
        database: str = 'public',
        url: Optional[str] = None,
        driver: str = 'psycopg2',
    ):
        self.backend = backend
        if url is None:
            if driver != 'psycopg2':
                raise NotImplementedError(
                    'psycopg2 is currently the only supported driver'
                )
            sa_url = sa.engine.url.URL(
                'postgresql+psycopg2',
                host=host,
                port=port,
                username=user,
                password=password,
                database=database,
            )
        else:
            sa_url = sa.engine.url.make_url(url)

        super().__init__(sa.create_engine(sa_url))
        self.backend.database_name = sa_url.database
