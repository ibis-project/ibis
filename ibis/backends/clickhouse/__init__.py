import ibis.config
import ibis.backends.base

from .client import ClickhouseClient
from .compiler import ClickhouseQueryBuilder, ClickhouseDialect

try:
    import lz4  # noqa: F401

    _default_compression = 'lz4'
except ImportError:
    _default_compression = False


class Backend(ibis.backends.base.BaseBackend):
    name = 'clickhouse'
    builder = ClickhouseQueryBuilder
    dialect = ClickhouseDialect

    def connect(self):
        return ClickhouseClient(
            host,
            port=port,
            database=database,
            user=user,
            password=password,
            client_name=client_name,
            compression=compression,
        )

    def register_options(self):
        ibis.config.register_option(
            'temp_db',
            '__ibis_tmp',
            'Database to use for temporary tables, views. functions, etc.',
        )
