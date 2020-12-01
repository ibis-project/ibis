import ibis
import ibis.backends.base

from .client import OmniSciDBClient
from .compiler import OmniSciDBQueryBuilder, OmniSciDBDialect


class Backend(ibis.backends.base.BaseBackend):
    name = 'omniscidb'
    builder = OmniSciDBQueryBuilder
    dialect = OmniSciDBDialect

    def connect(self):
        ibis.options.sql.default_limit = None

        client = OmniSciDBClient(
            uri=self.connection_string,
            session_id=session_id,
            ipc=ipc,
            gpu_device=gpu_device,
        )

        return client
