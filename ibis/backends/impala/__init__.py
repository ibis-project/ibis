import ibis.config
import ibis.backends.base

# these objects are exposed in the public API and are not used in the module
from .client import (  # noqa: F401
    ImpalaClient,
    ImpalaConnection,
    ImpalaDatabase,
    ImpalaTable,
)
from .compiler import ImpalaDialect, ImpalaQueryBuilder
from .hdfs import HDFS, WebHDFS, hdfs_connect  # noqa: F401
from .udf import *  # noqa: F401,F403


class Backend(ibis.backends.base.BaseBackend):
    name = 'impala'
    builder = ImpalaQueryBuilder
    dialect = ImpalaDialect

    def connect(self):
        params = {
            'host': host,
            'port': port,
            'database': database,
            'timeout': timeout,
            'use_ssl': use_ssl,
            'ca_cert': ca_cert,
            'user': user,
            'password': password,
            'auth_mechanism': auth_mechanism,
            'kerberos_service_name': kerberos_service_name,
        }

        con = ImpalaConnection(pool_size=pool_size, **params)
        try:
            client = ImpalaClient(con, hdfs_client=hdfs_client)
        except Exception:
            con.close()
            raise

        return client

    def register_options(self):
        ibis.config.register_option(
            'temp_db',
            '__ibis_tmp',
            'Database to use for temporary tables, views. functions, etc.',
        )
        ibis.config.register_option(
            'temp_hdfs_path',
            '/tmp/ibis',
            'HDFS path for storage of temporary data',
        )
