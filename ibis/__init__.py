"""Initialize Ibis module."""
from contextlib import suppress

import ibis.config_init  # noqa: F401
import ibis.expr.api as api  # noqa: F401
import ibis.expr.types as ir  # noqa: F401

# pandas backend is mandatory
import ibis.pandas.api as pandas  # noqa: F401
import ibis.util as util  # noqa: F401
from ibis.common.exceptions import IbisError  # noqa: F401
from ibis.config import options  # noqa: F401
from ibis.expr.api import *  # noqa: F401,F403
from ibis.filesystems import HDFS, WebHDFS, hdfs_connect  # noqa: F401

from ._version import get_versions  # noqa: E402

with suppress(ImportError):
    # pip install ibis-framework[csv]
    import ibis.file.csv as csv  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[parquet]
    import ibis.file.parquet as parquet  # noqa: F401

with suppress(ImportError):
    # pip install  ibis-framework[hdf5]
    import ibis.file.hdf5 as hdf5  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[impala]
    import ibis.impala.api as impala  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[sqlite]
    import ibis.sql.sqlite.api as sqlite  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[postgres]
    import ibis.sql.postgres.api as postgres  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[mysql]
    import ibis.sql.mysql.api as mysql  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[clickhouse]
    import ibis.backends.clickhouse.api as clickhouse  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[bigquery]
    import ibis.bigquery.api as bigquery  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[omniscidb]
    from ibis.backends import omniscidb  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[spark]
    import ibis.spark.api as spark  # noqa: F401

with suppress(ImportError):
    import ibis.pyspark.api as pyspark  # noqa: F401


__version__ = get_versions()['version']
del get_versions
