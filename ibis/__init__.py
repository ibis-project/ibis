"""Initialize Ibis module."""
import warnings
from contextlib import suppress

import ibis.config
import ibis.expr.types as ir
from ibis import util

# pandas backend is mandatory
from ibis.backends import pandas  # noqa: F401
from ibis.common.exceptions import IbisError
from ibis.config import options
from ibis.expr import api
from ibis.expr.api import *  # noqa: F401,F403

from ._version import get_versions  # noqa: E402

__all__ = ['api', 'ir', 'util', 'IbisError', 'options']
__all__ += api.__all__


ibis.config.register_option(
    'interactive', False, validator=ibis.config.is_bool
)
ibis.config.register_option('verbose', False, validator=ibis.config.is_bool)
ibis.config.register_option('verbose_log', None)
ibis.config.register_option(
    'graphviz_repr',
    True,
    """\
Whether to render expressions as GraphViz PNGs when repr-ing in a Jupyter
notebook.
""",
    validator=ibis.config.is_bool,
)
ibis.config.register_option('default_backend', None)

with suppress(ImportError):
    # pip install ibis-framework[csv]
    from ibis.backends import csv  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[parquet]
    from ibis.backends import parquet  # noqa: F401

with suppress(ImportError):
    # pip install  ibis-framework[hdf5]
    from ibis.backends import hdf5  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[impala]
    from ibis.backends import impala  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[sqlite]
    from ibis.backends import sqlite  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[postgres]
    from ibis.backends import postgres  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[mysql]
    from ibis.backends import mysql  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[clickhouse]
    from ibis.backends import clickhouse  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[bigquery]
    from ibis.backends import bigquery  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[omniscidb]
    from ibis.backends import omniscidb  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[spark]
    from ibis.backends import spark  # noqa: F401

with suppress(ImportError):
    from ibis.backends import pyspark  # noqa: F401

with suppress(ImportError):
    # pip install ibis-framework[dask]
    import ibis.backends as dask  # noqa: F401

__version__ = get_versions()['version']
del get_versions


def __getattr__(name):
    if name in ('HDFS', 'WebHDFS', 'hdfs_connect'):
        warnings.warn(
            f'`ibis.{name}` has been deprecated and will be removed in a '
            f'future version, use `ibis.impala.{name}` instead',
            FutureWarning,
            stacklevel=2,
        )
        if 'impala' in globals():
            return getattr(impala, name)
        else:
            raise AttributeError(
                f'`ibis.{name}` requires impala backend to be installed'
            )
    raise AttributeError
