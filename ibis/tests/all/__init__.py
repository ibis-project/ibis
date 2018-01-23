from __future__ import absolute_import

from ibis.tests.all.config.impala import Impala
from ibis.tests.all.config.sqlite import SQLite
from ibis.tests.all.config.postgres import Postgres
from ibis.tests.all.config.clickhouse import Clickhouse
from ibis.tests.all.config.bigquery import BigQuery
from ibis.tests.all.config.pandas import Pandas
from ibis.tests.all.config.csv import CSV
from ibis.tests.all.config.hdf5 import HDF5
from ibis.tests.all.config.parquet import Parquet

backends = (
    Impala,
    SQLite,
    Postgres,
    Clickhouse,
    BigQuery,
    Pandas,
    CSV,
    HDF5,
    Parquet,
)
