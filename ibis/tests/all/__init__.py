from __future__ import absolute_import

from ibis.tests.all.impala import Impala
from ibis.tests.all.sqlite import SQLite
from ibis.tests.all.postgres import Postgres
from ibis.tests.all.clickhouse import Clickhouse
from ibis.tests.all.bigquery import BigQuery
from ibis.tests.all.pandas import Pandas
from ibis.tests.all.csv import CSV
from ibis.tests.all.hdf5 import HDF5
from ibis.tests.all.parquet import Parquet

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
