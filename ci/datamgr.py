#!/usr/bin/env python

from __future__ import annotations

import concurrent.futures
import logging
import os
import shutil
import subprocess
from pathlib import Path

import click

from ibis.backends.conftest import TEST_TABLES, read_tables

IBIS_HOME = Path(__file__).parent.parent.absolute()
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = Path(
    os.environ.get(
        'IBIS_TEST_DATA_DIRECTORY',
        SCRIPT_DIR / 'ibis-testing-data',
    )
)
# without setting the pool size, connections are dropped from the urllib3
# connection pool when the number of workers exceeds this value. this doesn't
# appear to be configurable through fsspec
URLLIB_DEFAULT_POOL_SIZE = 10


def get_logger(name, level=None, format=None, propagate=False):
    logging.basicConfig()
    handler = logging.StreamHandler()

    if format is None:
        format = (
            '%(relativeCreated)6d '
            '%(name)-20s '
            '%(levelname)-8s '
            '%(threadName)-25s '
            '%(message)s'
        )
    handler.setFormatter(logging.Formatter(fmt=format))
    logger = logging.getLogger(name)
    logger.propagate = propagate
    logger.setLevel(
        level
        or getattr(logging, os.environ.get('LOGLEVEL', 'WARNING').upper())
    )
    logger.addHandler(handler)
    return logger


logger = get_logger(Path(__file__).with_suffix('').name)


@click.group()
@click.option("-v", "--verbose", count=True)
def cli(verbose):
    codes = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    logger.setLevel(codes.get(verbose, logging.DEBUG))


@cli.command()
@click.pass_context
def generate_parquet(ctx):
    import pyarrow.parquet as pq

    executor = ctx.obj["executor"]

    for name, table in read_tables(
        (name for name in TEST_TABLES if name != "functional_alltypes"),
        DATA_DIR,
    ):
        logger.info(f"Creating {name}.parquet from CSV")
        dirname = DATA_DIR / "parquet" / name
        dirname.mkdir(parents=True, exist_ok=True)
        executor.submit(pq.write_table, table, dirname / f"{name}.parquet")


@cli.group()
def load():
    pass


@cli.command()
@click.option(
    '--repo-url',
    '-r',
    default='https://github.com/ibis-project/testing-data',
    help="Data repository URL",
    show_default=True,
)
@click.option(
    "--rev",
    "-R",
    type=str,
    default=None,
    help="Git revision. Defaults to master",
)
@click.option(
    '-d',
    '--directory',
    type=click.Path(file_okay=False, dir_okay=True, path_type=str),
    default=DATA_DIR,
    help="Output directory",
    show_default=True,
)
def download(repo_url, rev, directory):
    shutil.rmtree(directory, ignore_errors=True)
    subprocess.run(
        ["git", "clone", repo_url, directory]
        + (["--depth", "1"] if rev is None else [])
    )
    if rev is not None:
        subprocess.run(["git", "reset", "--hard", rev], cwd=directory)


@load.command()
@click.option('-h', '--host', default='localhost')
@click.option(
    '-P',
    '--port',
    default=5432,
    envvar=["PGPORT", "IBIS_TEST_POSTGRES_PORT"],
    type=int,
)
@click.option('-u', '--username', default='postgres')
@click.option('-p', '--password', default='postgres')
@click.option('-D', '--database', default='ibis_testing')
@click.option(
    '-S',
    '--schema',
    type=click.File('rt'),
    default=SCRIPT_DIR / 'schema' / 'postgresql.sql',
    help='Path to SQL file that initializes the database via DDL.',
)
@click.option(
    '-t',
    '--tables',
    multiple=True,
    default=list(TEST_TABLES) + ['geo'],
)
@click.option(
    '-d',
    '--data-directory',
    default=DATA_DIR,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        path_type=Path,
    ),
)
def postgres(
    host,
    port,
    username,
    password,
    database,
    schema,
    tables,
    data_directory,
):
    from ibis.backends.postgres.tests.conftest import (
        TestConf as PostgresTestConf,
    )

    logger.info('Initializing PostgreSQL...')
    kwargs = {
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "database": database,
        "schema": schema,
        "tables": tables,
    }
    PostgresTestConf.load_data(data_directory, SCRIPT_DIR, **kwargs)


@load.command()
@click.option(
    '-D',
    '--database',
    default=DATA_DIR / 'ibis_testing.db',
    type=click.Path(path_type=Path),
)
@click.option(
    '-S',
    '--schema',
    type=click.File('rt'),
    default=SCRIPT_DIR / 'schema' / 'sqlite.sql',
    help='Path to SQL file that initializes the database via DDL.',
)
@click.option(
    '-t',
    '--tables',
    multiple=True,
    default=list(TEST_TABLES.keys()),
)
@click.option(
    '-d',
    '--data-directory',
    default=DATA_DIR,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        path_type=Path,
    ),
)
def sqlite(database, schema, tables, data_directory):
    from ibis.backends.sqlite.tests.conftest import TestConf as SqliteTestConf

    logger.info('Initializing SQLite...')
    kwargs = {
        "database": database,
        "schema": schema,
        "tables": tables,
    }
    SqliteTestConf.load_data(data_directory, SCRIPT_DIR, **kwargs)


@load.command()
@click.option('-h', '--host', default='localhost')
@click.option('-P', '--port', default=3306, type=int)
@click.option('-u', '--username', default='ibis')
@click.option('-p', '--password', default='ibis')
@click.option('-D', '--database', default='ibis_testing')
@click.option(
    '-S',
    '--schema',
    type=click.File('rt'),
    default=SCRIPT_DIR / 'schema' / 'mysql.sql',
    help='Path to SQL file that initializes the database via DDL.',
)
@click.option(
    '-t',
    '--tables',
    multiple=True,
    default=list(TEST_TABLES.keys()),
)
@click.option(
    '-d',
    '--data-directory',
    default=DATA_DIR,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        path_type=Path,
    ),
)
def mysql(
    host,
    port,
    username,
    password,
    database,
    schema,
    tables,
    data_directory,
):
    from ibis.backends.mysql.tests.conftest import TestConf as MySQLTestConf

    logger.info('Initializing MySQL...')
    kwargs = {
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "database": database,
        "schema": schema,
        "tables": tables,
    }

    MySQLTestConf.load_data(data_directory, SCRIPT_DIR, **kwargs)


@load.command()
@click.option('-h', '--host', default='localhost')
@click.option('-P', '--port', default=9000, type=int)
@click.option('-u', '--user', default='default')
@click.option('-p', '--password', default='')
@click.option('-D', '--database', default='ibis_testing')
@click.option(
    '-S',
    '--schema',
    type=click.File('rt'),
    default=SCRIPT_DIR / 'schema' / 'clickhouse.sql',
    help='Path to SQL file that initializes the database via DDL.',
)
@click.option(
    '-t',
    '--tables',
    multiple=True,
    default=list(TEST_TABLES.keys()),
)
@click.option(
    '-d',
    '--data-directory',
    default=DATA_DIR,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        path_type=Path,
    ),
)
def clickhouse(schema, tables, data_directory, **params):
    from ibis.backends.clickhouse.tests.conftest import (
        TestConf as ClickhouseTestConf,
    )

    logger.info('Initializing ClickHouse...')
    params["schema"] = schema
    params["tables"] = tables
    ClickhouseTestConf.load_data(data_directory, SCRIPT_DIR, **params)


@load.command()
@click.option('--data-dir', help='Path to testing data', default=DATA_DIR)
@click.pass_context
def impala(ctx, data_dir):
    """Load impala test data for Ibis."""
    from ibis.backends.impala.tests.conftest import TestConf as ImpalaTestConf

    logger.info('Initializing Impala...')
    ImpalaTestConf.load_data(data_dir, SCRIPT_DIR)


@load.command()
def pandas():
    """No-op to allow `python ci/datamgr.py load pandas`."""


@load.command()
def dask():
    """No-op to allow `python ci/datamgr.py load dask`."""


@load.command()
def datafusion():
    """No-op to allow `python ci/datamgr.py load datafusion`."""


@load.command()
def pyspark():
    """No-op to allow `python ci/datamgr.py load pyspark`."""


@load.command()
@click.pass_context
def all(ctx):
    info_name = ctx.info_name
    executor = ctx.obj["executor"]

    for future in concurrent.futures.as_completed(
        executor.submit(ctx.forward, command)
        for name, command in ctx.parent.command.commands.items()
        if name != info_name
    ):
        future.result()


@load.command()
@click.option('-D', '--database', default='ibis_testing')
@click.option(
    '-S',
    '--schema',
    type=click.File('rt'),
    default=SCRIPT_DIR / 'schema' / 'duckdb.sql',
    help='Path to SQL file that initializes the database via DDL.',
)
@click.option('-t', '--tables', multiple=True, default=TEST_TABLES)
@click.option(
    '-d',
    '--data-directory',
    default=DATA_DIR,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        path_type=Path,
    ),
)
def duckdb(schema, tables, data_directory, database, **_):
    from ibis.backends.duckdb.tests.conftest import TestConf as DuckDBTestConf

    logger.info('Initializing DuckDB...')
    kwargs = {
        "schema": schema,
        "tables": tables,
        "database": database,
    }
    DuckDBTestConf.load_data(data_directory, SCRIPT_DIR, **kwargs)


if __name__ == '__main__':
    """
    Environment Variables are automatically parsed:
     - IBIS_TEST_{BACKEND}_PORT
     - IBIS_TEST_{BACKEND}_HOST
     - IBIS_TEST_{BACKEND}_USER
     - IBIS_TEST_{BACKEND}_PASSWORD
     - etc.
    """
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(
            os.environ.get(
                "IBIS_DATA_MAX_WORKERS",
                URLLIB_DEFAULT_POOL_SIZE,
            )
        )
    ) as executor:
        cli(auto_envvar_prefix='IBIS_TEST', obj=dict(executor=executor))
