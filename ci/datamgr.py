#!/usr/bin/env python

from __future__ import annotations

import collections
import concurrent.futures
import itertools
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator

import click
import sqlalchemy as sa

if TYPE_CHECKING:
    import pyarrow as pa

import ibis

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

TEST_TABLES = {
    "functional_alltypes": ibis.schema(
        {
            "index": "int64",
            "Unnamed: 0": "int64",
            "id": "int32",
            "bool_col": "boolean",
            "tinyint_col": "int8",
            "smallint_col": "int16",
            "int_col": "int32",
            "bigint_col": "int64",
            "float_col": "float32",
            "double_col": "float64",
            "date_string_col": "string",
            "string_col": "string",
            "timestamp_col": "timestamp",
            "year": "int32",
            "month": "int32",
        }
    ),
    "diamonds": ibis.schema(
        {
            "carat": "float64",
            "cut": "string",
            "color": "string",
            "clarity": "string",
            "depth": "float64",
            "table": "float64",
            "price": "int64",
            "x": "float64",
            "y": "float64",
            "z": "float64",
        }
    ),
    "batting": ibis.schema(
        {
            "playerID": "string",
            "yearID": "int64",
            "stint": "int64",
            "teamID": "string",
            "lgID": "string",
            "G": "int64",
            "AB": "int64",
            "R": "int64",
            "H": "int64",
            "X2B": "int64",
            "X3B": "int64",
            "HR": "int64",
            "RBI": "int64",
            "SB": "int64",
            "CS": "int64",
            "BB": "int64",
            "SO": "int64",
            "IBB": "int64",
            "HBP": "int64",
            "SH": "int64",
            "SF": "int64",
            "GIDP": "int64",
        }
    ),
    "awards_players": ibis.schema(
        {
            "playerID": "string",
            "awardID": "string",
            "yearID": "int64",
            "lgID": "string",
            "tie": "string",
            "notes": "string",
        }
    ),
}


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


def impala_create_test_database(con, env):
    con.drop_database(env.test_data_db, force=True)
    con.create_database(env.test_data_db)
    con.create_table(
        'alltypes',
        schema=ibis.schema(
            [
                ('a', 'int8'),
                ('b', 'int16'),
                ('c', 'int32'),
                ('d', 'int64'),
                ('e', 'float'),
                ('f', 'double'),
                ('g', 'string'),
                ('h', 'boolean'),
                ('i', 'timestamp'),
            ]
        ),
        database=env.test_data_db,
    )


PARQUET_SCHEMAS = {
    'functional_alltypes': TEST_TABLES["functional_alltypes"].delete(
        ["index", "Unnamed: 0"]
    ),
    'tpch_region': ibis.schema(
        [
            ('r_regionkey', 'int16'),
            ('r_name', 'string'),
            ('r_comment', 'string'),
        ]
    ),
}


AVRO_SCHEMAS = {
    'tpch_region_avro': {
        'type': 'record',
        'name': 'a',
        'fields': [
            {'name': 'R_REGIONKEY', 'type': ['null', 'int']},
            {'name': 'R_NAME', 'type': ['null', 'string']},
            {'name': 'R_COMMENT', 'type': ['null', 'string']},
        ],
    }
}

ALL_SCHEMAS = collections.ChainMap(PARQUET_SCHEMAS, AVRO_SCHEMAS)


def impala_create_tables(con, env, *, executor=None):
    test_data_dir = env.test_data_dir
    avro_files = [
        (con.avro_file, os.path.join(test_data_dir, 'avro', path))
        for path in con.hdfs.ls(os.path.join(test_data_dir, 'avro'))
    ]
    parquet_files = [
        (con.parquet_file, os.path.join(test_data_dir, 'parquet', path))
        for path in con.hdfs.ls(os.path.join(test_data_dir, 'parquet'))
    ]
    for method, path in itertools.chain(parquet_files, avro_files):
        logger.debug(os.path.basename(path))
        yield executor.submit(
            method,
            path,
            ALL_SCHEMAS.get(os.path.basename(path)),
            name=os.path.basename(path),
            database=env.test_data_db,
            persist=True,
        )


def impala_build_and_upload_udfs(hdfs, env, *, fs):
    logger.info("Building UDFs...")

    cwd = str(IBIS_HOME / 'ci' / 'udf')
    subprocess.run(["cmake", ".", "-G", "Ninja"], cwd=cwd)
    subprocess.run(["ninja"], cwd=cwd)
    build_dir = IBIS_HOME / 'ci' / 'udf' / 'build'
    bitcode_dir = os.path.join(env.test_data_dir, 'udf')

    hdfs.mkdir(bitcode_dir, create_parents=True)

    for file in fs.find(build_dir):
        bitcode_path = os.path.join(
            bitcode_dir, os.path.relpath(file, build_dir)
        )
        logger.debug(f"{file} -> ${bitcode_path}")
        yield hdfs.put_file, file, bitcode_path


def recreate_database(
    url: sa.engine.url.URL,
    database: str,
    **kwargs: Any,
) -> None:
    engine = sa.create_engine(url, **kwargs)

    if url.database is not None:
        with engine.connect() as conn:
            conn.execute(f'DROP DATABASE IF EXISTS {database}')
            conn.execute(f'CREATE DATABASE {database}')


def init_database(
    url: sa.engine.url.URL,
    database: str,
    schema: str | None = None,
    recreate: bool = True,
    **kwargs: Any,
) -> sa.engine.Engine:
    if recreate:
        recreate_database(url, database, **kwargs)

    try:
        url.database = database
    except AttributeError:
        url = url.set(database=database)

    engine = sa.create_engine(url, **kwargs)

    if schema:
        with engine.connect() as conn:
            for stmt in filter(None, map(str.strip, schema.read().split(';'))):
                conn.execute(stmt)

    return engine


def read_tables(
    names: Iterable[str],
    data_directory: Path,
) -> Iterator[tuple[str, pa.Table]]:
    import pyarrow.csv as pac

    import ibis.expr.datatypes.pyarrow as pa_dt

    for name in names:
        schema = TEST_TABLES[name]
        convert_options = pac.ConvertOptions(
            column_types={
                name: pa_dt.to_pyarrow_type(type)
                for name, type in schema.items()
            }
        )
        yield name, pac.read_csv(
            data_directory / f'{name}.csv',
            convert_options=convert_options,
        )


@click.group()
@click.option("-v", "--verbose", count=True)
def cli(verbose):
    codes = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    logger.setLevel(codes.get(verbose, logging.DEBUG))


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
    logger.info('Initializing PostgreSQL...')
    engine = init_database(
        url=sa.engine.make_url(
            f"postgresql://{username}:{password}@{host}:{port}",
        ),
        database=database,
        schema=schema,
        isolation_level='AUTOCOMMIT',
    )

    for table in tables:
        src = data_directory / f'{table}.csv'
        # Here we insert rows using COPY table FROM STDIN, by way of
        # psycopg2's `copy_expert` API.
        #
        # We could use DataFrame.to_sql(method=callable), but that incurs
        # an unnecessary round trip and requires more code: the `data_iter`
        # argument would have to be turned back into a CSV before being
        # passed to `copy_expert`.
        sql = (
            f"COPY {table} FROM STDIN "
            "WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',')"
        )
        with src.open('r') as file:
            with engine.begin() as con, con.connection.cursor() as cur:
                cur.copy_expert(sql=sql, file=file)

    engine.execute('VACUUM FULL ANALYZE')


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
    logger.info('Initializing SQLite...')

    init_database(
        url=sa.engine.make_url("sqlite://"),
        database=str(database.absolute()),
        schema=schema,
    )

    with tempfile.TemporaryDirectory() as tempdir:
        for table in tables:
            basename = f"{table}.csv"
            with Path(tempdir).joinpath(basename).open("w") as f:
                with data_directory.joinpath(basename).open("r") as lines:
                    # skip the first line
                    f.write("".join(itertools.islice(lines, 1, None)))
        subprocess.run(
            ["sqlite3", database],
            input="\n".join(
                [
                    ".separator ,",
                    *(
                        f".import {str(path.absolute())!r} {path.stem}"
                        for path in Path(tempdir).glob("*.csv")
                    ),
                ]
            ).encode(),
        )


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
    logger.info('Initializing MySQL...')

    engine = init_database(
        url=sa.engine.make_url(
            f"mysql+pymysql://{username}:{password}@{host}:{port}?local_infile=1",  # noqa: E501
        ),
        database=database,
        schema=schema,
        isolation_level='AUTOCOMMIT',
    )
    with engine.connect() as con:
        for table in tables:
            con.execute(
                f"""\
LOAD DATA LOCAL INFILE '{data_directory / f"{table}.csv"}'
INTO TABLE {table}
COLUMNS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\\n'
IGNORE 1 LINES"""
            )


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
    import clickhouse_driver

    logger.info('Initializing ClickHouse...')
    database = params.pop("database")
    client = clickhouse_driver.Client(**params)

    client.execute(f"DROP DATABASE IF EXISTS {database}")
    client.execute(f"CREATE DATABASE {database}")
    client.execute(f"USE {database}")

    for stmt in filter(None, map(str.strip, schema.read().split(';'))):
        client.execute(stmt)

    for table, df in read_tables(tables, data_directory):
        query = f"INSERT INTO {table} VALUES"
        client.insert_dataframe(
            query,
            df.to_pandas(),
            settings={"use_numpy": True},
        )


def hdfs_make_dir_and_put_file(fs, src, target):
    logger.debug(f"{src} -> {target}")
    fs.mkdir(os.path.dirname(target), create_parents=True)
    fs.put_file(src, target)


@load.command()
@click.option('--data-dir', help='Path to testing data', default=DATA_DIR)
@click.pass_context
def impala(ctx, data_dir):
    """Load impala test data for Ibis."""
    import fsspec

    from ibis.backends.impala.tests.conftest import IbisTestEnv

    logger.info('Initializing Impala...')
    env = IbisTestEnv()
    con = ibis.impala.connect(
        host=env.impala_host,
        port=env.impala_port,
        hdfs_client=fsspec.filesystem(
            env.hdfs_protocol,
            host=env.nn_host,
            port=env.hdfs_port,
            user=env.hdfs_user,
        ),
        pool_size=URLLIB_DEFAULT_POOL_SIZE,
    )

    fs = fsspec.filesystem("file")

    data_files = {
        data_file
        for data_file in fs.find(data_dir)
        # ignore sqlite databases and markdown files
        if not data_file.endswith((".db", ".md"))
        # ignore files in the test data .git directory
        if (
            # ignore .git
            os.path.relpath(data_file, data_dir).split(os.sep, 1)[0]
            != ".git"
        )
    }

    executor = ctx.obj["executor"]

    hdfs = con.hdfs
    tasks = {
        # make the database
        executor.submit(impala_create_test_database, con, env),
        # build and upload UDFs
        *itertools.starmap(
            executor.submit,
            impala_build_and_upload_udfs(hdfs, env, fs=fs),
        ),
        # upload data files
        *(
            executor.submit(
                hdfs_make_dir_and_put_file,
                hdfs,
                data_file,
                os.path.join(
                    env.test_data_dir,
                    os.path.relpath(data_file, data_dir),
                ),
            )
            for data_file in data_files
        ),
    }

    for future in concurrent.futures.as_completed(tasks):
        future.result()

    # create the tables and compute stats
    for future in concurrent.futures.as_completed(
        executor.submit(table_future.result().compute_stats)
        for table_future in concurrent.futures.as_completed(
            impala_create_tables(con, env, executor=executor)
        )
    ):
        future.result()


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
def duckdb(schema, tables, data_directory, database, **params):
    import duckdb  # noqa: F401

    logger.info('Initializing DuckDB...')
    conn = duckdb.connect(f"ci/ibis-testing-data/{database}.ddb")
    for stmt in filter(None, map(str.strip, schema.read().split(';'))):
        conn.execute(stmt)

    for table in tables:
        src = data_directory / f'{table}.csv'
        sql = f"INSERT INTO {table} SELECT * FROM '{src}';"
        conn.execute(sql)


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
        max_workers=URLLIB_DEFAULT_POOL_SIZE
    ) as executor:
        cli(auto_envvar_prefix='IBIS_TEST', obj=dict(executor=executor))
