#!/usr/bin/env python

from __future__ import annotations

import collections
import concurrent.futures
import itertools
import logging
import multiprocessing
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
CPU_COUNT = multiprocessing.cpu_count()

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


def impala_raise_if_cannot_write_to_hdfs(con, env):
    test_path = os.path.join(env.test_data_dir, ibis.util.guid())
    hdfs = con.hdfs
    with tempfile.NamedTemporaryFile(mode="wb+", delete=True) as f:
        f.write(ibis.util.guid().encode('UTF-8'))
        f.seek(0)
        hdfs.put_file(f.name, test_path)
    hdfs.rm(test_path)


def impala_upload_ibis_test_data_to_hdfs(con, data_path, env):
    hdfs = con.hdfs
    if hdfs.exists(env.test_data_dir):
        hdfs.rm(env.test_data_dir, recursive=True)
    hdfs.put(
        data_path,
        f"{os.path.dirname(env.test_data_dir)}/",
        recursive=True,
    )


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
        logger.info(os.path.basename(path))
        args = path, ALL_SCHEMAS.get(os.path.basename(path))
        kwargs = dict(
            name=os.path.basename(path),
            database=env.test_data_db,
            persist=True,
        )
        if executor is not None:
            yield executor.submit(method, *args, **kwargs)
        else:
            yield method(*args, **kwargs)


def impala_build_udfs():
    cwd = str(IBIS_HOME / 'ci' / 'udf')
    subprocess.run(["cmake", ".", "-G", "Ninja"], cwd=cwd)
    subprocess.run(["ninja"], cwd=cwd)


def impala_upload_udfs(con, env):
    build_dir = IBIS_HOME / 'ci' / 'udf' / 'build'
    bitcode_dir = os.path.join(env.test_data_dir, 'udf')
    if con.hdfs.exists(bitcode_dir):
        con.hdfs.rm(bitcode_dir, recursive=True)
    con.hdfs.put(str(build_dir), bitcode_dir, recursive=True)


def load_impala_data(con, data_dir, env):
    impala_upload_ibis_test_data_to_hdfs(con, data_dir, env)
    impala_create_test_database(con, env)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=CPU_COUNT
    ) as executor:
        table_futures = (
            executor.submit(table_future.result().compute_stats)
            for table_future in concurrent.futures.as_completed(
                impala_create_tables(con, env, executor=executor)
            )
        )
        for stats_future in concurrent.futures.as_completed(table_futures):
            stats_future.result()


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
@click.option('--quiet/--verbose', '-q/-v', default=False, is_flag=True)
def cli(quiet):
    if quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)


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


@cli.command()
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


@cli.command()
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
def sqlite(database, schema, tables, data_directory, **params):
    logger.info('Initializing SQLite...')

    init_database(
        url=sa.engine.make_url("sqlite://"),
        database=str(database.absolute()),
        schema=schema,
    )

    exe = "sqlite3.exe" if os.name == "nt" else "sqlite3"
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for table in tables:
            stem = f"{table}.csv"
            dst = Path(d).joinpath(stem)
            with dst.open("w") as f, data_directory.joinpath(stem).open(
                "r"
            ) as lines:
                # skip the first line
                f.write("".join(itertools.islice(lines, 1, None)))
                f.seek(0)
            paths.append((table, dst))
        subprocess.run(
            [exe, database],
            input="\n".join(
                [
                    ".separator ,",
                    *(
                        f".import {str(path.absolute())!r} {table}"
                        for table, path in paths
                    ),
                ]
            ).encode(),
        )


@cli.command()
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


@cli.command()
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


@cli.command()
@click.option('--data-dir', help='Path to testing data', default=DATA_DIR)
def impala(data_dir):
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
        pool_size=CPU_COUNT,
    )

    # validate our environment before performing possibly expensive operations
    impala_raise_if_cannot_write_to_hdfs(con, env)

    # load the data files
    load_impala_data(con, str(data_dir), env)

    # build and upload the UDFs
    impala_build_udfs()
    impala_upload_udfs(con, env)


@cli.command()
def pandas():
    """No-op to allow `python ./datamgr.py pandas`."""


@cli.command()
def dask():
    """No-op to allow `python ./datamgr.py dask`."""


@cli.command()
def datafusion():
    """No-op to allow `python ./datamgr.py datafusion`."""


@cli.command()
def pyspark():
    """No-op to allow `python ./datamgr.py pyspark`."""


if __name__ == '__main__':
    """
    Environment Variables are automatically parsed:
     - IBIS_TEST_{BACKEND}_PORT
     - IBIS_TEST_{BACKEND}_HOST
     - IBIS_TEST_{BACKEND}_USER
     - IBIS_TEST_{BACKEND}_PASSWORD
     - etc.
    """
    cli(auto_envvar_prefix='IBIS_TEST')
