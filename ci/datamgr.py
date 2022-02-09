#!/usr/bin/env python
import logging
import os
import warnings
import zipfile
from pathlib import Path

import click
import pandas as pd
import sqlalchemy as sa
from toolz import dissoc

SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR_NAME = 'ibis-testing-data'
DATA_DIR = Path(
    os.environ.get('IBIS_TEST_DATA_DIRECTORY', SCRIPT_DIR / DATA_DIR_NAME)
)

TEST_TABLES = ['functional_alltypes', 'diamonds', 'batting', 'awards_players']


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


def recreate_database(driver, params, **kwargs):
    url = sa.engine.url.URL(driver, **dissoc(params, 'database'))
    engine = sa.create_engine(url, **kwargs)

    with engine.connect() as conn:
        conn.execute('DROP DATABASE IF EXISTS {}'.format(params['database']))
        conn.execute('CREATE DATABASE {}'.format(params['database']))


def init_database(driver, params, schema=None, recreate=True, **kwargs):
    new_params = params.copy()
    new_params['username'] = new_params.pop('user', None)

    if recreate:
        recreate_database(driver, new_params, **kwargs)

    url = sa.engine.url.URL(driver, **new_params)
    engine = sa.create_engine(url, **kwargs)

    if schema:
        with engine.connect() as conn:
            # clickhouse doesn't support multi-statements
            for stmt in schema.read().split(';'):
                if len(stmt.strip()):
                    conn.execute(stmt)

    return engine


def read_tables(names, data_directory):
    for name in names:
        path = data_directory / f'{name}.csv'

        params = {}

        if name == 'geo':
            params['quotechar'] = '"'

        df = pd.read_csv(str(path), index_col=None, header=0, **params)

        if name == 'functional_alltypes':
            df['bool_col'] = df['bool_col'].astype(bool)
            # string_col is actually dt.int64
            df['string_col'] = df['string_col'].astype(str)
            df['date_string_col'] = df['date_string_col'].astype(str)
            # timestamp_col has object dtype
            df['timestamp_col'] = pd.to_datetime(df['timestamp_col'])

        yield name, df


def convert_to_database_compatible_value(value):
    """Pandas 0.23 broke DataFrame.to_sql, so we workaround it by rolling our
    own extremely low-tech conversion routine
    """
    if pd.isnull(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    try:
        return value.item()
    except AttributeError:
        return value


def insert(engine, tablename, df):
    keys = df.columns
    rows = [
        dict(zip(keys, map(convert_to_database_compatible_value, row)))
        for row in df.itertuples(index=False, name=None)
    ]
    t = sa.Table(tablename, sa.MetaData(bind=engine), autoload=True)
    engine.execute(t.insert(), rows)


def insert_tables(engine, names, data_directory):
    for table, df in read_tables(names, data_directory):
        with engine.begin() as connection:
            insert(connection, table, df)


@click.group()
@click.option('--quiet/--verbose', '-q', default=False, is_flag=True)
def cli(quiet):
    if quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)


@cli.command()
@click.option(
    '--repo-url', '-r', default='https://github.com/ibis-project/testing-data'
)
@click.option('-d', '--directory', default=DATA_DIR)
def download(repo_url, directory):
    from shutil import rmtree

    from plumbum.cmd import curl

    directory = Path(directory)
    # download the master branch
    url = repo_url + '/archive/master.zip'
    # download the zip next to the target directory with the same name
    path = directory.with_suffix('.zip')

    if not path.exists():
        logger.info(f'Downloading {url} to {path}...')
        path.parent.mkdir(parents=True, exist_ok=True)
        download = curl[url, '-o', path, '-L']
        download(
            stdout=click.get_binary_stream('stdout'),
            stderr=click.get_binary_stream('stderr'),
        )
    else:
        logger.info(f'Skipping download: {path} already exists')

    logger.info(f'Extracting archive to {directory}')

    # extract all files
    extract_to = directory.with_name(directory.name + '_extracted')
    with zipfile.ZipFile(str(path), 'r') as f:
        f.extractall(str(extract_to))

    # remove existent folder
    if directory.exists():
        rmtree(str(directory))

    # rename to the target directory
    (extract_to / 'testing-data-master').rename(directory)

    # remove temporary extraction folder
    extract_to.rmdir()


@cli.command()
@click.option('-h', '--host', default='localhost')
@click.option(
    '-P',
    '--port',
    default=5432,
    envvar=["PGPORT", "IBIS_TEST_POSTGRES_PORT"],
    type=int,
)
@click.option('-u', '--user', default='postgres')
@click.option('-p', '--password', default='postgres')
@click.option('-D', '--database', default='ibis_testing')
@click.option(
    '-S',
    '--schema',
    type=click.File('rt'),
    default=SCRIPT_DIR / 'schema' / 'postgresql.sql',
    help='Path to SQL file that initializes the database via DDL.',
)
@click.option('-t', '--tables', multiple=True, default=TEST_TABLES + ['geo'])
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
@click.option(
    '--plpython/--no-plpython',
    help='Create PL/Python extension in database',
    default=True,
)
def postgres(schema, tables, data_directory, plpython, **params):
    logger.info('Initializing PostgreSQL...')
    engine = init_database(
        'postgresql', params, schema, isolation_level='AUTOCOMMIT'
    )

    engine.execute("CREATE SEQUENCE IF NOT EXISTS test_sequence;")

    use_postgis = 'geo' in tables
    if use_postgis:
        engine.execute("CREATE EXTENSION IF NOT EXISTS postgis")

    if plpython:
        engine.execute("CREATE EXTENSION IF NOT EXISTS plpython3u")

    for table in tables:
        src = data_directory / f'{table}.csv'

        # If we are loading the geo sample data, handle the data types
        # specifically so that PostGIS understands them as geometries.
        if table == 'geo':
            if not use_postgis:
                continue
            from geoalchemy2 import Geometry, WKTElement

            srid = 0
            df = pd.read_csv(src)
            df[df.columns[1:]] = df[df.columns[1:]].applymap(
                lambda x: WKTElement(x, srid=srid)
            )
            df.to_sql(
                'geo',
                engine,
                index=False,
                dtype={
                    "geo_point": Geometry("POINT", srid=srid),
                    "geo_linestring": Geometry("LINESTRING", srid=srid),
                    "geo_polygon": Geometry("POLYGON", srid=srid),
                    "geo_multipolygon": Geometry("MULTIPOLYGON", srid=srid),
                },
            )
        else:
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
    default=SCRIPT_DIR / 'ibis_testing.db',
    type=click.Path(path_type=Path),
)
@click.option(
    '-S',
    '--schema',
    type=click.File('rt'),
    default=SCRIPT_DIR / 'schema' / 'sqlite.sql',
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
def sqlite(database, schema, tables, data_directory, **params):
    logger.info('Initializing SQLite...')

    try:
        database.unlink()
    except OSError:
        pass

    params['database'] = str(database)
    engine = init_database('sqlite', params, schema, recreate=False)
    insert_tables(engine, tables, data_directory)


@cli.command()
@click.option('-h', '--host', default='localhost')
@click.option('-P', '--port', default=3306, type=int)
@click.option('-u', '--user', default='ibis')
@click.option('-p', '--password', default='ibis')
@click.option('-D', '--database', default='ibis_testing')
@click.option(
    '-S',
    '--schema',
    type=click.File('rt'),
    default=SCRIPT_DIR / 'schema' / 'mysql.sql',
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
def mysql(schema, tables, data_directory, **params):
    logger.info('Initializing MySQL...')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = init_database(
            'mysql+pymysql', params, schema, isolation_level='AUTOCOMMIT'
        )
    insert_tables(engine, tables, data_directory)


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
def clickhouse(schema, tables, data_directory, **params):
    logger.info('Initializing ClickHouse...')
    engine = init_database('clickhouse+native', params, schema)

    for table, df in read_tables(tables, data_directory):
        if table == 'batting':
            # float nan problem
            cols = df.select_dtypes([float]).columns
            df[cols] = df[cols].fillna(0).astype(int)
            # string None driver problem
            cols = df.select_dtypes([object]).columns
            df[cols] = df[cols].fillna('')
        elif table == 'awards_players':
            # string None driver problem
            cols = df.select_dtypes([object]).columns
            df[cols] = df[cols].fillna('')
        insert(engine, table, df)


@cli.command()
def pandas(**params):
    """
    The pandas backend does not need test data, but we still
    have an option for the backend for consistency, and to not
    have to avoid calling `./datamgr.py pandas` in the CI.
    """


@cli.command()
def dask(**params):
    """
    The dask backend does not need test data, but we still
    have an option for the backend for consistency, and to not
    have to avoid calling `./datamgr.py dask` in the CI.
    """


@cli.command()
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
@click.option('-i', '--ignore-missing-dependency', is_flag=True, default=True)
def datafusion(tables, data_directory, ignore_missing_dependency, **params):
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
    except ImportError:
        msg = 'PyArrow dependency is missing'
        if ignore_missing_dependency:
            logger.warning('Ignored: %s', msg)
            return 0
        else:
            raise click.ClickException(msg)

    for table, df in read_tables(tables, data_directory):
        arrow_table = pa.Table.from_pandas(df)
        target_path = data_directory / f'{table}.parquet'
        pq.write_table(arrow_table, str(target_path))


@cli.command()
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
def hdf5(tables, data_directory, **params):
    """
    The hdf5 backend does not need test data, but we still
    have an option for the backend for consistency, and to not
    have to avoid calling `./datamgr.py hdf5` in the CI.
    """
    for table, df in read_tables(tables, data_directory):
        df.to_hdf(
            data_directory / f'{table}.h5',
            key=table,
            mode='w',
            format='table',
        )


@cli.command()
def pyspark(**params):
    """
    The hdf5 backend does not need test data, but we still
    have an option for the backend for consistency, and to not
    have to avoid calling `./datamgr.py pyspark` in the CI.
    """


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
