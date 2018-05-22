#!/usr/bin/env python

import os
import six
import click
import tarfile

import pandas as pd
import sqlalchemy as sa

from toolz import dissoc
from plumbum import local
from plumbum.cmd import curl, psql
from ibis.compat import Path


SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = Path(os.environ.get('IBIS_TEST_DATA_DIRECTORY',
                               SCRIPT_DIR / 'ibis-testing-data'))

TEST_TABLES = ['functional_alltypes', 'diamonds', 'batting',
               'awards_players']


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
        path = data_directory / '{}.csv'.format(name)
        click.echo(path)
        df = pd.read_csv(str(path), index_col=None, header=0)

        if name == 'functional_alltypes':
            df['bool_col'] = df['bool_col'].astype(bool)
            # string_col is actually dt.int64
            df['string_col'] = df['string_col'].astype(six.text_type)
            df['date_string_col'] = df['date_string_col'].astype(six.text_type)
            # timestamp_col has object dtype
            df['timestamp_col'] = pd.to_datetime(df['timestamp_col'])

        yield (name, df)


def insert_tables(engine, names, data_directory):
    for table, df in read_tables(names, data_directory):
        with engine.begin() as connection:
            df.to_sql(
                table, connection, index=False, if_exists='append',
                chunksize=1 if os.name == 'nt' else None
                # Pandas 0.23 uses multi value inserts which is very slow for a
                # chunksize of 1. For some reason this only shows up on
                # Appveyor Windows CI
            )


@click.group()
def cli():
    pass


@cli.command()
@click.argument('name', default='ibis-testing-data.tar.gz')
@click.option('--base-url',
              default='https://storage.googleapis.com/ibis-testing-data')
@click.option('-d', '--directory', default=SCRIPT_DIR)
def download(base_url, directory, name):
    directory = Path(directory)
    if not directory.exists():
        directory.mkdir()

    data_url = '{}/{}'.format(base_url, name)
    path = directory / name

    if not path.exists():
        download = curl[data_url, '-o', path, '-L']
        download(stdout=click.get_binary_stream('stdout'),
                 stderr=click.get_binary_stream('stderr'))
    else:
        click.echo('Skipping download due to {} already exists.'.format(name))

    click.echo('Extracting archive to {} ...'.format(directory))
    if path.suffix in ('.tar', '.gz', '.bz2', '.xz'):
        with tarfile.open(str(path), mode='r|gz') as f:
            f.extractall(path=str(directory))


@cli.command()
@click.option('-t', '--tables', multiple=True, default=TEST_TABLES)
@click.option('-d', '--data-directory', default=DATA_DIR)
@click.option('-i', '--ignore-missing-dependency', is_flag=True, default=False)
def parquet(tables, data_directory, ignore_missing_dependency, **params):
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
    except ImportError:
        msg = 'PyArrow dependency is missing'
        if ignore_missing_dependency:
            click.echo('Ignored: {}'.format(msg))
            return 0
        else:
            raise click.ClickException(msg)

    data_directory = Path(data_directory)
    for table, df in read_tables(tables, data_directory):
        if table == 'functional_alltypes':
            schema = pa.schema([
                pa.field('string_col', pa.string()),
                pa.field('date_string_col', pa.string())
            ])
        else:
            schema = None
        arrow_table = pa.Table.from_pandas(df, schema=schema)
        target_path = data_directory / '{}.parquet'.format(table)

        pq.write_table(arrow_table, str(target_path))


@cli.command()
@click.option('-h', '--host', default='localhost')
@click.option('-P', '--port', default=5432, type=int)
@click.option('-u', '--user', default='postgres')
@click.option('-p', '--password', default='postgres')
@click.option('-D', '--database', default='ibis_testing')
@click.option('-S', '--schema', type=click.File('rt'),
              default=str(SCRIPT_DIR / 'schema' / 'postgresql.sql'))
@click.option('-t', '--tables', multiple=True, default=TEST_TABLES)
@click.option('-d', '--data-directory', default=DATA_DIR)
def postgres(schema, tables, data_directory, **params):
    data_directory = Path(data_directory)
    click.echo('Initializing PostgreSQL...')
    engine = init_database('postgresql', params, schema,
                           isolation_level='AUTOCOMMIT')

    query = "COPY {} FROM STDIN WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',')"
    database = params['database']
    for table in tables:
        src = data_directory / '{}.csv'.format(table)
        click.echo(src)
        load = psql['--host', params['host'], '--port', params['port'],
                    '--username', params['user'], '--dbname', database,
                    '--command', query.format(table)]
        with local.env(PGPASSWORD=params['password']):
            with src.open('r') as f:
                load(stdin=f)

    engine.execute('VACUUM FULL ANALYZE')


@cli.command()
@click.option('-D', '--database', default=SCRIPT_DIR / 'ibis_testing.db')
@click.option('-S', '--schema', type=click.File('rt'),
              default=str(SCRIPT_DIR / 'schema' / 'sqlite.sql'))
@click.option('-t', '--tables', multiple=True, default=TEST_TABLES)
@click.option('-d', '--data-directory', default=DATA_DIR)
def sqlite(database, schema, tables, data_directory, **params):
    database = Path(database)
    data_directory = Path(data_directory)
    click.echo('Initializing SQLite...')

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
@click.option('-S', '--schema', type=click.File('rt'),
              default=str(SCRIPT_DIR / 'schema' / 'mysql.sql'))
@click.option('-t', '--tables', multiple=True, default=TEST_TABLES)
@click.option('-d', '--data-directory', default=DATA_DIR)
def mysql(schema, tables, data_directory, **params):
    data_directory = Path(data_directory)
    click.echo('Initializing MySQL...')
    engine = init_database('mysql+pymysql', params, schema,
                           isolation_level='AUTOCOMMIT')
    insert_tables(engine, tables, data_directory)


@cli.command()
@click.option('-h', '--host', default='localhost')
@click.option('-P', '--port', default=9000, type=int)
@click.option('-u', '--user', default='default')
@click.option('-p', '--password', default='')
@click.option('-D', '--database', default='ibis_testing')
@click.option('-S', '--schema', type=click.File('rt'),
              default=str(SCRIPT_DIR / 'schema' / 'clickhouse.sql'))
@click.option('-t', '--tables', multiple=True, default=TEST_TABLES)
@click.option('-d', '--data-directory', default=DATA_DIR)
def clickhouse(schema, tables, data_directory, **params):
    data_directory = Path(data_directory)
    click.echo('Initializing ClickHouse...')
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

        df.to_sql(table, engine, index=False, if_exists='append')


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
