#!/usr/bin/env python

import os
import tarfile

import click

import numpy as np
import pandas as pd
import sqlalchemy as sa

from toolz import dissoc

try:
    import sh
except ImportError:
    import pbs as sh


if os.environ.get('APPVEYOR', None) is not None:
    curl = sh.Command(r'C:\Tools\curl\bin\curl.exe')
    psql = sh.Command(r'C:\Program Files\PostgreSQL\10\bin\psql.exe')
else:
    curl = sh.curl
    psql = sh.psql

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_TABLES = ['functional_alltypes', 'diamonds', 'batting',
               'awards_players']
TEST_DATA = os.environ.get('IBIS_TEST_DATA_DIRECTORY',
                           os.path.join(SCRIPT_DIR, 'ibis-testing-data'))


def recreate_database(driver, params, **kwargs):
    url = sa.engine.url.URL(driver, **dissoc(params, 'database'))
    engine = sa.create_engine(url, **kwargs)

    with engine.connect() as conn:
        conn.execute('DROP DATABASE IF EXISTS "{}"'.format(params['database']))
        conn.execute('CREATE DATABASE "{}"'.format(params['database']))


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
    dtype = {'bool_col': np.bool_}
    for name in names:
        path = os.path.join(data_directory, '{}.csv'.format(name))
        click.echo(path)
        df = pd.read_csv(path, index_col=None, header=0, dtype=dtype)
        yield (name, df)


def insert_tables(engine, names, data_directory):
    for table, df in read_tables(names, data_directory):
        with engine.begin() as connection:
            df.to_sql(table, connection, index=False, if_exists='append')


@click.group()
def cli():
    pass


@cli.command()
@click.argument('name', default='ibis-testing-data.tar.gz')
@click.option('--base-url',
              default='https://storage.googleapis.com/ibis-ci-data')
@click.option('-d', '--directory', default=SCRIPT_DIR)
def download(base_url, directory, name):
    if not os.path.exists(directory):
        os.mkdir(directory)

    data_url = '{}/{}'.format(base_url, name)
    path = os.path.join(directory, name)

    if not os.path.exists(path):
        curl(data_url, o=path, L=True,
             _out=click.get_binary_stream('stdout'),
             _err=click.get_binary_stream('stderr'))
    else:
        click.echo('Skipping download due to {} already exists.'.format(name))

    click.echo('Extracting archive to {} ...'.format(directory))
    if path.endswith(('.tar', '.gz', '.bz2', '.xz')):
        with tarfile.open(path, mode='r|gz') as f:
            f.extractall(path=directory)


@cli.command()
@click.option('-h', '--host', default='localhost')
@click.option('-P', '--port', default=5432, type=int)
@click.option('-u', '--user', default='postgres')
@click.option('-p', '--password', default='ibis')
@click.option('-D', '--database', default='ibis_testing')
@click.option('-S', '--schema', type=click.File('rt'),
              default=os.path.join(SCRIPT_DIR, 'schema/postgresql.sql'))
@click.option('-t', '--tables', multiple=True, default=TEST_TABLES)
@click.option('-d', '--data-directory', default=TEST_DATA)
def postgres(schema, tables, data_directory, **params):
    click.echo('Initializing PostgreSQL...')
    engine = init_database('postgresql', params, schema,
                           isolation_level='AUTOCOMMIT')

    query = "COPY {} FROM STDIN WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',')"
    database = params['database']
    for table in tables:
        src = os.path.abspath(os.path.join(data_directory, table + '.csv'))
        click.echo(src)
        with open(src, 'rb') as f:
            psql(
                host=params['host'],
                port=params['port'],
                user=params['user'],
                dbname=database,
                command=query.format(table),
                _in=f
            )
    engine.execute('VACUUM FULL ANALYZE')


@cli.command()
@click.option('-D', '--database',
              default=os.path.join(SCRIPT_DIR, 'ibis_testing.db'))
@click.option('-S', '--schema', type=click.File('rt'),
              default=os.path.join(SCRIPT_DIR, 'schema/sqlite.sql'))
@click.option('-t', '--tables', multiple=True, default=TEST_TABLES)
@click.option('-d', '--data-directory', default=TEST_DATA)
def sqlite(database, schema, tables, data_directory, **params):
    click.echo('Initializing SQLite...')
    if os.path.exists(database):
        try:
            os.remove(database)
        except OSError:
            pass

    params['database'] = database
    engine = init_database('sqlite', params, schema, recreate=False)
    insert_tables(engine, tables, data_directory)

    # engine.execute('VACUUM')
    # engine.execute('VACUUM ANALYZE')


@cli.command()
@click.option('-h', '--host', default='localhost')
@click.option('-P', '--port', default=9000, type=int)
@click.option('-u', '--user', default='default')
@click.option('-p', '--password', default='')
@click.option('-D', '--database', default='ibis_testing')
@click.option('-S', '--schema', type=click.File('rt'),
              default=os.path.join(SCRIPT_DIR, 'schema/clickhouse.sql'))
@click.option('-t', '--tables', multiple=True, default=TEST_TABLES)
@click.option('-d', '--data-directory', default=TEST_DATA)
def clickhouse(schema, tables, data_directory, **params):
    click.echo('Initializing ClickHouse...')
    engine = init_database('clickhouse+native', params, schema)

    for table, df in read_tables(tables, data_directory):
        if table == 'functional_alltypes':
            # string_col is actually dt.int64
            df['string_col'] = df['string_col'].astype(str)
            # timestamp_col has object dtype
            df['timestamp_col'] = df['timestamp_col'].astype('datetime64[s]')
        elif table == 'batting':
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
