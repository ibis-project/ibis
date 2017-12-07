#!/usr/bin/env python

import os
import click
import tarfile

import numpy as np
import pandas as pd
import sqlalchemy as sa

from toolz import compose, dissoc

try:
    import sh
except ImportError:
    import pbs as sh


if os.environ.get('APPVEYOR', None) is not None:
    curl = sh.Command('C:\\Tools\\curl\\bin\\curl.exe')
else:
    curl = sh.curl


TEST_TABLES = ['functional_alltypes', 'diamonds', 'batting',
               'awards_players']


DEFAULT_MAP = dict(
    download=dict(
        data_url='https://storage.googleapis.com/ibis-ci-data'
    ),
    impala=dict(
        port=21050
    ),
    sqlite=dict(
        database='ibis_testing.db',
        schema='sqlite_schema.sql'
    ),
    postgres=dict(
        host='localhost',
        port=5432,
        user='postgres',
        password='ibis',
        database='ibis_testing',
        schema='postgresql_schema.sql'
    ),
    clickhouse=dict(
        host='localhost',
        port=9000,
        user='default',
        password='',
        database='ibis_testing',
        schema='clickhouse_schema.sql'
    )
)

DATA_DIRECTORY = os.environ.get('IBIS_TEST_DATA_DIRECTORY',
                                './ibis-testing-data')


options = compose(
    click.option('-h', '--host', required=False),
    click.option('-P', '--port', required=False, type=int),
    click.option('-u', '--user', required=False),
    click.option('-p', '--password', required=False),
    click.option('-D', '--database'),
    click.option('-S', '--schema', type=click.File('rt')),
    click.option('-t', '--tables', multiple=True, default=TEST_TABLES),
    click.option('-d', '--data-directory', default=DATA_DIRECTORY)
)


def recreate_database(driver, params, **kwargs):
    url = sa.engine.url.URL(driver, **dissoc(params, 'database'))
    engine = sa.create_engine(url, **kwargs)

    with engine.connect() as conn:
        conn.execute('DROP DATABASE IF EXISTS "{}"'.format(params['database']))
        conn.execute('CREATE DATABASE "{}"'.format(params['database']))


def init_database(driver, params, schema=None, recreate=True, **kwargs):
    params['username'] = params.pop('user', None)
    if recreate:
        recreate_database(driver, params, **kwargs)

    url = sa.engine.url.URL(driver, **params)
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
        df.to_sql(table, engine, index=False, if_exists='append')


@click.group(context_settings=dict(default_map=DEFAULT_MAP))
def cli():
    pass


@cli.command()
@click.argument('name', default='ibis-testing-data.tar.gz')
@click.option('--base-url')
@click.option('-d', '--data-directory', default='.')
def download(base_url, data_directory, name):
    if not os.path.exists(data_directory):
        os.mkdir(data_directory)

    data_url = '{}/{}'.format(base_url, name)
    path = os.path.join(data_directory, name)

    if not os.path.exists(path):
        curl(data_url, o=path, L=True,
             _out=click.get_binary_stream('stdout'),
             _err=click.get_binary_stream('stderr'))
    else:
        click.echo('Skipping download due to {} already exists.'.format(name))

    click.echo('Extracting archive...')
    if path.endswith(('.tar', '.gz', '.bz2', '.xz')):
        with tarfile.open(path, mode='r|gz') as f:
            f.extractall(path=data_directory)


@cli.command()
@options
def postgres(schema, tables, data_directory, **params):
    engine = init_database('postgresql', params, schema,
                           isolation_level='AUTOCOMMIT')
    insert_tables(engine, tables, data_directory)
    engine.execute('VACUUM FULL ANALYZE')


@cli.command()
@options
def sqlite(schema, tables, data_directory, **params):
    database = os.path.abspath(params['database'])
    if os.path.exists(database):
        try:
            os.remove(database)
        except OSError:
            pass

    engine = init_database('sqlite', params, schema, recreate=False)
    insert_tables(engine, tables, data_directory)

    engine.execute('VACUUM')
    engine.execute('VACUUM ANALYZE')


@cli.command()
@options
def clickhouse(schema, tables, data_directory, **params):
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
     - IBIS_IMPALA_PORT
     - IBIS_CLICKHOUSE_HOST
    """
    cli(auto_envvar_prefix='IBIS_TEST_')
