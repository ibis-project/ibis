#!/usr/bin/env python

import os
import getpass
import tempfile
import tarfile
import operator

import sqlalchemy as sa

import numpy as np
import pandas as pd

import click

try:
    import sh
except ImportError:
    import pbs as sh


@click.group()
def cli():
    pass


@cli.command()
@click.argument('tables', nargs=-1)
@click.option('-S', '--script', type=click.File('rt'), required=True)
@click.option(
    '-d', '--database',
    default=os.environ.get('IBIS_TEST_CLICKHOUSE_DB', 'ibis_testing')
)
@click.option(
    '-D', '--data-directory',
    default=tempfile.gettempdir(), type=click.Path(exists=True)
)
def clickhouse(script, tables, database, data_directory):
    username = os.environ.get('IBIS_CLICKHOUSE_USER', 'default')
    host = os.environ.get('IBIS_CLICKHOUSE_HOST', 'localhost')
    password = os.environ.get('IBIS_CLICKHOUSE_PASS', '')

    url = sa.engine.url.URL(
        'clickhouse+native',
        username=username,
        host=host,
        password=password,
    )
    engine = sa.create_engine(str(url))
    engine.execute('DROP DATABASE IF EXISTS "{}"'.format(database))
    engine.execute('CREATE DATABASE "{}"'.format(database))

    url = sa.engine.url.URL(
        'clickhouse+native',
        username=username,
        host=host,
        password=password,
        database=database,
    )
    engine = sa.create_engine(str(url))
    script_text = script.read()

    # missing stmt
    # INSERT INTO array_types (x, y, z, grouper, scalar_column) VALUES
    # ([1, 2, 3], ['a', 'b', 'c'], [1.0, 2.0, 3.0], 'a', 1.0),
    # ([4, 5], ['d', 'e'], [4.0, 5.0], 'a', 2.0),
    # ([6], ['f'], [6.0], 'a', 3.0),
    # ([1], ['a'], [], 'b', 4.0),
    # ([2, 3], ['b', 'c'], [], 'b', 5.0),
    # ([4, 5], ['d', 'e'], [4.0, 5.0], 'c', 6.0);

    with engine.begin() as con:
        # doesn't support multiple statements
        for stmt in script_text.split(';'):
            if len(stmt.strip()):
                con.execute(stmt)

    table_paths = [
        os.path.join(data_directory, '{}.csv'.format(table))
        for table in tables
    ]
    dtype = {'bool_col': np.bool_}
    for table, path in zip(tables, table_paths):
        # correct dtypes per table to be able to insert
        # TODO: cleanup, kinda ugly
        df = pd.read_csv(path, index_col=None, header=0, dtype=dtype)
        if table == 'functional_alltypes':
            df = df.rename(columns={'Unnamed: 0': 'Unnamed_0'})
            cols = ['date_string_col', 'string_col']
            df[cols] = df[cols].astype(str)
            df.timestamp_col = df.timestamp_col.astype('datetime64[s]')
        elif table == 'batting':
            cols = ['playerID', 'teamID', 'lgID']
            df[cols] = df[cols].astype(str)
            cols = df.select_dtypes([float]).columns
            df[cols] = df[cols].fillna(0).astype(int)
        elif table == 'awards_players':
            cols = ['playerID', 'awardID', 'lgID', 'tie', 'notes']
            df[cols] = df[cols].astype(str)

        df.to_sql(table, engine, index=False, if_exists='append')


@cli.command()
@click.argument('tables', nargs=-1)
@click.option('-S', '--script', type=click.File('rt'), required=True)
@click.option(
    '-d', '--database',
    default=os.environ.get(
        'IBIS_TEST_POSTGRES_DB', os.environ.get('PGDATABASE', 'ibis_testing')
    ),
)
@click.option(
    '-D', '--data-directory',
    default=tempfile.gettempdir(), type=click.Path(exists=True)
)
def postgres(script, tables, database, data_directory):
    username = os.environ.get(
        'IBIS_POSTGRES_USER', os.environ.get('PGUSER', getpass.getuser())
    )
    host = os.environ.get('PGHOST', 'localhost')
    password = os.environ.get('IBIS_POSTGRES_PASS', os.environ.get('PGPASS'))
    url = sa.engine.url.URL(
        'postgresql',
        username=username,
        host=host,
        password=password,
    )
    engine = sa.create_engine(str(url), isolation_level='AUTOCOMMIT')
    engine.execute('DROP DATABASE IF EXISTS "{}"'.format(database))
    engine.execute('CREATE DATABASE "{}"'.format(database))

    url = sa.engine.url.URL(
        'postgresql',
        username=username,
        host=host,
        password=password,
        database=database,
    )
    engine = sa.create_engine(str(url))
    script_text = script.read()
    with engine.begin() as con:
        con.execute(script_text)

    table_paths = [
        os.path.join(data_directory, '{}.csv'.format(table))
        for table in tables
    ]
    dtype = {'bool_col': np.bool_}
    for table, path in zip(tables, table_paths):
        df = pd.read_csv(path, index_col=None, header=0, dtype=dtype)
        df.to_sql(table, engine, index=False, if_exists='append')
    engine = sa.create_engine(str(url), isolation_level='AUTOCOMMIT')
    engine.execute('VACUUM FULL ANALYZE')


@cli.command()
@click.argument('tables', nargs=-1)
@click.option('-S', '--script', type=click.File('rt'), required=True)
@click.option(
    '-d', '--database',
    default=os.environ.get('IBIS_TEST_SQLITE_DB_PATH', 'ibis_testing.db')
)
@click.option(
    '-D', '--data-directory',
    default=tempfile.gettempdir(), type=click.Path(exists=True)
)
def sqlite(script, tables, database, data_directory):
    database = os.path.abspath(database)
    if os.path.exists(database):
        try:
            os.remove(database)
        except OSError:
            pass
    engine = sa.create_engine('sqlite:///{}'.format(database))
    script_text = script.read()
    with engine.begin() as con:
        con.connection.connection.executescript(script_text)
    table_paths = [
        os.path.join(data_directory, '{}.csv'.format(table))
        for table in tables
    ]
    click.echo(tables)
    click.echo(table_paths)
    for table, path in zip(tables, table_paths):
        df = pd.read_csv(path, index_col=None, header=0)
        with engine.begin() as con:
            df.to_sql(table, con, index=False, if_exists='append')
    engine.execute('VACUUM')
    engine.execute('VACUUM ANALYZE')


if os.environ.get('APPVEYOR', None) is not None:
    curl = sh.Command('C:\\Tools\\curl\\bin\\curl.exe')
else:
    curl = sh.curl


@cli.command()
@click.argument(
    'base_url',
    required=False,
    default='https://storage.googleapis.com/ibis-ci-data'  # noqa: E501
)
@click.option('-d', '--data', multiple=True)
@click.option('-D', '--directory', default='.', type=click.Path(exists=False))
def download(base_url, data, directory):
    if not data:
        data = 'ibis-testing-data.tar.gz',

    if not os.path.exists(directory):
        os.mkdir(directory)

    for piece in data:
        data_url = '{}/{}'.format(base_url, piece)
        path = os.path.join(directory, piece)

        curl(
            data_url, o=path, L=True,
            _out=click.get_binary_stream('stdout'),
            _err=click.get_binary_stream('stderr'),
        )

        if piece.endswith(('.tar', '.gz', '.bz2', '.xz')):
            with tarfile.open(path, mode='r|gz') as f:
                f.extractall(path=directory)


def parse_env(ctx, param, values):
    pairs = []
    for envar in values:
        try:
            name, value = envar.split('=', 1)
        except ValueError:
            raise click.ClickException(
                'Environment variables must be of the form NAME=VALUE. '
                '{} is not in this format'.format(envar)
            )
        pairs.append((name, value))
    return dict(pairs)


@cli.command()
@click.argument('data_directory', type=click.Path(exists=True))
@click.option('-e', '--environment', multiple=True, callback=parse_env)
def env(data_directory, environment):
    envars = dict([
        ('IBIS_TEST_IMPALA_HOST', 'impala'),
        ('IBIS_TEST_NN_HOST', 'impala'),
        ('IBIS_TEST_IMPALA_POST', 21050),
        ('IBIS_TEST_WEBHDFS_PORT', 50070),
        ('IBIS_TEST_WEBHDFS_USER', 'ubuntu'),
        (
            'IBIS_TEST_SQLITE_DB_PATH',
            os.path.join(data_directory, 'ibis_testing.db'),
        ),
        (
            'DIAMONDS_CSV',
            os.path.join(data_directory, 'diamonds.csv')
        ),
        (
            'BATTING_CSV',
            os.path.join(data_directory, 'batting.csv')
        ),
        (
            'AWARDS_PLAYERS_CSV',
            os.path.join(data_directory, 'awards_players.csv')
        ),
        (
            'FUNCTIONAL_ALLTYPES_CSV',
            os.path.join(data_directory, 'functional_alltypes.csv')
        ),
        ('IBIS_TEST_POSTGRES_DB', 'ibis_testing'),
        ('IBIS_POSTGRES_USER', getpass.getuser()),
        ('IBIS_POSTGRES_PASS', ''),
    ])
    envars.update(environment)
    string = '\n'.join(
        '='.join((name, str(value)))
        for name, value in sorted(envars.items(), key=operator.itemgetter(0))
    )
    click.echo(string)


if __name__ == '__main__':
    cli()
