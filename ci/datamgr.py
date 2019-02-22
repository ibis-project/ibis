#!/usr/bin/env python

import json
import os
import tarfile
import tempfile
import warnings

import click

import pandas as pd
import sqlalchemy as sa

from toolz import dissoc
from plumbum import local

import ibis
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR_NAME = 'ibis-testing-data'
DATA_DIR = Path(os.environ.get('IBIS_TEST_DATA_DIRECTORY',
                               SCRIPT_DIR / DATA_DIR_NAME))

TEST_TABLES = ['functional_alltypes', 'diamonds', 'batting',
               'awards_players']


logger = ibis.util.get_logger('datamgr')


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
def cli():
    pass


@cli.command()
@click.argument('name', default='{}.tar.gz'.format(DATA_DIR_NAME))
@click.option(
    '--base-url',
    default='https://api.github.com/repos/ibis-project/testing-data/'
            'tarball/master'
)
@click.option('-d', '--directory', default=SCRIPT_DIR)
def download(base_url, directory, name):
    from plumbum.cmd import curl
    from shutil import rmtree

    directory = Path(directory)
    if not directory.exists():
        directory.mkdir()

    path = directory / name
    data_dir = directory / DATA_DIR_NAME
    # when testing-data repository is updated, update this variable if needed
    download_data_dir_name = 'ibis-project-testing-data-a88a4b3'

    if not path.exists():
        download = curl[base_url, '-o', path, '-L']
        download(stdout=click.get_binary_stream('stdout'),
                 stderr=click.get_binary_stream('stderr'))
    else:
        logger.info('Skipping download: %s already exists', name)

    logger.info('Extracting archive to %s', directory)
    if path.suffix in ('.tar', '.gz', '.bz2', '.xz'):
        # remove existent folder
        if os.path.exists(data_dir):
            rmtree(str(data_dir))
        # extract all files
        with tarfile.open(str(path), mode='r|gz') as f:
            f.extractall(path=str(directory))
        # rename
        os.rename(directory / download_data_dir_name, data_dir)


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
            logger.warning('Ignored: %s', msg)
            return 0
        else:
            raise click.ClickException(msg)

    data_directory = Path(data_directory)
    for table, df in read_tables(tables, data_directory):
        arrow_table = pa.Table.from_pandas(df)
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
@click.option('-l', '--psql-path', type=click.Path(exists=True), default=None)
def postgres(schema, tables, data_directory, psql_path, **params):
    psql = local.get('psql', psql_path)
    data_directory = Path(data_directory)
    logger.info('Initializing PostgreSQL...')
    engine = init_database('postgresql', params, schema,
                           isolation_level='AUTOCOMMIT')

    query = "COPY {} FROM STDIN WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',')"
    database = params['database']
    for table in tables:
        src = data_directory / '{}.csv'.format(table)
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
@click.option('-P', '--port', default=9091, type=int)
@click.option('-u', '--user', default='mapd')
@click.option('-p', '--password', default='HyperInteractive')
@click.option('-D', '--database', default='ibis_testing')
@click.option('-S', '--schema', type=click.File('rt'),
              default=str(SCRIPT_DIR / 'schema' / 'mapd.sql'))
@click.option('-t', '--tables', multiple=True, default=TEST_TABLES + ['geo'])
@click.option('-d', '--data-directory', default=DATA_DIR)
def mapd(schema, tables, data_directory, **params):
    import pymapd

    data_directory = Path(data_directory)
    reserved_words = ['table', 'year', 'month']

    # connection
    logger.info('Initializing MapD...')
    if params['database'] != 'mapd':
        conn = pymapd.connect(
            host=params['host'],
            user=params['user'],
            password=params['password'],
            port=params['port'],
            dbname='mapd'
        )
        stmt = 'CREATE DATABASE {}'.format(params['database'])
        try:
            conn.execute(stmt)
        except Exception:
            logger.exception('MapD DDL statement %r failed', stmt)
        conn.close()

    conn = pymapd.connect(
        host=params['host'], user=params['user'],
        password=params['password'],
        port=params['port'], dbname=params['database']
    )

    # create tables
    for stmt in filter(None, map(str.strip, schema.read().split(';'))):
        try:
            conn.execute(stmt)
        except Exception:
            logger.exception('MapD DDL statement \n%r\n failed', stmt)

    # import data
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

        # rename fields
        for df_col in df.columns:
            if ' ' in df_col or ':' in df_col:
                column = df_col.replace(' ', '_').replace(':', '_')
            elif df_col in reserved_words:
                column = '{}_'.format(df_col)
            else:
                continue
            df.rename(columns={df_col: column}, inplace=True)

        # load geospatial data
        if table == 'geo':
            conn.load_table_rowwise(
                table, list(df.itertuples(index=False, name=None))
            )
        else:
            conn.load_table_columnar(table, df)

    conn.close()
    logger.info('Done!')


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
    logger.info('Initializing MySQL...')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
@click.option('-d', '--data-directory', default=DATA_DIR)
@click.option('-i', '--ignore-missing-dependency', is_flag=True, default=False)
def bigquery(data_directory, ignore_missing_dependency, **params):
    try:
        import google.api_core.exceptions
        from google.cloud import bigquery
    except ImportError:
        msg = 'google-cloud-bigquery dependency is missing'
        if ignore_missing_dependency:
            logger.warning('Ignored: %s', msg)
            return 0
        else:
            raise click.ClickException(msg)

    project_id = os.environ['GOOGLE_BIGQUERY_PROJECT_ID']
    bqclient = bigquery.Client(project=project_id)

    # Create testing dataset.
    testing_dataset = bqclient.dataset('testing')
    try:
        bqclient.create_dataset(bigquery.Dataset(testing_dataset))
    except google.api_core.exceptions.Conflict:
        pass  # Skip if already created.

    # Set up main data table.
    data_directory = Path(data_directory)
    functional_alltypes_path = data_directory / 'functional_alltypes.csv'
    functional_alltypes_schema = []
    schema_path = data_directory / 'functional_alltypes_bigquery_schema.json'
    with open(str(schema_path)) as schemafile:
        schema_json = json.load(schemafile)
        for field in schema_json:
            functional_alltypes_schema.append(
                bigquery.SchemaField.from_api_repr(field))
    load_config = bigquery.LoadJobConfig()
    load_config.skip_leading_rows = 1  # skip the header row.
    load_config.schema = functional_alltypes_schema

    # Load main data table.
    functional_alltypes_schema = []
    with open(str(functional_alltypes_path), 'rb') as csvfile:
        job = bqclient.load_table_from_file(
            csvfile,
            testing_dataset.table('functional_alltypes'),
            job_config=load_config).result()

        if job.error_result:
            raise click.ClickException(str(job.error_result))

    # Load an ingestion time partitioned table.
    functional_alltypes_path = data_directory / 'functional_alltypes.csv'
    with open(str(functional_alltypes_path), 'rb') as csvfile:
        load_config.time_partitioning = bigquery.TimePartitioning()
        job = bqclient.load_table_from_file(
            csvfile,
            testing_dataset.table('functional_alltypes_parted'),
            job_config=load_config).result()

        if job.error_result:
            raise click.ClickException(str(job.error_result))

    # Create a table with complex data types (nested and repeated).
    struct_table_path = data_directory / 'struct_table.avro'
    with open(str(struct_table_path), 'rb') as avrofile:
        load_config = bigquery.LoadJobConfig()
        load_config.source_format = 'AVRO'
        job = bqclient.load_table_from_file(
            avrofile,
            testing_dataset.table('struct_table'),
            job_config=load_config)

        if job.error_result:
            raise click.ClickException(str(job.error_result))

    # Create empty date-partitioned table.
    date_table = bigquery.Table(testing_dataset.table('date_column_parted'))
    date_table.schema = [
        bigquery.SchemaField('my_date_parted_col', 'DATE'),
        bigquery.SchemaField('string_col', 'STRING'),
        bigquery.SchemaField('int_col', 'INTEGER'),
    ]
    date_table.time_partitioning = bigquery.TimePartitioning(
        field='my_date_parted_col')
    bqclient.create_table(date_table)

    # Create empty timestamp-partitioned tables.
    timestamp_table = bigquery.Table(
        testing_dataset.table('timestamp_column_parted'))
    timestamp_table.schema = [
        bigquery.SchemaField('my_timestamp_parted_col', 'DATE'),
        bigquery.SchemaField('string_col', 'STRING'),
        bigquery.SchemaField('int_col', 'INTEGER'),
    ]
    timestamp_table.time_partitioning = bigquery.TimePartitioning(
        field='my_timestamp_parted_col')
    bqclient.create_table(timestamp_table)

    # Create a table with a numeric column
    numeric_table = bigquery.Table(
        testing_dataset.table('numeric_table'))
    numeric_table.schema = [
        bigquery.SchemaField('string_col', 'STRING'),
        bigquery.SchemaField('numeric_col', 'NUMERIC'),
    ]
    bqclient.create_table(numeric_table)

    df = pd.read_csv(
        str(data_directory / 'functional_alltypes.csv'),
        usecols=['string_col', 'double_col'],
        header=0,
    )
    with tempfile.NamedTemporaryFile(mode='a+b') as csvfile:
        df.to_csv(csvfile, header=False, index=False)
        csvfile.seek(0)

        load_config = bigquery.LoadJobConfig()
        load_config.skip_leading_rows = 1  # skip the header row.
        load_config.schema = numeric_table.schema

        job = bqclient.load_table_from_file(
            csvfile,
            testing_dataset.table('numeric_table'),
            job_config=load_config).result()

        if job.error_result:
            raise click.ClickException(str(job.error_result))


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
