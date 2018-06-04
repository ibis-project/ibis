#!/usr/bin/env python

import os
import ibis
import click
import tempfile

from plumbum import local, CommandNotFound
from plumbum.cmd import rm, make, cmake

from ibis.compat import BytesIO, Path
from ibis.common import IbisError
from ibis.impala.tests.conftest import IbisTestEnv


SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = Path(os.environ.get('IBIS_TEST_DATA_DIRECTORY',
                               SCRIPT_DIR / 'ibis-testing-data'))


ENV = IbisTestEnv()


def make_ibis_client():
    hc = ibis.hdfs_connect(
        host=ENV.nn_host,
        port=ENV.webhdfs_port,
        auth_mechanism=ENV.auth_mechanism,
        verify=ENV.auth_mechanism not in ['GSSAPI', 'LDAP'],
        user=ENV.webhdfs_user
    )
    auth_mechanism = ENV.auth_mechanism
    if auth_mechanism == 'GSSAPI' or auth_mechanism == 'LDAP':
        print("Warning: ignoring invalid Certificate Authority errors")
    return ibis.impala.connect(
        host=ENV.impala_host,
        port=ENV.impala_port,
        auth_mechanism=ENV.auth_mechanism,
        hdfs_client=hc
    )


def can_write_to_hdfs(con):
    test_path = os.path.join(ENV.test_data_dir, ibis.util.guid())
    test_file = BytesIO(ibis.util.guid().encode('utf-8'))
    try:
        con.hdfs.put(test_path, test_file)
        con.hdfs.rm(test_path)
        return True
    except Exception:
        return False


def can_build_udfs():
    try:
        local.which('cmake')
    except CommandNotFound:
        print('Could not find cmake on PATH')
        return False
    try:
        local.which('make')
    except CommandNotFound:
        print('Could not find make on PATH')
        return False
    try:
        local.which('clang++')
    except CommandNotFound:
        print('Could not find LLVM on PATH; if IBIS_TEST_LLVM_CONFIG is set, '
              'try setting PATH="$($IBIS_TEST_LLVM_CONFIG --bindir):$PATH"')
        return False
    return True


def is_impala_loaded(con):
    if not con.hdfs.exists(ENV.test_data_dir):
        return False
    if not con.exists_database(ENV.test_data_db):
        return False
    return True


def is_udf_loaded(con):
    bitcode_dir = os.path.join(ENV.test_data_dir, 'udf')
    if con.hdfs.exists(bitcode_dir):
        return True
    return False


def upload_ibis_test_data_to_hdfs(con, data_path):
    hdfs = con.hdfs
    if hdfs.exists(ENV.test_data_dir):
        hdfs.rmdir(ENV.test_data_dir)
    hdfs.put(ENV.test_data_dir, data_path, verbose=True)


def create_test_database(con):
    if con.exists_database(ENV.test_data_db):
        con.drop_database(ENV.test_data_db, force=True)
    con.create_database(ENV.test_data_db)
    print('Created database {}'.format(ENV.test_data_db))

    con.create_table(
        'alltypes',
        schema=ibis.schema([
            ('a', 'int8'),
            ('b', 'int16'),
            ('c', 'int32'),
            ('d', 'int64'),
            ('e', 'float'),
            ('f', 'double'),
            ('g', 'string'),
            ('h', 'boolean'),
            ('i', 'timestamp')
        ]),
        database=ENV.test_data_db
    )
    print('Created empty table {}.`alltypes`'.format(ENV.test_data_db))


def create_parquet_tables(con):
    parquet_files = con.hdfs.ls(os.path.join(ENV.test_data_dir, 'parquet'))
    schemas = {
        'functional_alltypes': ibis.schema(
            [('id', 'int32'),
             ('bool_col', 'boolean'),
             ('tinyint_col', 'int8'),
             ('smallint_col', 'int16'),
             ('int_col', 'int32'),
             ('bigint_col', 'int64'),
             ('float_col', 'float'),
             ('double_col', 'double'),
             ('date_string_col', 'string'),
             ('string_col', 'string'),
             ('timestamp_col', 'timestamp'),
             ('year', 'int32'),
             ('month', 'int32')]),
        'tpch_region': ibis.schema(
            [('r_regionkey', 'int16'),
             ('r_name', 'string'),
             ('r_comment', 'string')])}
    tables = []
    for table_name in parquet_files:
        print('Creating {}'.format(table_name))
        # if no schema infer!
        schema = schemas.get(table_name)
        path = os.path.join(ENV.test_data_dir, 'parquet', table_name)
        table = con.parquet_file(path, schema=schema, name=table_name,
                                 database=ENV.test_data_db, persist=True)
        tables.append(table)
    return tables


def create_avro_tables(con):
    avro_files = con.hdfs.ls(os.path.join(ENV.test_data_dir, 'avro'))
    schemas = {
        'tpch_region_avro': {
            'type': 'record',
            'name': 'a',
            'fields': [
                {'name': 'R_REGIONKEY', 'type': ['null', 'int']},
                {'name': 'R_NAME', 'type': ['null', 'string']},
                {'name': 'R_COMMENT', 'type': ['null', 'string']}]}}
    tables = []
    for table_name in avro_files:
        print('Creating {}'.format(table_name))
        schema = schemas[table_name]
        path = os.path.join(ENV.test_data_dir, 'avro', table_name)
        table = con.avro_file(path, schema, name=table_name,
                              database=ENV.test_data_db, persist=True)
        tables.append(table)
    return tables


def build_udfs():
    print('Building UDFs')
    ibis_home_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    udf_dir = os.path.join(ibis_home_dir, 'ci', 'udf')

    with local.cwd(udf_dir):
        assert (cmake('.') and make('VERBOSE=1'))


def upload_udfs(con):
    ibis_home_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.path.join(ibis_home_dir, 'ci', 'udf', 'build')
    bitcode_dir = os.path.join(ENV.test_data_dir, 'udf')
    print('Uploading UDFs to {}'.format(bitcode_dir))
    if con.hdfs.exists(bitcode_dir):
        con.hdfs.rmdir(bitcode_dir)
    con.hdfs.put(bitcode_dir, build_dir, verbose=True)


# ==========================================


@click.group(context_settings={'help_option_names': ['-h', '--help']})
def main():
    """Manage test data for Ibis"""
    pass


@main.command()
@click.option(
    '--data/--no-data', default=True, help='Load (skip) ibis testing data'
)
@click.option(
    '--udf/--no-udf', default=True, help='Build/upload (skip) test UDFs'
)
@click.option(
    '--data-dir',
    help=(
        'Path to testing data. This downloads data from Google Cloud Storage '
        'if unset'
    ),
    default=DATA_DIR
)
@click.option(
    '--overwrite', is_flag=True, help='Forces overwriting of data/UDFs'
)
def load(data, udf, data_dir, overwrite):
    """Load Ibis test data and build/upload UDFs"""
    print(str(ENV))

    con = make_ibis_client()

    # validate our environment before performing possibly expensive operations
    if not can_write_to_hdfs(con):
        raise IbisError('Failed to write to HDFS; check your settings')
    if udf and not can_build_udfs():
        raise IbisError('Build environment does not support building UDFs')

    # load the data files
    if data:
        tmp_dir = tempfile.mkdtemp(prefix='__ibis_tmp_')
        try:
            load_impala_data(con, str(data_dir), overwrite)
        finally:
            rm('-rf', tmp_dir)
    else:
        print('Skipping Ibis test data load (--no-data)')

    # build and upload the UDFs
    if udf:
        already_loaded = is_udf_loaded(con)
        print('Attempting to build and load test UDFs')
        if already_loaded and not overwrite:
            print('UDFs already loaded and not overwriting; moving on')
        else:
            if already_loaded:
                print('UDFs already loaded; attempting to overwrite')
            print('Building UDFs')
            build_udfs()
            print('Uploading UDFs')
            upload_udfs(con)
    else:
        print('Skipping UDF build/load (--no-udf)')


def load_impala_data(con, data_dir, overwrite=False):
    already_loaded = is_impala_loaded(con)
    print('Attempting to load Ibis Impala test data (--data)')
    if already_loaded and not overwrite:
        print('Data is already loaded and not overwriting; moving on')
    else:
        if already_loaded:
            print('Data is already loaded; attempting to overwrite')

        print('Uploading to HDFS')
        upload_ibis_test_data_to_hdfs(con, data_dir)
        print('Creating Ibis test data database')
        create_test_database(con)
        parquet_tables = create_parquet_tables(con)
        avro_tables = create_avro_tables(con)
        for table in parquet_tables + avro_tables:
            print('Computing stats for', table.op().name)
            table.compute_stats()


@main.command()
@click.option(
    '--test-data', is_flag=True,
    help='Cleanup Ibis test data, test database, and also the test UDFs if '
    'they are stored in the test data directory/database'
)
@click.option('--udfs', is_flag=True, help='Cleanup Ibis test UDFs only')
@click.option(
    '--tmp-data', is_flag=True, help='Cleanup Ibis temporary HDFS directory'
)
@click.option('--tmp-db', is_flag=True, help='Cleanup Ibis temporary database')
def cleanup(test_data, udfs, tmp_data, tmp_db):
    """Cleanup Ibis test data and UDFs"""
    print(str(ENV))

    con = make_ibis_client()

    if udfs:
        # this comes before test_data bc the latter clobbers this too
        con.hdfs.rmdir(os.path.join(ENV.test_data_dir, 'udf'))

    if test_data:
        con.drop_database(ENV.test_data_db, force=True)
        con.hdfs.rmdir(ENV.test_data_dir)

    if tmp_data:
        con.hdfs.rmdir(ENV.tmp_dir)

    if tmp_db:
        con.drop_database(ENV.tmp_db, force=True)


if __name__ == '__main__':
    main()
