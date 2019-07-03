"""Test support for already-defined UDFs in Postgres"""

import random
import string

import pytest

import pandas as pd
import ibis.expr.datatypes as dt
from ibis.sql.postgres import existing_udf, func_to_udf
from ibis.sql.postgres.client import PostgreSQLClient


# mark test module as postgresql (for ability to easily exclude,
# e.g. in conda build tests)
# (Temporarily adding `postgis` marker so Azure Windows pipeline will exclude
#     pl/python tests.
#     TODO: update Windows pipeline to exclude postgres_extensions
#     TODO: remove postgis marker below once Windows pipeline updated
pytestmark = [
    pytest.mark.postgresql,
    pytest.mark.postgis,
    pytest.mark.postgres_extensions,
]

# Database setup (tables and UDFs)


def gen_schema_name(basename='test'):
    """Generate a random alpha string starting with 'test_' to be used as a
    test schema name"""
    schema_name = '{}_{}'.format(
        basename,
        ''.join([random.choice(string.ascii_lowercase) for i in range(6)])
    )
    return schema_name


def create_test_schema(connection: PostgreSQLClient, max_tries=20):
    """Create a random test schema, necessary to allow for parallel pytest
    testing"""
    schema_exists = True
    tries = 0
    while schema_exists and tries < max_tries:
        new_name = gen_schema_name()
        sql_check_schema = """SELECT count(schema_name) as n_recs
        FROM information_schema.schemata
        WHERE lower(schema_name) = lower('{}');""".format(new_name)
        df_result = pd.read_sql(sql_check_schema, connection.con)
        schema_exists = df_result['n_recs'].iloc[0] > 0
        tries += 1
    assert tries > 0
    test_schema = new_name
    connection.con.execute("CREATE SCHEMA {};".format(new_name))
    return test_schema


@pytest.fixture
def test_schema(con):
    return create_test_schema(con)


@pytest.fixture
def table_name():
    return 'udf_test_users'


@pytest.fixture
def sql_table_setup(test_schema, table_name):
    return """DROP TABLE IF EXISTS {schema}.{table_name};
CREATE TABLE {schema}.{table_name} (
    user_id integer,
    user_name varchar,
    name_length integer
);
INSERT INTO {schema}.{table_name} VALUES
(1, 'Raj', 3),
(2, 'Judy', 4),
(3, 'Jonathan', 8)
;
""".format(schema=test_schema, table_name=table_name)


@pytest.fixture
def sql_define_py_udf(test_schema):
    return """CREATE OR REPLACE FUNCTION {schema}.pylen(x varchar)
RETURNS integer
LANGUAGE plpythonu
AS
$$
return len(x)
$$;""".format(schema=test_schema)


@pytest.fixture
def sql_define_udf(test_schema):
    return """CREATE OR REPLACE FUNCTION {schema}.custom_len(x varchar)
RETURNS integer
LANGUAGE SQL
AS
$$
SELECT length(x);
$$;""".format(schema=test_schema)


@pytest.fixture
def con_for_udf(
        con,
        test_schema,
        sql_table_setup,
        sql_define_udf,
        sql_define_py_udf
):
    create_test_schema(con)
    con.con.execute(sql_table_setup)
    con.con.execute(sql_define_udf)
    con.con.execute(sql_define_py_udf)
    try:
        yield con
    finally:
        # teardown
        con.con.execute("DROP SCHEMA IF EXISTS {} CASCADE".format(test_schema))


@pytest.fixture
def table(con_for_udf, table_name, test_schema):
    return con_for_udf.table(table_name, schema=test_schema)

# Tests


def test_sql_length_udf_worked(test_schema, table):
    """Test creating ibis UDF object based on existing UDF in the database"""
    # Create ibis UDF objects referring to UDFs already created in the database
    custom_length_udf = existing_udf(
        'custom_len',
        input_types=[dt.string],
        output_type=dt.int32,
        schema=test_schema
    )
    result_obj = table[
        table,
        custom_length_udf(table['user_name']).name('custom_len')
    ]
    result = result_obj.execute()
    assert result['custom_len'].sum() == result['name_length'].sum()


def test_py_length_udf_worked(test_schema, table):
    # Create ibis UDF objects referring to UDFs already created in the database
    py_length_udf = existing_udf(
        'pylen',
        input_types=[dt.string],
        output_type=dt.int32,
        schema=test_schema
    )
    result_obj = table[
        table,
        py_length_udf(table['user_name']).name('custom_len')
    ]
    result = result_obj.execute()
    assert result['custom_len'].sum() == result['name_length'].sum()


def mult_a_b(a, b):
    """Test function to be defined in-database as a UDF
    and used via ibis UDF"""
    return a * b


def test_func_to_udf_smoke(con_for_udf, test_schema, table):
    """Test creating a UDF in database based on Python function
    and then creating an ibis UDF object based on that"""
    mult_a_b_udf = func_to_udf(
        con_for_udf.con,
        mult_a_b,
        (dt.int32, dt.int32),
        dt.int32,
        schema=test_schema,
        overwrite=True
    )
    table_filt = table.filter(table['user_id'] == 2)
    expr = table_filt[
        mult_a_b_udf(
            table_filt['user_id'],
            table_filt['name_length']
        ).name('mult_result')
    ]
    result = expr.execute()
    assert result['mult_result'].iloc[0] == 8


@pytest.mark.xfail
def test_client_udf_api(con_for_udf, test_schema, table):
    """Test creating a UDF in database based on Python function
    using an ibis client method."""
    @con_for_udf.udf(
        [dt.int32, dt.int32],
        dt.int32,
        schema=test_schema,
        overwrite=True)
    def multiply(a, b):
        return a * b

    table_filt = table.filter(table['user_id'] == 2)
    expr = table_filt[
        multiply(
            table_filt['user_id'],
            table_filt['name_length']
        ).name('mult_result')
    ]
    result = expr.execute()
    assert result['mult_result'].iloc[0] == 8
