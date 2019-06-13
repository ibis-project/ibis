"""Test support for already-defined UDFs in Postgres"""

import pytest

import ibis.expr.datatypes
import ibis.sql.postgres.udf.api


# Database setup (tables and UDFs)


table_name = 'udf_test_users'

sql_table_setup = """DROP TABLE IF EXISTS public.{table_name};
CREATE TABLE public.{table_name} (
    user_id integer,
    user_name varchar,
    name_length integer
);
INSERT INTO {table_name} VALUES
(1, 'Raj', 3),
(2, 'Judy', 4),
(3, 'Jonathan', 8)
;
""".format(table_name=table_name)

sql_create_plpython = "CREATE EXTENSION plpythonu"

sql_define_py_udf = """CREATE OR REPLACE FUNCTION public.pylen(x varchar)
RETURNS integer
LANGUAGE plpythonu
AS
$$
return len(x)
$$;"""

sql_define_udf = """CREATE OR REPLACE FUNCTION public.custom_len(x varchar)
RETURNS integer
LANGUAGE SQL
AS
$$
SELECT length(x);
$$;"""


@pytest.fixture
def con_for_udf(con):
    # with con.con.begin() as cur_conn:
    #     cur_conn.execute(sql_table_setup)
    #     yield con
    #     con.rollback()
    con.con.execute(sql_table_setup)
    con.con.execute(sql_define_udf)
    con.con.execute(sql_define_py_udf)
    # con.con.execute(sql_create_plpython)
    yield con


@pytest.fixture
def table(con_for_udf):
    return con_for_udf.table(table_name, schema='public')


# Create ibis UDF objects referring to UDFs already created in the database

custom_length_udf = ibis.sql.postgres.udf.api.existing_udf(
    'custom_len',
    input_type=[ibis.expr.datatypes.String()],
    output_type=ibis.expr.datatypes.Integer(),
    schema='public'
)

py_length_udf = ibis.sql.postgres.udf.api.existing_udf(
    'pylen',
    input_type=[ibis.expr.datatypes.String()],
    output_type=ibis.expr.datatypes.Integer(),
    schema='public'
)


# Tests

def test_sql_length_udf_worked(table):
    result_obj = table[
        table,
        custom_length_udf(table['user_name']).name('custom_len')
    ]
    result = result_obj.execute()
    assert result['custom_len'].sum() == result['name_length'].sum()


def test_py_length_udf_worked(table):
    result_obj = table[
        table,
        py_length_udf(table['user_name']).name('custom_len')
    ]
    result = result_obj.execute()
    assert result['custom_len'].sum() == result['name_length'].sum()
