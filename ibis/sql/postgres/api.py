# Copyright 2015 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from ibis.sql.alchemy import to_sqlalchemy

from .client import PostgreSQLClient, PostgreSQLDialect
from .compiler import rewrites  # noqa


def compile(expr):
    """Compile an ibis expression to the PostgreSQL target.

    Parameters
    ----------
    expr : ibis.expr.types.Expr
        The ibis expression to compile

    Returns
    -------
    sqlalchemy_expression : sqlalchemy.sql.expression.ClauseElement

    Examples
    --------
    >>> import os
    >>> import getpass
    >>> user = os.environ.get('IBIS_POSTGRES_USER', getpass.getuser())
    >>> password = os.environ.get('IBIS_POSTGRES_PASS')
    >>> database = os.environ.get('IBIS_TEST_POSTGRES_DB', 'ibis_testing')
    >>> con = connect(
    ...     database=database,
    ...     host='localhost',
    ...     user=user,
    ...     password=password
    ... )
    >>> t = con.table('functional_alltypes')
    >>> expr = t.double_col + 1
    >>> sqla = compile(expr)
    >>> print(str(sqla))  # doctest: +NORMALIZE_WHITESPACE
    SELECT t0.double_col + %(param_1)s AS tmp
    FROM functional_alltypes AS t0
    """
    return to_sqlalchemy(expr, dialect=PostgreSQLDialect)


def connect(
    host=None,
    user=None,
    password=None,
    port=None,
    database=None,
    url=None,
    driver=None
):

    """Create an Ibis client located at `user`:`password`@`host`:`port`
    connected to a PostgreSQL database named `database`.

    Parameters
    ----------
    host : string, default None
    user : string, default None
    password : string, default None
    port : string or integer, default None
    database : string, default None
    url : string, default None
        Complete SQLAlchemy connection string. If passed, the other connection
        arguments are ignored.
    driver : string, default 'psycopg2'

    Returns
    -------
    PostgreSQLClient

    Examples
    --------
    >>> import os
    >>> import getpass
    >>> user = os.environ.get('IBIS_POSTGRES_USER', getpass.getuser())
    >>> password = os.environ.get('IBIS_POSTGRES_PASS')
    >>> database = os.environ.get('IBIS_TEST_POSTGRES_DB', 'ibis_testing')
    >>> con = connect(
    ...     database=database,
    ...     host='localhost',
    ...     user=user,
    ...     password=password
    ... )
    >>> con.list_tables()  # doctest: +ELLIPSIS
    [...]
    >>> t = con.table('functional_alltypes')
    >>> t
    PostgreSQLTable[table]
      name: functional_alltypes
      schema:
        index : int64
        Unnamed: 0 : int64
        id : int32
        bool_col : boolean
        tinyint_col : int16
        smallint_col : int16
        int_col : int32
        bigint_col : int64
        float_col : float
        double_col : double
        date_string_col : string
        string_col : string
        timestamp_col : timestamp
        year : int32
        month : int32
    """
    return PostgreSQLClient(
        host=host,
        user=user,
        password=password,
        port=port,
        database=database,
        url=url,
        driver=driver,
    )
