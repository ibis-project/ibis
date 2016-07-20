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


from .client import PostgreSQLClient
from .compiler import rewrites  # noqa


def compile(expr):
    """
    Force compilation of expression for the PostgreSQL target
    """
    from .client import PostgreSQLDialect
    from ibis.sql.alchemy import to_sqlalchemy
    return to_sqlalchemy(expr, dialect=PostgreSQLDialect)


def connect(host=None, user=None, password=None, port=None, database=None,
            url=None, driver=None):

    """
    Create an Ibis client connected to a PostgreSQL database.

    Multiple database files can be created using the attach() method

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
    """
    return PostgreSQLClient(host=host, user=user, password=password, port=port,
                            database=database, url=url, driver=driver)
