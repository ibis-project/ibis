# Copyright 2015 Cloudera Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import traceback
import turbodbc

from ibis.impala.client import (ImpalaConnection,
                                ImpalaClient,
                                ImpalaCursor)
from ibis.config import options
from ibis.client import Query
from ibis.sql.compiler import DDL


def connect(dsn=None, connection_string=None, turbodbc_options=None,
            nthreads=None, hdfs_client=None, pool_size=8):
    """
    Using ODBC  to create ImpalaClient connection.

    Parameters
    ----------
    dsn : str, optional
        Data source name as given in the (unix) odbc.ini file or (Windows)
        ODBC Data Source Administrator tool.
    connection_string : str, optional
        Preformatted ODBC connection string. Specifying this and dsn at the
        same time raises ParameterError
    turbodbc_options : dict, optional
        to be passed to turbodbc_options
    nthreads : int, default max(1, multiprocessing.cpu_count() / 2)
        to be passed to pyarrow.to_pandas() from turbodbc

    Returns
    -------
    ImpalaClient
    """

    if turbodbc_options is None:
        turbodbc_options = {}

    con = ImpalaODBCConnection(pool_size=pool_size,
                               connection_string=connection_string,
                               dsn=dsn, nthreads=nthreads,
                               turbodbc_options=turbodbc_options)

    try:
        client = ImpalaClient(con, hdfs_client=hdfs_client)

        if options.default_backend is None:
            options.default_backend = client
    except:
        con.close()
        raise

    return client


class ImpalaODBCConnection(ImpalaConnection):
    def _new_cursor(self):
        params = self.params.copy()

        options = turbodbc.make_options(autocommit=True,
                                        **params['turbodbc_options'])
        con = turbodbc.connect(dsn=params['dsn'],
                               connection_string=params['connection_string'],
                               turbodbc_options=options)

        self._connections.add(con)

        cursor = con.cursor()

        wrapper = ImpalaODBCCursor(cursor, self, con, self.database,
                                   self.options.copy())
        wrapper.set_options()

        return wrapper

    def ping(self):
        self._get_cursor()

    def execute(self, query, async=False):
        if isinstance(query, DDL):
            query = query.compile()

        cursor = self._get_cursor()
        self.log(query)

        try:
            cursor.execute(query)
        except:
            exc = traceback.format_exc()
            self.error('Exception caused by {}: {}'.format(query, exc))
            raise

        return cursor


class ImpalaODBCCursor(ImpalaCursor):
    def execute(self, stmt):
        self._cursor.execute(stmt)

    def fetchall(self):
        nthreads = self.con.params['nthreads']
        return self._cursor.fetchallarrow().to_pandas(nthreads=nthreads)


class ImpalaODBCQuery(Query):
    def _fetch(self, cursor):
        return cursor.fetchall()
