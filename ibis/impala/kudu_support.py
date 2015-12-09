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

from six import StringIO

from ibis.common import IbisError
from ibis.expr.api import schema
from ibis.impala import ddl
from ibis.util import implements as copydoc
import ibis.expr.datatypes as dt
import kudu


_kudu_type_to_ibis_typeclass = {
    'int8': dt.Int8,
    'int16': dt.Int16,
    'int32': dt.Int32,
    'int64': dt.Int64,
    'float': dt.Float,
    'double': dt.Double,
    'bool': dt.Boolean,
    'string': dt.String,
    'timestamp': dt.Timestamp
}


class KuduImpalaInterface(object):

    """
    User-facing wrapper layer for the ImpalaClient
    """

    def __init__(self, impala_client):
        self.impala_client = impala_client
        self.client = None

    @copydoc(kudu.client.Client.list_tables)
    def list_tables(self, filter=''):
        return self.client.list_tables(filter)

    @copydoc(kudu.client.Client.table_exists)
    def table_exists(self, name):
        return self.client.table_exists(name)

    def connect(self, host_or_hosts, port_or_ports=7051, rpc_timeout=None):
        """
        Pass-through connection interface to the Kudu client

        Parameters
        ----------
        host_or_hosts : string or list of strings
          If you have multiple Kudu masters for HA, pass a list
        port_or_ports : int or list of int, default 7051
          If you pass multiple host names, pass multiple ports
        rpc_timeout : kudu.TimeDelta
          See Kudu client documentation for details

        Returns
        -------
        None
        """
        self.client = kudu.connect(host_or_hosts, port_or_ports,
                                   rpc_timeout=rpc_timeout)

    def _check_connected(self):
        if not self.is_connected:
            raise IbisError('Please first connect to a Kudu cluster '
                            'with client.kudu.connect')

    @property
    def is_connected(self):
        # crude check for now
        return self.client is not None

    def table(self, kudu_name, name=None, database=None, persist=False,
              external=True):
        """
        Expose the indicated Kudu table (using CREATE TABLE) as an Impala
        table.

        Note: all tables created are EXTERNAL for now. Creates a temporary
        table (like parquet_file and others) unless persist=True.

        If you create a persistent table you can thereafter use it like any
        other Impala table.

        Parameters
        ----------
        kudu_name : string
          The name of the table in the Kudu cluster
        name : string, optional
          Name of the created table in Impala / Hive Metastore. Randomly
          generated if not specified.
        database : string, optional
          Database to create the table in. Uses the temp db if not provided
        persist : boolean, default False
          If True, do not drop the table upon Ibis garbage collection /
          interpreter shutdown. Be careful using this in conjunction with the
          `external` option.
        external : boolean, default True
          If True, create the Impala table as EXTERNAL so the Kudu data is not
          deleted when the Impala table is dropped

        Returns
        -------
        parquet_table : ImpalaTable
        """
        self._check_connected()

        # Law of demeter, but OK for now because internal class coupling
        name, database = (self.impala_client
                          ._get_concrete_table_path(name, database,
                                                    persist=persist))
        ktable = self.client.table(kudu_name)
        kschema = ktable.schema

        ibis_schema = schema_kudu_to_ibis(kschema)
        primary_keys = kschema.primary_keys()

        stmt = CreateTableKudu(name, kudu_name,
                               self.client.master_addrs,
                               ibis_schema, primary_keys,
                               external=external,
                               database=database,
                               can_exist=False)
        self.impala_client._execute(stmt)
        return self.impala_client._wrap_new_table(name, database, persist)


class CreateTableKudu(ddl.CreateTable):

    """
    Creates an Impala table that scans from a Kudu table
    """

    # TODO
    # - DISTRIBUTE BY HASH
    # - DISTRIBUTE BY RANGE`
    # - multi master test

    def __init__(self, table_name, kudu_table_name,
                 master_addrs, schema, key_columns,
                 external=True, **kwargs):
        self.kudu_table_name = kudu_table_name
        self.master_addrs = master_addrs
        self.schema = schema
        self.key_columns = key_columns
        ddl.CreateTable.__init__(self, table_name, external=external,
                                 **kwargs)

        self._validate()

    def _validate(self):
        pass

    def compile(self):
        buf = StringIO()
        buf.write(self._create_line())

        schema = ddl.format_schema(self.schema)
        buf.write('\n{0}'.format(schema))

        props = self._get_table_properties()
        buf.write('\n')
        buf.write(ddl.format_tblproperties(props))
        return buf.getvalue()

    _table_props_base = {
        'storage_handler': 'com.cloudera.kudu.hive.KuduStorageHandler'
    }

    def _get_table_properties(self):
        tbl_props = self._table_props_base.copy()

        addr_string = ', '.join(self.master_addrs)
        keys_string = ', '.join(self.key_columns)

        tbl_props.update({
            'kudu.table_name': self.kudu_table_name,
            'kudu.master_addresses': addr_string,
            'kudu.key_columns': keys_string
        })

        return tbl_props


class CTASKudu(CreateTableKudu):

    def __init__(self, table_name, kudu_name, master_addrs,
                 select, key_columns, database=None,
                 external=False, can_exist=False):
        self.select = select
        CreateTableKudu.__init__(self, table_name, kudu_name,
                                 master_addrs, None, key_columns,
                                 database=database,
                                 external=external,
                                 can_exist=can_exist)

    def compile(self):
        buf = StringIO()
        buf.write(self._create_line())

        props = self._get_table_properties()
        buf.write('\n')
        buf.write(ddl.format_tblproperties(props))

        select_query = self.select.compile()
        buf.write(' AS\n{0}'.format(select_query))
        return buf.getvalue()


def schema_kudu_to_ibis(kschema, drop_nn=False):
    ibis_types = []
    for i in range(len(kschema)):
        col = kschema[i]

        typeclass = _kudu_type_to_ibis_typeclass[col.type.name]

        if drop_nn:
            # For testing, because Impala does not have nullable types
            itype = typeclass(True)
        else:
            itype = typeclass(col.nullable)

        ibis_types.append((col.name, itype))

    return schema(ibis_types)
