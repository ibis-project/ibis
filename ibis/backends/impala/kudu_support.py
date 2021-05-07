from functools import wraps as copydoc

import kudu
import pandas as pd

import ibis.expr.datatypes as dt
from ibis.backends.base_sql.ddl import (
    CreateTable,
    format_schema,
    format_tblproperties,
)
from ibis.common.exceptions import IbisError
from ibis.expr.api import schema

_kudu_type_to_ibis_typeclass = {
    'int8': dt.Int8,
    'int16': dt.Int16,
    'int32': dt.Int32,
    'int64': dt.Int64,
    'float': dt.Float,
    'double': dt.Double,
    'bool': dt.Boolean,
    'string': dt.String,
    'timestamp': dt.Timestamp,
}


class KuduImpalaInterface:
    """User-facing wrapper layer for the ImpalaClient."""

    def __init__(self, impala_client):
        self.impala_client = impala_client
        self.client = None

    @copydoc(kudu.client.Client.list_tables)
    def list_tables(self, filter=''):
        return self.client.list_tables(filter)

    @copydoc(kudu.client.Client.table_exists)
    def table_exists(self, name):
        return self.client.table_exists(name)

    def connect(
        self,
        host_or_hosts,
        port_or_ports=7051,
        rpc_timeout=None,
        admin_timeout=None,
    ):
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
        admin_timeout : kudu.TimeDelta
          See Kudu client documentation for details

        Returns
        -------
        None
        """
        self.client = kudu.connect(
            host_or_hosts,
            port_or_ports,
            rpc_timeout_ms=rpc_timeout,
            admin_timeout_ms=admin_timeout,
        )

    def _check_connected(self):
        if not self.is_connected:
            raise IbisError(
                'Please first connect to a Kudu cluster '
                'with client.kudu.connect'
            )

    @property
    def is_connected(self):
        # crude check for now
        return self.client is not None

    def create_table(
        self,
        impala_name,
        kudu_name,
        primary_keys=None,
        obj=None,
        schema=None,
        database=None,
        external=False,
        force=False,
    ):
        """
        Create an Kudu-backed table in the connected Impala cluster. For
        non-external tables, this will create a Kudu table with a compatible
        storage schema.

        This function is patterned after the ImpalaClient.create_table function
        designed for physical filesystems (like HDFS).

        Parameters
        ----------
        impala_name : string
          Name of the created Impala table
        kudu_name : string
          Name of hte backing Kudu table. Will be created if external=False
        primary_keys : list of column names
          List of
        obj : TableExpr or pandas.DataFrame, optional
          If passed, creates table from select statement results
        schema : ibis.Schema, optional
          Mutually exclusive with expr, creates an empty table with a
          particular schema
        database : string, default None (optional)
        external : boolean, default False
          If False, a new Kudu table will be created. Otherwise, the Kudu table
          must already exist.
        """
        self._check_connected()

        if not external and (primary_keys is None or len(primary_keys) == 0):
            raise ValueError(
                'Must specify primary keys when DDL creates a '
                'new Kudu table'
            )

        if obj is not None:
            if external:
                raise ValueError(
                    'Cannot create an external Kudu-Impala table '
                    'from an expression or DataFrame'
                )

            if isinstance(obj, pd.DataFrame):
                from .pandas_interop import write_temp_dataframe

                writer, to_insert = write_temp_dataframe(
                    self.impala_client, obj
                )
            else:
                to_insert = obj
            # XXX: exposing a lot of internals
            ast = self.impala_client._build_ast(to_insert)
            select = ast.queries[0]

            stmt = CTASKudu(
                impala_name,
                kudu_name,
                self.client.master_addrs,
                select,
                primary_keys,
                database=database,
            )
        else:
            if external:
                ktable = self.client.table(kudu_name)
                kschema = ktable.schema
                schema = schema_kudu_to_ibis(kschema)
                primary_keys = kschema.primary_keys()
            elif schema is None:
                raise ValueError(
                    'Must specify schema for new empty ' 'Kudu-backed table'
                )

            stmt = CreateTableKudu(
                impala_name,
                kudu_name,
                self.client.master_addrs,
                schema,
                primary_keys,
                external=external,
                database=database,
                can_exist=False,
            )

        self.impala_client._execute(stmt)

    def table(
        self, kudu_name, name=None, database=None, persist=False, external=True
    ):
        """
        Convenience to expose an existing Kudu table (using CREATE TABLE) as an
        Impala table. To create a new table both in the Hive Metastore with
        storage in Kudu, use create_table.

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
        # Law of demeter, but OK for now because internal class coupling
        name, database = self.impala_client._get_concrete_table_path(
            name, database, persist=persist
        )
        self.create_table(name, kudu_name, database=database, external=True)
        return self.impala_client._wrap_new_table(name, database, persist)


class CreateTableKudu(CreateTable):

    """
    Creates an Impala table that scans from a Kudu table
    """

    # TODO
    # - DISTRIBUTE BY HASH
    # - DISTRIBUTE BY RANGE`
    # - multi master test

    def __init__(
        self,
        table_name,
        kudu_table_name,
        master_addrs,
        schema,
        key_columns,
        external=True,
        **kwargs,
    ):
        self.kudu_table_name = kudu_table_name
        self.master_addrs = master_addrs
        self.schema = schema
        self.key_columns = key_columns
        CreateTable.__init__(self, table_name, external=external, **kwargs)

        self._validate()

    def _validate(self):
        pass

    def compile(self):
        return '{}\n{}\n{}'.format(
            self._create_line(),
            format_schema(self.schema),
            format_tblproperties(self._get_table_properties()),
        )

    _table_props_base = {
        'storage_handler': 'com.cloudera.kudu.hive.KuduStorageHandler'
    }

    def _get_table_properties(self):
        tbl_props = self._table_props_base.copy()

        addr_string = ', '.join(self.master_addrs)
        keys_string = ', '.join(self.key_columns)

        tbl_props.update(
            {
                'kudu.table_name': self.kudu_table_name,
                'kudu.master_addresses': addr_string,
                'kudu.key_columns': keys_string,
            }
        )

        return tbl_props


class CTASKudu(CreateTableKudu):
    def __init__(
        self,
        table_name,
        kudu_name,
        master_addrs,
        select,
        key_columns,
        database=None,
        external=False,
        can_exist=False,
    ):
        self.select = select
        CreateTableKudu.__init__(
            self,
            table_name,
            kudu_name,
            master_addrs,
            None,
            key_columns,
            database=database,
            external=external,
            can_exist=can_exist,
        )

    def compile(self):
        return '{}\n{} AS\n{}'.format(
            self._create_line(),
            format_tblproperties(self._get_table_properties()),
            self.select.compile(),
        )


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
