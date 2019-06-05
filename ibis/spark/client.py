import copy
from collections import OrderedDict

import pyspark as ps
import pyspark.sql.types as pt
import regex as re

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
from ibis.spark import compiler as comp
from ibis.client import Database, Query, SQLClient


_DTYPE_TO_IBIS_TYPE = {
    pt.NullType : dt.null,
    pt.StringType : dt.string,
    pt.BinaryType : dt.binary,
    pt.BooleanType : dt.boolean,
    pt.DateType : dt.date,
    pt.TimestampType : dt.timestamp,
    pt.DoubleType : dt.double,
    pt.FloatType : dt.float,
    pt.ByteType : dt.int8,
    pt.IntegerType : dt.int32,
    pt.LongType : dt.int64,
    pt.ShortType : dt.int16,
}

@dt.dtype.register(pt.StructField)
def spark_field_to_ibis_dtype(field):
    """Convert Spark SQL `StructField` to an ibis type."""
    type_obj = field.dataType
    typ = type(type_obj)

    if typ is pt.DecimalType:
        precision = type_obj.precision
        scale = type_obj.scale
        ibis_type = dt.Decimal(precision, scale)
    elif typ is pt.ArrayType:
        value_type = dt.dtype(type_obj.elementType)
        nullable = type_obj.containsNull
        ibis_type = dt.Array(value_type, nullable)
    elif typ is pt.MapType:
        key_type = dt.dtype(type_obj.keyType)
        value_type = dt.dtype(type_obj.valueType)
        nullable = type_obj.valueContainsNull
        ibis_type = dt.Map(key_type, value_type, nullable)
    elif typ is pt.StructType:
        names = field.names
        fields = field.fields
        ibis_types = list(map(dt.dtype, fields))
        ibis_type = dt.Struct(names, ibis_types)
    else:
        ibis_type = _DTYPE_TO_IBIS_TYPE.get(typ)
    
    return ibis_type

@sch.infer.register(ps.sql.dataframe.DataFrame)
def spark_dataframe_schema(df):
    """Infer the schema of a Spark SQL `DataFrame` object."""
    fields = OrderedDict((el.name, dt.dtype(el)) for el in df.schema)

    # TODO
    # partition_info = table._properties.get('timePartitioning', None)

    # # We have a partitioned table
    # if partition_info is not None:
    #     partition_field = partition_info.get('field', NATIVE_PARTITION_COL)

    #     # Only add a new column if it's not already a column in the schema
    #     fields.setdefault(partition_field, dt.timestamp)
    return sch.schema(fields)


class SparkCursor:
    """Spark cursor.

    This allows the Spark client to reuse machinery in
    :file:`ibis/client.py`.

    """

    def __init__(self, query):
        """
        
        Construct a SparkCursor with query `query`.

        Parameters
        ----------
        query : pyspark.sql.DataFrame
          Contains result of query.

        """
        self.query = query

    def fetchall(self):
        """Fetch all rows."""
        result = self.query.collect() # blocks until finished
        return result

    @property
    def columns(self):
        """Return the columns of the result set."""
        return self.query.columns

    @property
    def description(self):
        """Get the fields of the result set's schema."""
        return self.query.schema

    def __enter__(self):
        # For compatibility when constructed from Query.execute()
        """No-op for compatibility.

        See Also
        --------
        ibis.client.Query.execute

        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """No-op for compatibility.

        See Also
        --------
        ibis.client.Query.execute

        """

class SparkQuery(Query):
    def __init__(self, client, ddl):
        super().__init__(client, ddl)

    def _fetch(self, cursor):
        df = cursor.query.toPandas() # blocks until finished
        schema = self.schema()
        return schema.apply_to(df)

    def execute(self):
        # synchronous by default
        with self.client._execute(
            self.compiled_sql,
            results=True,
        ) as cur:
            result = self._fetch(cur)

        return self._wrap_result(result)

class SparkDatabase(Database):
    pass

class SparkTable(ops.DatabaseTable):
    pass

class SparkClient(SQLClient):

    """
    An Ibis client interface that uses Spark SQL.
    """

    dialect = comp.SparkDialect
    database_class = SparkDatabase
    query_class = SparkQuery
    table_class = SparkTable

    def __init__(self):
        # TODO arguments
        self._context = ps.SparkContext()
        self._session = ps.sql.SparkSession(self._context)
        self._catalog = self._session.catalog
    
    # TODO necessary?
    def close(self):
        """
        Close Spark connection and drop any temporary objects
        """
        self._context.stop()
    
    def table(self, table_name, database=None):
        t = super().table(table_name, database)
        return t
    
    def _build_ast(self, expr, context):
        result = comp.build_ast(expr, context)
        return result

    def _execute(self, stmt):
        query = self._session.sql(stmt)
        return SparkCursor(query)
    
    def database(self, name=None):
        return self.database_class(name or self.current_database, self)

    @property
    def current_database(self):
        return self._catalog.currentDatabase()

    def _get_table_schema(self, table_name):
        # TODO check if this is correct functionality
        return self.get_schema(table_name)
    
    def list_tables(self, like=None, database=None):
        """
        List tables in the current (or indicated) database. Like the SHOW
        TABLES command.

        Parameters
        ----------
        like : string, default None
          e.g. 'foo*' to match all tables starting with 'foo'
        database : string, default None
          If not passed, uses the current/default database

        Returns
        -------
        results : list of strings
        """
        result = [t.name for t in self._catalog.listTables(dbName=database)]
        if like:
            results = [
                database_name
                for database_name in results
                if re.match(like, database_name) is not None
            ]
        
        return result

    def set_database(self, name):
        """
        Set the default database scope for client
        """
        self._catalog.setCurrentDatabase(name)

    def exists_database(self, name):
        """
        Checks if a given database exists

        Parameters
        ----------
        name : string
          Database name

        Returns
        -------
        if_exists : boolean
        """
        return bool(self.list_databases(like=name))
    
    def list_databases(self, like=None):
        """
        List databases in the Spark SQL cluster.

        Parameters
        ----------
        like : string, default None
          e.g. 'foo*' to match all tables starting with 'foo'

        Returns
        -------
        results : list of strings
        """
        results = [db.name for db in self._catalog.listDatabases()]
        if like:
            results = [
                database_name
                for database_name in results
                if re.match(like, database_name) is not None
            ]
        
        return results

    def get_schema(self, table_name, database=None):
        # TODO: Spark doesn't use database for its .table() method
        """
        Return a Schema object for the indicated table and database

        Parameters
        ----------
        table_name : string
          May be fully qualified
        database : string, default None

        Returns
        -------
        schema : ibis Schema
        """
        df = self._session.table(table_name)

        return sch.infer(df)