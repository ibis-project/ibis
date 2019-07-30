import pyspark as ps
import pyspark.sql.types as pt
import regex as re
import toolz
from pkg_resources import parse_version

import ibis.common as com
import ibis.expr.datatypes as dt
import ibis.expr.lineage as lin
import ibis.expr.operations as ops
import ibis.expr.schema as sch
from ibis.client import Database, Query, SQLClient
from ibis.spark import compiler as comp

# maps pyspark type class to ibis type class
_SPARK_DTYPE_TO_IBIS_DTYPE = {
    pt.NullType : dt.Null,
    pt.StringType : dt.String,
    pt.BinaryType : dt.Binary,
    pt.BooleanType : dt.Boolean,
    pt.DateType : dt.Date,
    pt.DoubleType : dt.Double,
    pt.FloatType : dt.Float,
    pt.ByteType : dt.Int8,
    pt.IntegerType : dt.Int32,
    pt.LongType : dt.Int64,
    pt.ShortType : dt.Int16,
}


@dt.dtype.register(pt.DataType)
def spark_dtype_to_ibis_dtype(spark_dtype_obj, nullable=True):
    """Convert Spark SQL type objects to ibis type objects."""
    ibis_type_class = _SPARK_DTYPE_TO_IBIS_DTYPE.get(type(spark_dtype_obj))
    return ibis_type_class(nullable=nullable)


@dt.dtype.register(pt.TimestampType)
def spark_timestamp_dtype_to_ibis_dtype(spark_dtype_obj, nullable=True):
    return dt.Timestamp(nullable=nullable)


@dt.dtype.register(pt.DecimalType)
def spark_decimal_dtype_to_ibis_dtype(spark_dtype_obj, nullable=True):
    precision = spark_dtype_obj.precision
    scale = spark_dtype_obj.scale
    return dt.Decimal(precision, scale, nullable=nullable)


@dt.dtype.register(pt.ArrayType)
def spark_array_dtype_to_ibis_dtype(spark_dtype_obj, nullable=True):
    value_type = dt.dtype(
        spark_dtype_obj.elementType,
        nullable=spark_dtype_obj.containsNull
    )
    return dt.Array(value_type, nullable=nullable)


@dt.dtype.register(pt.MapType)
def spark_map_dtype_to_ibis_dtype(spark_dtype_obj, nullable=True):
    key_type = dt.dtype(spark_dtype_obj.keyType)
    value_type = dt.dtype(
        spark_dtype_obj.valueType,
        nullable=spark_dtype_obj.valueContainsNull
    )
    return dt.Map(key_type, value_type, nullable=nullable)


@dt.dtype.register(pt.StructType)
def spark_struct_dtype_to_ibis_dtype(spark_dtype_obj, nullable=True):
    names = spark_dtype_obj.names
    fields = spark_dtype_obj.fields
    ibis_types = [dt.dtype(f.dataType, nullable=f.nullable) for f in fields]
    return dt.Struct(names, ibis_types, nullable=nullable)


@sch.infer.register(ps.sql.dataframe.DataFrame)
def spark_dataframe_schema(df):
    """Infer the schema of a Spark SQL `DataFrame` object."""
    # df.schema is a pt.StructType
    schema_struct = dt.dtype(df.schema)

    return sch.schema(schema_struct.names, schema_struct.types)


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
        result = self.query.collect()  # blocks until finished
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


def find_spark_udf(expr):
    result = expr.op()
    if not isinstance(result, (comp.SparkUDFNode, comp.SparkUDAFNode)):
        result = None
    return lin.proceed, result


class SparkQuery(Query):

    def execute(self):
        udf_nodes = lin.traverse(find_spark_udf, self.expr)

        # UDFs are uniquely identified by the name of the Node subclass we
        # generate.
        udf_nodes_unique = toolz.unique(
            udf_nodes,
            key=lambda node: type(node).__name__
        )

        # register UDFs in pyspark
        for node in udf_nodes_unique:
            self.client._session.udf.register(
                type(node).__name__,
                node.udf_func,
            )

        return super().execute()

    def _fetch(self, cursor):
        df = cursor.query.toPandas()  # blocks until finished
        schema = self.schema()
        return schema.apply_to(df)


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

    def __init__(self, **kwargs):
        self._context = ps.SparkContext(**kwargs)
        self._session = ps.sql.SparkSession(self._context)
        self._catalog = self._session.catalog

    def close(self):
        """
        Close Spark connection and drop any temporary objects
        """
        self._context.stop()

    def _build_ast(self, expr, context):
        result = comp.build_ast(expr, context)
        return result

    def _execute(self, stmt, results=False):
        query = self._session.sql(stmt)
        if results:
            return SparkCursor(query)

    def database(self, name=None):
        return self.database_class(name or self.current_database, self)

    @property
    def current_database(self):
        """
        String name of the current database.
        """
        return self._catalog.currentDatabase()

    def _get_table_schema(self, table_name):
        return self.get_schema(table_name)

    def _get_schema_using_query(self, query):
        cur = self._execute(query, results=True)
        return spark_dataframe_schema(cur.query)

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
        results = [t.name for t in self._catalog.listTables(dbName=database)]
        if like:
            results = [
                database_name
                for database_name in results
                if re.match(like, database_name) is not None
            ]

        return results

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
        """
        Return a Schema object for the indicated table and database

        Parameters
        ----------
        table_name : string
          May be fully qualified
        database : string
          Spark does not have a database argument for its table() method,
          so this must be None

        Returns
        -------
        schema : ibis Schema
        """
        if database is not None:
            raise com.UnsupportedArgumentError(
                'Spark does not support database param for table'
            )

        df = self._session.table(table_name)

        return sch.infer(df)

    @property
    def version(self):
        return parse_version(ps.__version__)
