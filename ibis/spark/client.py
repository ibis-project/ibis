from ibis.client import Query, SQLClient
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.spark.compiler import SparkSQLQueryBuilder

from pyspark.sql.dataframe import DataFrame
import pyspark.sql.types as pt


class SparkSQLQuery(Query):

    def __init__(self, client, ddl, query_parameters=None):
        super(SparkSQLQuery, self).__init__(client, ddl)
        self.query_parameters = query_parameters or {}

    def execute(self):
        return self.client.session.sql(self.compiled_sql).toPandas()


class SparkClient(SQLClient):
    query_class = SparkSQLQuery
    _table_expr_klass = ir.TableExpr

    def __init__(self, session):
        self.session = session

    def _get_table_schema(self, name):
        t = self.session.table(name)
        return sch.infer(t)

    def list_tables(self):
        return self.session.catalog.listTables()

    def _build_ast(self, expr, context):
        builder = SparkSQLQueryBuilder(expr, context=context)
        return builder.get_result()


@dt.dtype.register(pt.StringType)
def pyspark_string(pstype, nullable=True):
    return dt.String(nullable=nullable)


@dt.dtype.register(pt.LongType)
def pyspark_long(pstype, nullable=True):
    return dt.Int64(nullable=nullable)


@dt.dtype.register(pt.DoubleType)
def pyspark_double(pstype, nullable=True):
    return dt.Float64(nullable=nullable)


@sch.infer.register(DataFrame)
def schema_from_table(table):
    pairs = []
    for f in table.schema.fields:
        dtype = dt.dtype(f.dataType)
        pairs.append((f.name, dtype))
    return sch.schema(pairs)
