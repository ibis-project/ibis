from __future__ import annotations

import itertools
import re
import zoneinfo
from typing import TYPE_CHECKING, Any

import sqlglot as sg
import sqlglot.expressions as sge

import ibis
import ibis.backends.sql.compilers as sc
import ibis.common.exceptions as exc
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import (
    CanCreateDatabase,
    HasCurrentCatalog,
    HasCurrentDatabase,
    NoUrl,
    PyArrowExampleLoader,
    SupportsTempTables,
)
from ibis.backends.beam.ddl import (
    CreateDatabase,
    CreateTableWithSchema,
    DropDatabase,
    DropTable,
    DropView,
    InsertSelect,
    RenameTable,
)
from ibis.backends.sql import SQLBackend
from ibis.expr.operations.udf import InputType
from ibis.util import gen_name

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import pandas as pd
    import pyarrow as pa
    import apache_beam as beam
    from apache_beam.pipeline import Pipeline
    from apache_beam.runners.runner import PipelineResult

    from ibis.expr.api import Watermark

_INPUT_TYPE_TO_FUNC_TYPE = {InputType.PYTHON: "general", InputType.PANDAS: "pandas"}


class Backend(
    SupportsTempTables,
    SQLBackend,
    CanCreateDatabase,
    HasCurrentCatalog,
    HasCurrentDatabase,
    NoUrl,
    PyArrowExampleLoader,
):
    name = "beam"
    compiler = sc.beam.compiler
    supports_temporary_tables = True
    supports_python_udfs = True

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        """No-op."""

    def do_connect(self, pipeline: Pipeline) -> None:
        """Create a Beam `Backend` for use with Ibis.

        Parameters
        ----------
        pipeline
            A Beam pipeline.

        Examples
        --------
        >>> import ibis
        >>> import apache_beam as beam
        >>> pipeline = beam.Pipeline()
        >>> ibis.beam.connect(pipeline)  # doctest: +ELLIPSIS
        <ibis.backends.beam.Backend object at 0x...>
        """
        self._pipeline = pipeline
        self._temp_tables = {}
        self._pipeline_options = {}
        self._runner_config = {}

    @util.experimental
    @classmethod
    def from_connection(cls, pipeline: Pipeline, /) -> Backend:
        """Create a Beam `Backend` from an existing pipeline.

        Parameters
        ----------
        pipeline
            A Beam pipeline.
        """
        return ibis.beam.connect(pipeline)

    def disconnect(self) -> None:
        pass

    def raw_sql(self, query: str) -> PipelineResult:
        """Execute raw SQL query using Beam SQL.
        
        Parameters
        ----------
        query
            SQL query string
            
        Returns
        -------
        PipelineResult
            Result of the pipeline execution
        """
        import apache_beam as beam
        from apache_beam.sql import SqlTransform
        
        # Check if this is a SET statement for configuration
        if self._is_set_statement(query):
            return self._handle_set_statement(query)
        
        # Check if this is a CREATE CATALOG statement
        if self._is_create_catalog_statement(query):
            return self._handle_create_catalog_statement(query)
        
        # Check if this is a CREATE DATABASE statement
        if self._is_create_database_statement(query):
            return self._handle_create_database_statement(query)
        
        # For regular SQL queries, create a transform from the SQL query
        sql_transform = SqlTransform(query)
        
        # Apply the transform to the pipeline
        result = self._pipeline | sql_transform
        
        return result

    def _is_set_statement(self, query: str) -> bool:
        """Check if the query is a SET statement for configuration.
        
        Parameters
        ----------
        query
            SQL query string
            
        Returns
        -------
        bool
            True if this is a SET statement
        """
        query_upper = query.strip().upper()
        return query_upper.startswith('SET ')

    def _handle_set_statement(self, query: str) -> PipelineResult:
        """Handle SET statements for pipeline configuration.
        
        Parameters
        ----------
        query
            SET statement string
            
        Returns
        -------
        PipelineResult
            Dummy result for SET statements
        """
        import apache_beam as beam
        import re
        
        # Parse SET statement: SET key = value
        pattern = r'SET\s+(\w+)\s*=\s*(.+)'
        match = re.match(pattern, query.strip(), re.IGNORECASE)
        
        if not match:
            raise exc.IbisError(f"Invalid SET statement: {query}")
        
        key = match.group(1).lower()
        value = match.group(2).strip().strip("'\"")
        
        # Handle different types of configuration
        if key == 'runner':
            self._configure_runner(value)
        elif key == 'catalog':
            # Set current catalog
            self._current_catalog = value
        elif key == 'database':
            # Set current database
            self._current_database = value
        elif key.startswith('catalog.'):
            # Catalog-specific options
            option_key = key.replace('catalog.', '')
            if not hasattr(self, '_catalog_options'):
                self._catalog_options = {}
            self._catalog_options[option_key] = value
        elif key.startswith('pipeline.'):
            # Pipeline options
            option_key = key.replace('pipeline.', '')
            self._pipeline_options[option_key] = value
        elif key.startswith('dataflow.'):
            # Dataflow-specific options
            option_key = key.replace('dataflow.', '')
            self._runner_config[option_key] = value
        else:
            # Generic configuration
            self._pipeline_options[key] = value
        
        # Return a dummy result for SET statements
        return beam.Create([{'status': 'SET', 'key': key, 'value': value}])

    def _configure_runner(self, runner_name: str) -> None:
        """Configure the Beam runner.
        
        Parameters
        ----------
        runner_name
            Name of the runner to configure
        """
        import apache_beam as beam
        
        runner_name = runner_name.lower()
        
        if runner_name == 'dataflow':
            self._runner_config['runner'] = 'DataflowRunner'
        elif runner_name == 'flink':
            self._runner_config['runner'] = 'FlinkRunner'
        elif runner_name == 'spark':
            self._runner_config['runner'] = 'SparkRunner'
        elif runner_name == 'direct':
            self._runner_config['runner'] = 'DirectRunner'
        elif runner_name == 'portable':
            self._runner_config['runner'] = 'PortableRunner'
        else:
            raise exc.IbisError(f"Unknown runner: {runner_name}")

    def get_pipeline_options(self) -> dict:
        """Get current pipeline options.
        
        Returns
        -------
        dict
            Dictionary of pipeline options
        """
        return self._pipeline_options.copy()

    def get_runner_config(self) -> dict:
        """Get current runner configuration.
        
        Returns
        -------
        dict
            Dictionary of runner configuration
        """
        return self._runner_config.copy()

    def list_catalogs(self, *, like: str | None = None) -> list[str]:
        """List available catalogs.
        
        Parameters
        ----------
        like
            Pattern to filter catalog names
            
        Returns
        -------
        list[str]
            List of catalog names
        """
        # With upcoming Apache Iceberg catalog support, we can list actual catalogs
        try:
            if hasattr(self._pipeline, 'list_catalogs'):
                catalogs = self._pipeline.list_catalogs()
            else:
                # Fallback to default catalog
                catalogs = ["default"]
        except Exception:
            catalogs = ["default"]
        
        return self._filter_with_like(catalogs, like)

    def create_catalog(
        self,
        name: str,
        /,
        *,
        catalog_type: str = "iceberg",
        properties: dict | None = None,
        force: bool = False,
    ) -> None:
        """Create a new catalog.

        Parameters
        ----------
        name
            Name of the new catalog.
        catalog_type
            Type of catalog (e.g., 'iceberg', 'hive').
        properties
            Properties of the catalog.
        force
            If `False`, an exception is raised if the catalog already exists.
        """
        # With upcoming Apache Iceberg catalog support, we can create actual catalogs
        try:
            if hasattr(self._pipeline, 'create_catalog'):
                self._pipeline.create_catalog(name, catalog_type=catalog_type, properties=properties)
            else:
                # For now, create a placeholder catalog entry
                if not hasattr(self, '_catalogs'):
                    self._catalogs = {}
                self._catalogs[name] = {'type': catalog_type, 'properties': properties or {}}
        except Exception as e:
            if not force:
                raise exc.IbisError(f"Failed to create catalog '{name}': {e}")

    def drop_catalog(
        self, name: str, /, *, force: bool = False
    ) -> None:
        """Drop a catalog.

        Parameters
        ----------
        name
            Name of the catalog to drop.
        force
            If `False`, an exception is raised if the catalog does not exist.
        """
        try:
            if hasattr(self._pipeline, 'drop_catalog'):
                self._pipeline.drop_catalog(name)
            else:
                # For now, remove from placeholder catalogs
                if hasattr(self, '_catalogs') and name in self._catalogs:
                    del self._catalogs[name]
        except Exception as e:
            if not force:
                raise exc.IbisError(f"Failed to drop catalog '{name}': {e}")

    def _is_create_catalog_statement(self, query: str) -> bool:
        """Check if the query is a CREATE CATALOG statement.
        
        Parameters
        ----------
        query
            SQL query string
            
        Returns
        -------
        bool
            True if this is a CREATE CATALOG statement
        """
        query_upper = query.strip().upper()
        return query_upper.startswith('CREATE CATALOG ')

    def _is_create_database_statement(self, query: str) -> bool:
        """Check if the query is a CREATE DATABASE statement.
        
        Parameters
        ----------
        query
            SQL query string
            
        Returns
        -------
        bool
            True if this is a CREATE DATABASE statement
        """
        query_upper = query.strip().upper()
        return query_upper.startswith('CREATE DATABASE ')

    def _handle_create_catalog_statement(self, query: str) -> PipelineResult:
        """Handle CREATE CATALOG statements.
        
        Parameters
        ----------
        query
            CREATE CATALOG statement string
            
        Returns
        -------
        PipelineResult
            Dummy result for CREATE CATALOG statements
        """
        import apache_beam as beam
        import re
        
        # Parse CREATE CATALOG statement
        # Example: CREATE CATALOG my_catalog WITH (type = 'iceberg', warehouse = 's3://bucket/warehouse')
        pattern = r'CREATE CATALOG\s+(\w+)(?:\s+WITH\s*\(([^)]+)\))?'
        match = re.match(pattern, query.strip(), re.IGNORECASE)
        
        if not match:
            raise exc.IbisError(f"Invalid CREATE CATALOG statement: {query}")
        
        catalog_name = match.group(1)
        properties_str = match.group(2) if match.group(2) else ""
        
        # Parse properties
        properties = {}
        if properties_str:
            # Simple property parsing: key = 'value', key2 = 'value2'
            prop_pattern = r"(\w+)\s*=\s*'([^']+)'"
            for prop_match in re.finditer(prop_pattern, properties_str):
                key = prop_match.group(1)
                value = prop_match.group(2)
                properties[key] = value
        
        # Determine catalog type from properties
        catalog_type = properties.get('type', 'iceberg')
        
        # Create the catalog
        self.create_catalog(catalog_name, catalog_type=catalog_type, properties=properties)
        
        # Return a dummy result
        return beam.Create([{'status': 'CREATE CATALOG', 'catalog': catalog_name, 'type': catalog_type}])

    def _handle_create_database_statement(self, query: str) -> PipelineResult:
        """Handle CREATE DATABASE statements.
        
        Parameters
        ----------
        query
            CREATE DATABASE statement string
            
        Returns
        -------
        PipelineResult
            Dummy result for CREATE DATABASE statements
        """
        import apache_beam as beam
        import re
        
        # Parse CREATE DATABASE statement
        # Example: CREATE DATABASE my_database IN CATALOG my_catalog
        pattern = r'CREATE DATABASE\s+(\w+)(?:\s+IN\s+CATALOG\s+(\w+))?(?:\s+WITH\s*\(([^)]+)\))?'
        match = re.match(pattern, query.strip(), re.IGNORECASE)
        
        if not match:
            raise exc.IbisError(f"Invalid CREATE DATABASE statement: {query}")
        
        database_name = match.group(1)
        catalog_name = match.group(2) if match.group(2) else self.current_catalog
        properties_str = match.group(3) if match.group(3) else ""
        
        # Parse properties
        properties = {}
        if properties_str:
            # Simple property parsing: key = 'value', key2 = 'value2'
            prop_pattern = r"(\w+)\s*=\s*'([^']+)'"
            for prop_match in re.finditer(prop_pattern, properties_str):
                key = prop_match.group(1)
                value = prop_match.group(2)
                properties[key] = value
        
        # Create the database
        self.create_database(database_name, db_properties=properties, catalog=catalog_name)
        
        # Return a dummy result
        return beam.Create([{'status': 'CREATE DATABASE', 'database': database_name, 'catalog': catalog_name}])

    def create_configured_pipeline(self) -> Pipeline:
        """Create a new pipeline with the configured options.
        
        Returns
        -------
        Pipeline
            Configured Beam pipeline
        """
        import apache_beam as beam
        from apache_beam.options.pipeline_options import PipelineOptions
        
        # Create pipeline options
        options = PipelineOptions()
        
        # Apply pipeline options
        for key, value in self._pipeline_options.items():
            setattr(options, key, value)
        
        # Apply runner configuration
        if 'runner' in self._runner_config:
            runner_name = self._runner_config['runner']
            if runner_name == 'DataflowRunner':
                from apache_beam.options.pipeline_options import GoogleCloudOptions
                options.view_as(GoogleCloudOptions)
            elif runner_name == 'FlinkRunner':
                from apache_beam.runners.flink.flink_runner import FlinkRunner
                options.view_as(FlinkRunner)
            elif runner_name == 'SparkRunner':
                from apache_beam.runners.spark.spark_runner import SparkRunner
                options.view_as(SparkRunner)
        
        # Apply Dataflow-specific options
        if 'runner' in self._runner_config and self._runner_config['runner'] == 'DataflowRunner':
            from apache_beam.options.pipeline_options import GoogleCloudOptions
            gcp_options = options.view_as(GoogleCloudOptions)
            
            for key, value in self._runner_config.items():
                if key == 'runner':
                    continue
                elif key == 'project':
                    gcp_options.project = value
                elif key == 'region':
                    gcp_options.region = value
                elif key == 'staging_location':
                    gcp_options.staging_location = value
                elif key == 'temp_location':
                    gcp_options.temp_location = value
                elif key == 'service_account':
                    gcp_options.service_account = value
                elif key == 'network':
                    gcp_options.network = value
                elif key == 'subnetwork':
                    gcp_options.subnetwork = value
                elif key == 'use_public_ips':
                    gcp_options.use_public_ips = value.lower() == 'true'
                elif key == 'num_workers':
                    gcp_options.num_workers = int(value)
                elif key == 'max_num_workers':
                    gcp_options.max_num_workers = int(value)
                elif key == 'machine_type':
                    gcp_options.machine_type = value
                elif key == 'disk_size_gb':
                    gcp_options.disk_size_gb = int(value)
                elif key == 'disk_type':
                    gcp_options.disk_type = value
                elif key == 'worker_machine_type':
                    gcp_options.worker_machine_type = value
                elif key == 'worker_disk_type':
                    gcp_options.worker_disk_type = value
                elif key == 'worker_disk_size_gb':
                    gcp_options.worker_disk_size_gb = int(value)
                elif key == 'autoscaling_algorithm':
                    gcp_options.autoscaling_algorithm = value
                elif key == 'enable_streaming_engine':
                    gcp_options.enable_streaming_engine = value.lower() == 'true'
                elif key == 'flexrs_goal':
                    gcp_options.flexrs_goal = value
                elif key == 'dataflow_kms_key':
                    gcp_options.dataflow_kms_key = value
                elif key == 'labels':
                    # Parse labels as key=value pairs
                    labels = {}
                    for label_pair in value.split(','):
                        if '=' in label_pair:
                            k, v = label_pair.split('=', 1)
                            labels[k.strip()] = v.strip()
                    gcp_options.labels = labels
        
        # Create and return the pipeline
        return beam.Pipeline(options=options)

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        """Get schema from a SQL query.
        
        Parameters
        ----------
        query
            SQL query string
            
        Returns
        -------
        sch.Schema
            Schema of the query result
        """
        # For Beam SQL, we need to analyze the query to determine the schema
        # This is a simplified implementation - in practice, you'd need to
        # parse the SQL and determine the output schema
        import apache_beam as beam
        from apache_beam.sql import SqlTransform
        
        # Create a temporary transform to analyze the schema
        sql_transform = SqlTransform(query)
        
        # This is a placeholder - actual implementation would need to
        # analyze the transform to determine output schema
        # For now, return an empty schema
        return sch.Schema({})

    def list_databases(self, *, like: str | None = None) -> list[str]:
        """List available databases.
        
        Parameters
        ----------
        like
            Pattern to filter database names
            
        Returns
        -------
        list[str]
            List of database names
        """
        # With upcoming Apache Iceberg catalog support, we can now list actual databases
        # For now, we'll use a placeholder implementation that will be enhanced
        # when the catalog support is available in Beam
        try:
            # Try to use catalog functionality if available
            if hasattr(self._pipeline, 'list_databases'):
                databases = self._pipeline.list_databases()
            else:
                # Fallback to default database
                databases = ["default"]
        except Exception:
            # If catalog functionality is not available yet, use default
            databases = ["default"]
        
        return self._filter_with_like(databases, like)

    @property
    def current_catalog(self) -> str:
        """Get current catalog name."""
        return getattr(self, '_current_catalog', "default")

    @property
    def current_database(self) -> str:
        """Get current database name."""
        return getattr(self, '_current_database', "default")

    def create_database(
        self,
        name: str,
        /,
        *,
        db_properties: dict | None = None,
        catalog: str | None = None,
        force: bool = False,
    ) -> None:
        """Create a new database.

        Parameters
        ----------
        name
            Name of the new database.
        db_properties
            Properties of the database.
        catalog
            Name of the catalog in which the new database will be created.
        force
            If `False`, an exception is raised if the database already exists.
        """
        # With upcoming Apache Iceberg catalog support, we can create actual databases
        try:
            # Try to use catalog functionality if available
            if hasattr(self._pipeline, 'create_database'):
                self._pipeline.create_database(name, properties=db_properties, catalog=catalog)
            else:
                # For now, create a placeholder database entry
                if not hasattr(self, '_databases'):
                    self._databases = set()
                self._databases.add(name)
        except Exception as e:
            if not force:
                raise exc.IbisError(f"Failed to create database '{name}': {e}")
            # If force=True, silently ignore the error

    def drop_database(
        self, name: str, /, *, catalog: str | None = None, force: bool = False
    ) -> None:
        """Drop a database.

        Parameters
        ----------
        name
            Database to drop.
        catalog
            Name of the catalog from which the database will be dropped.
        force
            If `False`, an exception is raised if the database does not exist.
        """
        # Beam SQL doesn't support database dropping in the traditional sense
        # This is a no-op for now
        pass

    def list_tables(
        self,
        *,
        like: str | None = None,
        database: str | None = None,
        catalog: str | None = None,
        temp: bool = False,
    ) -> list[str]:
        """Return the list of table/view names.

        Parameters
        ----------
        like
            A pattern in Python's regex format.
        temp
            Whether to list temporary tables or permanent tables.
        database
            The database to list tables of, if not the current one.
        catalog
            The catalog to list tables of, if not the current one.

        Returns
        -------
        list[str]
            The list of the table/view names that match the pattern `like`.
        """
        if temp:
            tables = list(self._temp_tables.keys())
        else:
            # Beam SQL doesn't have a registry of permanent tables
            # This would need to be implemented based on your specific use case
            tables = []
        
        return self._filter_with_like(tables, like)

    def list_views(
        self,
        like: str | None = None,
        temp: bool = False,
    ) -> list[str]:
        """Return the list of view names.

        Parameters
        ----------
        like
            A pattern in Python's regex format.
        temp
            Whether to list temporary views or permanent views.

        Returns
        -------
        list[str]
            The list of the view names that match the pattern `like`.
        """
        # Beam SQL doesn't distinguish between tables and views in the same way
        # as traditional databases
        return self.list_tables(like=like, temp=temp)

    def table(
        self,
        name: str,
        /,
        *,
        database: str | None = None,
        catalog: str | None = None,
    ) -> ir.Table:
        """Return a table expression from a table or view in the database.

        Parameters
        ----------
        name
            Table name.
        database
            Database in which the table resides.
        catalog
            Catalog in which the table resides.

        Returns
        -------
        Table
            Table named `name` from `database`
        """
        if database is not None and not isinstance(database, str):
            raise exc.IbisTypeError(
                f"`database` must be a string; got {type(database)}"
            )
        
        # Check if it's a temporary table
        if name in self._temp_tables:
            schema = self._temp_tables[name]
        else:
            # For permanent tables, we'd need to implement schema discovery
            # This is a placeholder implementation
            schema = sch.Schema({})
        
        node = ops.DatabaseTable(
            name,
            schema=schema,
            source=self,
            namespace=ops.Namespace(catalog=catalog, database=database),
        )
        return node.to_expr()

    def get_schema(
        self,
        table_name: str,
        *,
        catalog: str | None = None,
        database: str | None = None,
    ) -> sch.Schema:
        """Return a Schema object for the indicated table and database.

        Parameters
        ----------
        table_name
            Table name.
        catalog
            Catalog name.
        database
            Database name.

        Returns
        -------
        sch.Schema
            Ibis schema
        """
        # Check if it's a temporary table
        if table_name in self._temp_tables:
            return self._temp_tables[table_name]
        
        # For permanent tables, we'd need to implement schema discovery
        # This is a placeholder implementation
        return sch.Schema({})

    @property
    def version(self) -> str:
        """Get Beam version."""
        import apache_beam as beam
        return beam.__version__

    def _register_udfs(self, expr: ir.Expr) -> None:
        """Register UDFs for the expression."""
        for udf_node in expr.op().find(ops.ScalarUDF):
            register_func = getattr(
                self, f"_register_{udf_node.__input_type__.name.lower()}_udf"
            )
            register_func(udf_node)

    def _register_udf(self, udf_node: ops.ScalarUDF):
        """Register a UDF with Beam."""
        import apache_beam as beam
        from ibis.backends.beam.datatypes import BeamType

        name = type(udf_node).__name__
        
        # Register the UDF with Beam
        # This is a simplified implementation
        beam.udf.register_udf(name, udf_node.__func__)

    _register_pandas_udf = _register_udf
    _register_python_udf = _register_udf

    def compile(
        self,
        expr: ir.Expr,
        /,
        *,
        limit: str | None = None,
        params: Mapping[ir.Expr, Any] | None = None,
        pretty: bool = False,
        **_: Any,
    ) -> str:
        """Compile an Ibis expression to Beam SQL."""
        return super().compile(
            expr, params=params, pretty=pretty
        )  # Discard `limit` and other kwargs

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        """Register an in-memory table."""
        if null_columns := op.schema.null_fields:
            raise exc.IbisTypeError(
                f"{self.name} cannot yet reliably handle `null` typed columns; "
                f"got null typed columns: {null_columns}"
            )
        self.create_view(op.name, op.data.to_frame(), schema=op.schema, temp=True)

    def execute(
        self,
        expr: ir.Expr,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | pd.Series | Any:
        """Execute an Ibis expression and return a pandas `DataFrame`, `Series`, or scalar.

        Parameters
        ----------
        expr
            Ibis expression to execute.
        params
            Mapping of scalar parameter expressions to value.
        limit
            An integer to effect a specific row limit.
        kwargs
            Keyword arguments
        """
        self._run_pre_execute_hooks(expr)

        sql = self.compile(expr.as_table(), params=params, **kwargs)
        
        # Execute the SQL using Beam SQL
        import apache_beam as beam
        from apache_beam.sql import SqlTransform
        
        # Create a transform from the SQL
        sql_transform = SqlTransform(sql)
        
        # Apply the transform and collect results
        result = self._pipeline | sql_transform | beam.Map(lambda row: dict(row))
        
        # Convert to pandas DataFrame
        # This is a simplified implementation
        df = pd.DataFrame(list(result))
        
        return expr.__pandas_result__(df)

    def create_table(
        self,
        name: str,
        /,
        obj: pd.DataFrame | pa.Table | ir.Table | None = None,
        *,
        schema: sch.Schema | None = None,
        database: str | None = None,
        catalog: str | None = None,
        tbl_properties: dict | None = None,
        watermark: Watermark | None = None,
        primary_key: str | list[str] | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ) -> ir.Table:
        """Create a new table in Beam.

        Parameters
        ----------
        name
            Name of the new table.
        obj
            An Ibis table expression, pandas DataFrame, or PyArrow Table.
        schema
            The schema for the new table.
        database
            Name of the database where the table will be created.
        catalog
            Name of the catalog where the table will be created.
        tbl_properties
            Table properties.
        watermark
            Watermark strategy for the table.
        primary_key
            Primary key columns.
        temp
            Whether a table is temporary or not.
        overwrite
            Whether to clobber existing data.

        Returns
        -------
        Table
            The table that was created.
        """
        import pandas as pd
        import pyarrow as pa
        import pyarrow_hotfix  # noqa: F401

        import ibis.expr.types as ir

        if obj is None and schema is None:
            raise exc.IbisError("`schema` or `obj` is required")

        if overwrite:
            if self.list_tables(like=name, temp=temp):
                self.drop_table(
                    name, catalog=catalog, database=database, temp=temp, force=True
                )

        # Handle in-memory data
        if obj is not None:
            if not isinstance(obj, ir.Table):
                obj = ibis.memtable(obj)

            op = obj.op()
            if isinstance(op, ops.InMemoryTable):
                dataframe = op.data.to_frame()
            else:
                raise exc.IbisError(
                    "`obj` is of type ibis.expr.types.Table but it is not in-memory. "
                    "Currently, only in-memory tables are supported."
                )

            return self.create_view(
                name,
                obj=dataframe,
                schema=schema,
                database=database,
                catalog=catalog,
                temp=temp,
                overwrite=overwrite,
            )

        # Handle external data
        else:  # obj is None, schema is not None
            if not tbl_properties:
                raise exc.IbisError(
                    "`tbl_properties` is required when creating table with schema"
                )

            statement = CreateTableWithSchema(
                table_name=name,
                schema=schema,
                tbl_properties=tbl_properties,
                watermark=watermark,
                primary_key=primary_key,
                temporary=temp,
                database=database,
                catalog=catalog,
            )
            sql = statement.compile()
            self.raw_sql(sql)

            return self.table(name, database=database, catalog=catalog)

    def drop_table(
        self,
        name: str,
        /,
        *,
        database: str | None = None,
        catalog: str | None = None,
        temp: bool = False,
        force: bool = False,
    ) -> None:
        """Drop a table.

        Parameters
        ----------
        name
            Name of the table to drop.
        database
            Name of the database where the table exists.
        catalog
            Name of the catalog where the table exists.
        temp
            Whether the table is temporary or not.
        force
            If `False`, an exception is raised if the table does not exist.
        """
        if temp and name in self._temp_tables:
            del self._temp_tables[name]
        else:
            statement = DropTable(
                table_name=name,
                database=database,
                catalog=catalog,
                must_exist=not force,
                temporary=temp,
            )
            sql = statement.compile()
            self.raw_sql(sql)

    def rename_table(
        self,
        old_name: str,
        new_name: str,
        force: bool = True,
    ) -> None:
        """Rename an existing table.

        Parameters
        ----------
        old_name
            The old name of the table.
        new_name
            The new name of the table.
        force
            If `False`, an exception is raised if the table does not exist.
        """
        statement = RenameTable(
            old_name=old_name,
            new_name=new_name,
            must_exist=not force,
        )
        sql = statement.compile()
        self.raw_sql(sql)

    def create_view(
        self,
        name: str,
        /,
        obj: pd.DataFrame | ir.Table,
        *,
        schema: sch.Schema | None = None,
        database: str | None = None,
        catalog: str | None = None,
        force: bool = False,
        temp: bool = False,
        overwrite: bool = False,
    ) -> ir.Table:
        """Create a new view from a dataframe or table.

        Parameters
        ----------
        name
            Name of the new view.
        obj
            An Ibis table expression that will be used to create the view.
        schema
            The schema for the new view.
        database
            Name of the database where the view will be created.
        catalog
            Name of the catalog where the table exists.
        force
            If `False`, an exception is raised if the table is already present.
        temp
            Whether the table is temporary or not.
        overwrite
            If `True`, remove the existing view, and create a new one.

        Returns
        -------
        Table
            The view that was created.
        """
        import pandas as pd

        if overwrite and self.list_views(like=name, temp=temp):
            self.drop_view(
                name, database=database, catalog=catalog, temp=temp, force=True
            )

        if isinstance(obj, pd.DataFrame):
            # Store the schema for temporary tables
            if temp:
                if schema is None:
                    schema = sch.Schema.from_pandas(obj)
                self._temp_tables[name] = schema
        elif isinstance(obj, ir.Table):
            query_expression = self.compile(obj)
            stmt = sge.Create(
                kind="VIEW",
                this=sg.table(
                    name, db=database, catalog=catalog, quoted=self.compiler.quoted
                ),
                expression=query_expression,
                exists=force,
                properties=sge.Properties(expressions=[sge.TemporaryProperty()])
                if temp
                else None,
            )
            self.raw_sql(stmt.sql(self.name))
        else:
            raise exc.IbisError(f"Unsupported `obj` type: {type(obj)}")

        return self.table(name, database=database, catalog=catalog)

    def drop_view(
        self,
        name: str,
        /,
        *,
        database: str | None = None,
        catalog: str | None = None,
        temp: bool = False,
        force: bool = False,
    ) -> None:
        """Drop a view.

        Parameters
        ----------
        name
            Name of the view to drop.
        database
            Name of the database where the view exists.
        catalog
            Name of the catalog where the view exists.
        temp
            Whether the view is temporary or not.
        force
            If `False`, an exception is raised if the view does not exist.
        """
        if temp and name in self._temp_tables:
            del self._temp_tables[name]
        else:
            statement = DropView(
                name=name,
                database=database,
                catalog=catalog,
                must_exist=(not force),
                temporary=temp,
            )
            sql = statement.compile()
            self.raw_sql(sql)

    def _read_file(
        self,
        file_type: str,
        path: str | Path,
        schema: sch.Schema | None = None,
        table_name: str | None = None,
    ) -> ir.Table:
        """Register a file as a table in the current database.

        Parameters
        ----------
        file_type
            File type, e.g., parquet, csv, json.
        path
            The data source.
        schema
            The schema for the new table.
        table_name
            An optional name to use for the created table.

        Returns
        -------
        ir.Table
            The just-registered table
        """
        if schema is None:
            raise ValueError(
                f"`schema` must be explicitly provided when calling `read_{file_type}`"
            )

        table_name = table_name or gen_name(f"read_{file_type}")
        tbl_properties = {
            "connector": "filesystem",
            "path": path,
            "format": file_type,
        }

        return self.create_table(
            table_name, schema=schema, tbl_properties=tbl_properties
        )

    def read_parquet(
        self,
        path: str | Path,
        /,
        *,
        schema: sch.Schema | None = None,
        table_name: str | None = None,
    ) -> ir.Table:
        """Register a parquet file as a table in the current database."""
        return self._read_file(
            file_type="parquet", path=path, schema=schema, table_name=table_name
        )

    def read_csv(
        self,
        path: str | Path,
        /,
        *,
        schema: sch.Schema | None = None,
        table_name: str | None = None,
    ) -> ir.Table:
        """Register a csv file as a table in the current database."""
        return self._read_file(
            file_type="csv", path=path, schema=schema, table_name=table_name
        )

    def read_json(
        self,
        path: str | Path,
        /,
        *,
        schema: sch.Schema | None = None,
        table_name: str | None = None,
    ) -> ir.Table:
        """Register a json file as a table in the current database."""
        return self._read_file(
            file_type="json", path=path, schema=schema, table_name=table_name
        )

    def insert(
        self,
        name: str,
        /,
        obj: pa.Table | pd.DataFrame | ir.Table | list | dict,
        *,
        database: str | None = None,
        catalog: str | None = None,
        overwrite: bool = False,
    ) -> PipelineResult:
        """Insert data into a table.

        Parameters
        ----------
        name
            The name of the table to insert data into.
        obj
            The source data or expression to insert.
        database
            Name of the attached database that the table is located in.
        catalog
            Name of the attached catalog that the table is located in.
        overwrite
            If `True` then replace existing contents of table.

        Returns
        -------
        PipelineResult
            The pipeline result.
        """
        import pandas as pd
        import pyarrow as pa
        import pyarrow_hotfix  # noqa: F401

        if isinstance(obj, ir.Table):
            statement = InsertSelect(
                name,
                self.compile(obj),
                database=database,
                catalog=catalog,
                overwrite=overwrite,
            )
            return self.raw_sql(statement.compile())

        identifier = sg.table(
            name, db=database, catalog=catalog, quoted=self.compiler.quoted
        ).sql(self.dialect)

        if isinstance(obj, pa.Table):
            obj = obj.to_pandas()
        if isinstance(obj, dict):
            obj = pd.DataFrame.from_dict(obj)
        if isinstance(obj, pd.DataFrame):
            # Convert DataFrame to Beam PCollection and insert
            import apache_beam as beam
            
            # Create a PCollection from the DataFrame
            pcoll = self._pipeline | beam.Create(obj.to_dict('records'))
            
            # Apply insert transform
            result = pcoll | beam.io.WriteToText(identifier)
            return result

        if isinstance(obj, list):
            # Convert list to Beam PCollection and insert
            import apache_beam as beam
            
            pcoll = self._pipeline | beam.Create(obj)
            result = pcoll | beam.io.WriteToText(identifier)
            return result

        raise ValueError(
            "No operation is being performed. Either the obj parameter "
            "is not a pandas DataFrame or is not a ibis Table."
            f"The given obj is of type {type(obj).__name__} ."
        )

    def to_pyarrow(
        self,
        expr: ir.Expr,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pa.Table:
        """Convert expression result to PyArrow Table."""
        import pyarrow as pa
        import pyarrow_hotfix  # noqa: F401

        pyarrow_batches = iter(
            self.to_pyarrow_batches(expr, params=params, limit=limit, **kwargs)
        )

        first_batch = next(pyarrow_batches, None)

        if first_batch is None:
            pa_table = expr.as_table().schema().to_pyarrow().empty_table()
        else:
            pa_table = pa.Table.from_batches(
                itertools.chain((first_batch,), pyarrow_batches)
            )
        return expr.__pyarrow_result__(pa_table)

    def to_pyarrow_batches(
        self,
        expr: ir.Table,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        chunk_size: int | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ):
        """Convert expression result to PyArrow batches."""
        import pyarrow as pa
        import pyarrow_hotfix  # noqa: F401

        ibis_table = expr.as_table()

        # Execute the expression and convert to PyArrow
        df = self.execute(ibis_table, limit=limit, **kwargs)
        if limit:
            df = df.head(limit)

        ibis_schema = ibis_table.schema()
        arrow_schema = ibis_schema.to_pyarrow()
        arrow_table = pa.Table.from_pandas(df, schema=arrow_schema)
        return arrow_table.to_reader()
