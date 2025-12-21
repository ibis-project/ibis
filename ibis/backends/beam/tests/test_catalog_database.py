"""Tests for catalog and database functionality in Beam backend."""

import pytest

import ibis
from ibis.backends.beam import Backend


def test_create_catalog_sql():
    """Test creating catalogs using SQL CREATE CATALOG statements."""
    import apache_beam as beam
    
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Test basic catalog creation
    result = backend.raw_sql("CREATE CATALOG my_catalog")
    assert result is not None
    
    # Test catalog creation with properties
    result = backend.raw_sql("CREATE CATALOG iceberg_catalog WITH (type = 'iceberg', warehouse = 's3://my-bucket/warehouse')")
    assert result is not None
    
    # Test catalog creation with multiple properties
    result = backend.raw_sql("CREATE CATALOG hive_catalog WITH (type = 'hive', metastore_uri = 'thrift://localhost:9083', warehouse = 'hdfs://localhost:9000/warehouse')")
    assert result is not None


def test_create_database_sql():
    """Test creating databases using SQL CREATE DATABASE statements."""
    import apache_beam as beam
    
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Test basic database creation
    result = backend.raw_sql("CREATE DATABASE my_database")
    assert result is not None
    
    # Test database creation in specific catalog
    result = backend.raw_sql("CREATE DATABASE my_database IN CATALOG my_catalog")
    assert result is not None
    
    # Test database creation with properties
    result = backend.raw_sql("CREATE DATABASE analytics_db WITH (location = 's3://my-bucket/analytics')")
    assert result is not None


def test_catalog_management():
    """Test catalog management operations."""
    import apache_beam as beam
    
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Create catalogs
    backend.create_catalog("iceberg_catalog", catalog_type="iceberg", properties={"warehouse": "s3://bucket/warehouse"})
    backend.create_catalog("hive_catalog", catalog_type="hive", properties={"metastore_uri": "thrift://localhost:9083"})
    
    # List catalogs
    catalogs = backend.list_catalogs()
    assert "iceberg_catalog" in catalogs
    assert "hive_catalog" in catalogs
    
    # Drop catalog
    backend.drop_catalog("hive_catalog")
    catalogs = backend.list_catalogs()
    assert "hive_catalog" not in catalogs
    assert "iceberg_catalog" in catalogs


def test_database_management():
    """Test database management operations."""
    import apache_beam as beam
    
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Create databases
    backend.create_database("analytics_db", db_properties={"location": "s3://bucket/analytics"})
    backend.create_database("staging_db", catalog="my_catalog")
    
    # List databases
    databases = backend.list_databases()
    assert "analytics_db" in databases
    assert "staging_db" in databases
    
    # Drop database
    backend.drop_database("staging_db")
    databases = backend.list_databases()
    assert "staging_db" not in databases
    assert "analytics_db" in databases


def test_set_catalog_database():
    """Test setting current catalog and database using SET statements."""
    import apache_beam as beam
    
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Set current catalog
    backend.raw_sql("SET catalog = 'my_iceberg_catalog'")
    assert backend.current_catalog == "my_iceberg_catalog"
    
    # Set current database
    backend.raw_sql("SET database = 'analytics'")
    assert backend.current_database == "analytics"
    
    # Set catalog-specific options
    backend.raw_sql("SET catalog.warehouse = 's3://my-bucket/warehouse'")
    backend.raw_sql("SET catalog.type = 'iceberg'")


def test_iceberg_catalog_setup():
    """Test complete Iceberg catalog setup."""
    import apache_beam as beam
    
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Create Iceberg catalog
    backend.raw_sql("CREATE CATALOG iceberg_catalog WITH (type = 'iceberg', warehouse = 's3://my-bucket/warehouse')")
    
    # Set as current catalog
    backend.raw_sql("SET catalog = 'iceberg_catalog'")
    
    # Create database in the catalog
    backend.raw_sql("CREATE DATABASE analytics IN CATALOG iceberg_catalog")
    
    # Set as current database
    backend.raw_sql("SET database = 'analytics'")
    
    # Verify configuration
    assert backend.current_catalog == "iceberg_catalog"
    assert backend.current_database == "analytics"


def test_hive_catalog_setup():
    """Test complete Hive catalog setup."""
    import apache_beam as beam
    
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Create Hive catalog
    backend.raw_sql("CREATE CATALOG hive_catalog WITH (type = 'hive', metastore_uri = 'thrift://localhost:9083', warehouse = 'hdfs://localhost:9000/warehouse')")
    
    # Set as current catalog
    backend.raw_sql("SET catalog = 'hive_catalog'")
    
    # Create database in the catalog
    backend.raw_sql("CREATE DATABASE staging IN CATALOG hive_catalog")
    
    # Set as current database
    backend.raw_sql("SET database = 'staging'")
    
    # Verify configuration
    assert backend.current_catalog == "hive_catalog"
    assert backend.current_database == "staging"


def test_catalog_with_dataflow():
    """Test catalog setup with DataflowRunner configuration."""
    import apache_beam as beam
    
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Configure DataflowRunner
    backend.raw_sql("SET runner = 'dataflow'")
    backend.raw_sql("SET dataflow.project = 'my-gcp-project'")
    backend.raw_sql("SET dataflow.region = 'us-central1'")
    backend.raw_sql("SET dataflow.staging_location = 'gs://my-bucket/staging'")
    backend.raw_sql("SET dataflow.temp_location = 'gs://my-bucket/temp'")
    
    # Create Iceberg catalog
    backend.raw_sql("CREATE CATALOG iceberg_catalog WITH (type = 'iceberg', warehouse = 'gs://my-bucket/warehouse')")
    
    # Set catalog and database
    backend.raw_sql("SET catalog = 'iceberg_catalog'")
    backend.raw_sql("SET database = 'analytics'")
    
    # Create database
    backend.raw_sql("CREATE DATABASE analytics IN CATALOG iceberg_catalog")
    
    # Verify configuration
    runner_config = backend.get_runner_config()
    assert runner_config['runner'] == 'DataflowRunner'
    assert runner_config['project'] == 'my-gcp-project'
    assert backend.current_catalog == "iceberg_catalog"
    assert backend.current_database == "analytics"


def test_catalog_properties_parsing():
    """Test parsing of catalog properties from SQL."""
    import apache_beam as beam
    
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Test complex properties
    result = backend.raw_sql("CREATE CATALOG complex_catalog WITH (type = 'iceberg', warehouse = 's3://bucket/warehouse', s3_endpoint = 'https://s3.amazonaws.com', s3_access_key = 'AKIAIOSFODNN7EXAMPLE', s3_secret_key = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY')")
    assert result is not None
    
    # Test with spaces in values
    result = backend.raw_sql("CREATE CATALOG spaced_catalog WITH (type = 'hive', warehouse = 'hdfs://localhost:9000/warehouse', description = 'My Hive Catalog')")
    assert result is not None


def test_database_properties_parsing():
    """Test parsing of database properties from SQL."""
    import apache_beam as beam
    
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Test database with properties
    result = backend.raw_sql("CREATE DATABASE props_db WITH (location = 's3://bucket/databases/props_db', comment = 'Database with properties')")
    assert result is not None
    
    # Test database in catalog with properties
    result = backend.raw_sql("CREATE DATABASE catalog_db IN CATALOG my_catalog WITH (location = 's3://bucket/catalog_db')")
    assert result is not None


def test_invalid_sql_statements():
    """Test error handling for invalid SQL statements."""
    import apache_beam as beam
    
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Test invalid CREATE CATALOG
    with pytest.raises(Exception):
        backend.raw_sql("CREATE CATALOG")
    
    # Test invalid CREATE DATABASE
    with pytest.raises(Exception):
        backend.raw_sql("CREATE DATABASE")
    
    # Test malformed properties
    with pytest.raises(Exception):
        backend.raw_sql("CREATE CATALOG bad_catalog WITH (type = iceberg)")  # Missing quotes


def test_catalog_database_integration():
    """Test integration between catalog and database operations."""
    import apache_beam as beam
    
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Create multiple catalogs
    backend.raw_sql("CREATE CATALOG prod_catalog WITH (type = 'iceberg', warehouse = 's3://prod-bucket/warehouse')")
    backend.raw_sql("CREATE CATALOG dev_catalog WITH (type = 'iceberg', warehouse = 's3://dev-bucket/warehouse')")
    
    # Create databases in different catalogs
    backend.raw_sql("CREATE DATABASE analytics IN CATALOG prod_catalog")
    backend.raw_sql("CREATE DATABASE staging IN CATALOG prod_catalog")
    backend.raw_sql("CREATE DATABASE test IN CATALOG dev_catalog")
    
    # List all catalogs and databases
    catalogs = backend.list_catalogs()
    databases = backend.list_databases()
    
    assert "prod_catalog" in catalogs
    assert "dev_catalog" in catalogs
    assert "analytics" in databases
    assert "staging" in databases
    assert "test" in databases
    
    # Switch between catalogs and databases
    backend.raw_sql("SET catalog = 'dev_catalog'")
    backend.raw_sql("SET database = 'test'")
    
    assert backend.current_catalog == "dev_catalog"
    assert backend.current_database == "test"
    
    # Switch back to production
    backend.raw_sql("SET catalog = 'prod_catalog'")
    backend.raw_sql("SET database = 'analytics'")
    
    assert backend.current_catalog == "prod_catalog"
    assert backend.current_database == "analytics"
