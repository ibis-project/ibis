"""Tests for DB2 backend functionality."""

from __future__ import annotations

import pytest

from ibis.backends import db2


class TestConnection:
    """Tests for connection functionality."""

    def test_connect_with_all_params(self, db2_config):
        """Test connection with all parameters."""
        con = db2.connect(**db2_config)
        assert con is not None
        assert isinstance(con, db2.Backend)
        con.disconnect()

    def test_connect_minimal_params(self):
        """Test connection with minimal parameters."""
        # This will fail without a real DB2 instance
        import ibis.common.exceptions as exc

        with pytest.raises(exc.OperationNotDefinedError):
            db2.connect(database="SAMPLE")

    def test_disconnect(self, db2_config):
        """Test disconnection."""
        con = db2.connect(**db2_config)
        con.disconnect()
        # After disconnect, connection should be None
        assert con._connection is None


@pytest.mark.integration
class TestBackendOperations:
    """Integration tests for backend operations."""

    def test_list_tables(self, con):
        """Test listing tables."""
        tables = con.list_tables()
        assert isinstance(tables, list)

    def test_list_databases(self, con):
        """Test listing databases/schemas."""
        databases = con.list_databases()
        assert isinstance(databases, list)
        assert len(databases) > 0

    def test_current_database(self, con):
        """Test getting current database."""
        current_db = con.current_database
        assert isinstance(current_db, str)
        assert len(current_db) > 0

    def test_version(self, con):
        """Test getting DB2 version."""
        version = con.version
        assert isinstance(version, str)
        assert len(version) > 0


@pytest.mark.integration
class TestTableOperations:
    """Integration tests for table operations."""

    def test_create_table_from_dataframe(self, con, test_table_name, sample_dataframe):
        """Test creating a table from a DataFrame."""
        table = con.create_table(test_table_name, sample_dataframe)
        assert table is not None

        # Verify table exists
        tables = con.list_tables()
        assert test_table_name.upper() in [t.upper() for t in tables]

        # Cleanup
        con.drop_table(test_table_name, force=True)

    def test_create_table_with_schema(self, con, test_table_name):
        """Test creating a table with explicit schema."""
        import ibis.expr.datatypes as dt
        import ibis.expr.schema as sch

        schema = sch.Schema(
            {
                "id": dt.int32,
                "name": dt.string,
                "value": dt.float64,
            }
        )

        table = con.create_table(test_table_name, schema=schema)
        assert table is not None

        # Verify schema
        result_schema = con.get_schema(test_table_name)
        assert "id" in result_schema
        assert "name" in result_schema
        assert "value" in result_schema

        # Cleanup
        con.drop_table(test_table_name, force=True)

    def test_create_temp_table(self, con, test_table_name, sample_dataframe):
        """Test creating a temporary table."""
        table = con.create_table(test_table_name, sample_dataframe, temp=True)
        assert table is not None

        # Cleanup
        con.drop_table(test_table_name, force=True)

    def test_create_table_overwrite(self, con, test_table_name, sample_dataframe):
        """Test creating a table with overwrite."""
        # Create table first time
        con.create_table(test_table_name, sample_dataframe)

        # Create again with overwrite
        table = con.create_table(test_table_name, sample_dataframe, overwrite=True)
        assert table is not None

        # Cleanup
        con.drop_table(test_table_name, force=True)

    def test_drop_table(self, con, test_table_name, sample_dataframe):
        """Test dropping a table."""
        con.create_table(test_table_name, sample_dataframe)
        con.drop_table(test_table_name)

        # Verify table doesn't exist
        tables = con.list_tables()
        assert test_table_name.upper() not in [t.upper() for t in tables]

    def test_drop_table_force(self, con, test_table_name):
        """Test dropping a non-existent table with force."""
        # Should not raise error
        con.drop_table(test_table_name, force=True)

    def test_get_schema(self, con, temp_table):
        """Test getting table schema."""
        schema = con.get_schema(temp_table)
        assert "id" in schema
        assert "name" in schema
        assert "age" in schema
        assert "salary" in schema
        assert "is_active" in schema

    def test_table_expression(self, con, temp_table):
        """Test creating a table expression."""
        table = con.table(temp_table)
        assert table is not None
        assert hasattr(table, "schema")


@pytest.mark.integration
class TestDataOperations:
    """Integration tests for data operations."""

    def test_insert_dataframe(self, con, test_table_name, sample_dataframe):
        """Test inserting data from DataFrame."""
        import ibis.expr.schema as sch

        # Create empty table
        schema = sch.infer(sample_dataframe)
        con.create_table(test_table_name, schema=schema)

        # Insert data
        con.insert(test_table_name, sample_dataframe)

        # Verify data
        table = con.table(test_table_name)
        result = table.execute()
        assert len(result) == len(sample_dataframe)

        # Cleanup
        con.drop_table(test_table_name, force=True)

    def test_insert_overwrite(self, con, temp_table, sample_dataframe):
        """Test inserting with overwrite."""
        # Insert new data with overwrite
        con.insert(temp_table, sample_dataframe, overwrite=True)

        # Verify data
        table = con.table(temp_table)
        result = table.execute()
        assert len(result) == len(sample_dataframe)

    def test_execute_query(self, con, temp_table):
        """Test executing a query."""
        table = con.table(temp_table)
        result = table.filter(table.age > 30).execute()
        assert len(result) > 0
        assert all(result["age"] > 30)

    def test_to_pandas(self, con, temp_table):
        """Test converting to pandas DataFrame."""
        table = con.table(temp_table)
        df = con.to_pandas(table)
        assert df is not None
        assert len(df) > 0

    def test_to_pyarrow(self, con, temp_table):
        """Test converting to PyArrow table."""
        table = con.table(temp_table)
        arrow_table = con.to_pyarrow(table)
        assert arrow_table is not None
        assert len(arrow_table) > 0


@pytest.mark.integration
class TestQueryOperations:
    """Integration tests for query operations."""

    def test_select(self, con, temp_table):
        """Test SELECT operation."""
        table = con.table(temp_table)
        result = table.select("name", "age").execute()
        assert "name" in result.columns
        assert "age" in result.columns
        assert "salary" not in result.columns

    def test_filter(self, con, temp_table):
        """Test WHERE/filter operation."""
        table = con.table(temp_table)
        result = table.filter(table.age >= 35).execute()
        assert all(result["age"] >= 35)

    def test_order_by(self, con, temp_table):
        """Test ORDER BY operation."""
        table = con.table(temp_table)
        result = table.order_by(table.age.desc()).execute()
        ages = result["age"].tolist()
        assert ages == sorted(ages, reverse=True)

    def test_limit(self, con, temp_table):
        """Test LIMIT operation."""
        table = con.table(temp_table)
        result = table.limit(3).execute()
        assert len(result) == 3

    def test_aggregate(self, con, temp_table):
        """Test aggregation operations."""
        table = con.table(temp_table)
        result = table.aggregate(
            avg_age=table.age.mean(),
            max_salary=table.salary.max(),
            count=table.count(),
        ).execute()
        assert "avg_age" in result.columns
        assert "max_salary" in result.columns
        assert "count" in result.columns

    def test_group_by(self, con, temp_table):
        """Test GROUP BY operation."""
        table = con.table(temp_table)
        result = table.group_by("is_active").aggregate(
            count=table.count(),
            avg_age=table.age.mean(),
        ).execute()
        assert len(result) <= 2  # True and False


@pytest.mark.integration
class TestComplexQueries:
    """Integration tests for complex queries."""

    def test_join(self, con, test_table_name, sample_dataframe):
        """Test JOIN operation."""
        # Create two tables
        table1_name = f"{test_table_name}_1"
        table2_name = f"{test_table_name}_2"

        con.create_table(table1_name, sample_dataframe)
        con.create_table(table2_name, sample_dataframe)

        try:
            table1 = con.table(table1_name)
            table2 = con.table(table2_name)

            result = table1.join(table2, table1.id == table2.id).execute()
            assert len(result) > 0
        finally:
            con.drop_table(table1_name, force=True)
            con.drop_table(table2_name, force=True)

    def test_union(self, con, temp_table):
        """Test UNION operation."""
        table = con.table(temp_table)
        result = table.union(table).execute()
        # Union should have twice the rows
        assert len(result) == len(table.execute()) * 2

    def test_subquery(self, con, temp_table):
        """Test subquery."""
        table = con.table(temp_table)
        avg_age = table.age.mean().name("avg_age")
        result = table.filter(table.age > avg_age).execute()
        assert len(result) > 0
