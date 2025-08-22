# SingleStoreDB Backend for Ibis

This backend provides Ibis support for [SingleStoreDB](https://www.singlestore.com/), a high-performance distributed SQL database designed for data-intensive applications.

## Installation

The SingleStoreDB backend requires the `singlestoredb` Python package. Install it using:

```bash
pip install 'ibis-framework[singlestoredb]'
```

Or install the SingleStoreDB client directly:

```bash
pip install singlestoredb
```

## Connection Parameters

### Basic Connection

Connect to SingleStoreDB using individual parameters:

```python
import ibis

con = ibis.singlestoredb.connect(
    host="localhost",
    port=3306,
    user="root",
    password="password",
    database="my_database"
)
```

### Connection String

Connect using a connection string:

```python
import ibis

# Basic connection string
con = ibis.connect("singlestoredb://user:password@host:port/database")

# With additional parameters
con = ibis.connect("singlestoredb://user:password@host:port/database?autocommit=true")
```

### Connection Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | `str` | `"localhost"` | SingleStoreDB host address |
| `port` | `int` | `3306` | Port number (usually 3306) |
| `user` | `str` | `"root"` | Username for authentication |
| `password` | `str` | `""` | Password for authentication |
| `database` | `str` | `""` | Database name to connect to |
| `autocommit` | `bool` | `True` | Enable/disable autocommit mode |
| `local_infile` | `int` | `0` | Enable/disable LOCAL INFILE capability |

### Additional Connection Options

SingleStoreDB supports additional connection parameters that can be passed as keyword arguments:

```python
con = ibis.singlestoredb.connect(
    host="localhost",
    user="root",
    password="password",
    database="my_db",
    # Additional options
    charset='utf8mb4',
    ssl_disabled=True,
    connect_timeout=30,
    read_timeout=30,
    write_timeout=30,
)
```

## Supported Data Types

The SingleStoreDB backend supports the following data types:

### Numeric Types
- `TINYINT`, `SMALLINT`, `MEDIUMINT`, `INT`, `BIGINT`
- `FLOAT`, `DOUBLE`, `DECIMAL`
- `BOOLEAN` (alias for `TINYINT(1)`)

### String Types
- `CHAR`, `VARCHAR`
- `TEXT`, `MEDIUMTEXT`, `LONGTEXT`
- `BINARY`, `VARBINARY`
- `BLOB`, `MEDIUMBLOB`, `LONGBLOB`

### Date/Time Types
- `DATE`
- `TIME`
- `DATETIME`
- `TIMESTAMP`
- `YEAR`

### Special SingleStoreDB Types
- `JSON` - for storing JSON documents
- `VECTOR` - for vector data (AI/ML workloads)
- `GEOGRAPHY` - for geospatial data

## Supported Operations

### Core SQL Operations
- ✅ SELECT queries with WHERE, ORDER BY, LIMIT
- ✅ INSERT, UPDATE, DELETE operations
- ✅ CREATE/DROP TABLE operations
- ✅ CREATE/DROP DATABASE operations

### Aggregations
- ✅ Basic aggregations: COUNT, SUM, AVG, MIN, MAX
- ✅ GROUP BY operations
- ✅ HAVING clauses
- ✅ Window functions: ROW_NUMBER, RANK, DENSE_RANK, etc.

### Joins
- ✅ INNER JOIN
- ✅ LEFT JOIN, RIGHT JOIN
- ✅ FULL OUTER JOIN
- ✅ CROSS JOIN

### Set Operations
- ✅ UNION, UNION ALL
- ✅ INTERSECT
- ✅ EXCEPT

### String Operations
- ✅ String functions: LENGTH, SUBSTRING, CONCAT, etc.
- ✅ Pattern matching with LIKE
- ✅ Regular expressions with REGEXP

### Mathematical Operations
- ✅ Arithmetic operators (+, -, *, /, %)
- ✅ Mathematical functions: ABS, ROUND, CEIL, FLOOR, etc.
- ✅ Trigonometric functions

### Date/Time Operations
- ✅ Date extraction: YEAR, MONTH, DAY, etc.
- ✅ Date arithmetic
- ✅ Date formatting functions

## Usage Examples

### Basic Query Operations

```python
import ibis

# Connect to SingleStoreDB
con = ibis.singlestoredb.connect(
    host="localhost",
    user="root",
    password="password",
    database="sample_db"
)

# Create a table reference
table = con.table('sales_data')

# Simple select
result = table.select(['product_id', 'revenue']).execute()

# Filtering
high_revenue = table.filter(table.revenue > 1000)

# Aggregation
revenue_by_product = (
    table
    .group_by('product_id')
    .aggregate(total_revenue=table.revenue.sum())
)

# Window functions  
ranked_sales = table.mutate(
    rank=table.revenue.rank().over(ibis.window(order_by=table.revenue.desc()))
)
```

### Working with JSON Data

```python
# Assuming a table with a JSON column 'metadata'
json_table = con.table('products')

# Extract JSON fields
extracted = json_table.mutate(
    category=json_table.metadata['category'].cast('string'),
    price=json_table.metadata['price'].cast('double')
)
```

### Creating Tables

```python
import ibis

# Create a new table
schema = ibis.schema([
    ('id', 'int64'),
    ('name', 'string'),
    ('price', 'float64'),
    ('created_at', 'timestamp')
])

con.create_table('new_products', schema=schema)

# Create table from query
expensive_products = existing_table.filter(existing_table.price > 100)
con.create_table('expensive_products', expensive_products)
```

## Known Limitations

### Unsupported Operations
- ❌ Some advanced window functions may not be available
- ❌ Certain JSON functions may have different syntax
- ❌ Some MySQL-specific functions may not be supported

### Performance Considerations
- SingleStoreDB is optimized for distributed queries; single-node operations may have different performance characteristics
- VECTOR and GEOGRAPHY types require specific SingleStoreDB versions
- Large result sets should use appropriate LIMIT clauses

### Transaction Behavior
- SingleStoreDB uses distributed transactions which may have different semantics than traditional RDBMS
- Some isolation levels may not be available

## Troubleshooting

### Connection Issues

**Problem**: `Can't connect to SingleStoreDB server`
```
Solution: Verify host, port, and credentials. Check if SingleStoreDB is running:
mysql -h <host> -P <port> -u <user> -p
```

**Problem**: `Unknown database 'database_name'`
```
Solution: Create the database first or use an existing database:
con.create_database('database_name')
```

**Problem**: `Access denied for user`
```
Solution: Check user permissions:
GRANT ALL PRIVILEGES ON database_name.* TO 'user'@'%';
```

### Data Type Issues

**Problem**: `Out of range value for column`
```
Solution: Check data types and ranges. SingleStoreDB may be stricter than MySQL:
- Use appropriate data types for your data
- Handle NULL values explicitly in data loading
```

**Problem**: `JSON column issues`
```
Solution: Ensure proper JSON syntax and use JSON functions correctly:
table.json_col['key'].cast('string')  # Extract and cast JSON values
```

### Performance Issues

**Problem**: `Slow query performance`
```
Solution:
- Use appropriate indexes
- Consider columnstore vs rowstore table types
- Use EXPLAIN to analyze query plans
- Leverage SingleStoreDB's distributed architecture
```

**Problem**: `Memory issues with large datasets`
```
Solution:
- Use streaming operations with .execute(limit=n)  
- Consider chunked processing for large data imports
- Monitor SingleStoreDB cluster capacity
```

### Docker/Development Issues

**Problem**: `SingleStoreDB container health check failing`
```
Solution:
- Check container logs: docker logs <container_name>
- Verify initialization scripts ran successfully
- Check license capacity warnings (these don't affect functionality)
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install -e '.[test,singlestoredb]'

# Start SingleStoreDB container
just up singlestoredb

# Run SingleStoreDB-specific tests
pytest -m singlestoredb

# Run with specific test data
IBIS_TEST_SINGLESTOREDB_PORT=3307 \
IBIS_TEST_SINGLESTOREDB_PASSWORD="ibis_testing" \
IBIS_TEST_SINGLESTOREDB_DATABASE="ibis_testing" \
pytest -m singlestoredb
```

### Contributing

When contributing to the SingleStoreDB backend:

1. Follow the existing code patterns from other SQL backends
2. Add tests for new functionality
3. Update documentation for new features
4. Ensure compatibility with SingleStoreDB's MySQL protocol
5. Test with both rowstore and columnstore table types when relevant

## Resources

- [SingleStoreDB Documentation](https://docs.singlestore.com/)
- [SingleStoreDB Python Client](https://pypi.org/project/singlestoredb/)
- [SingleStoreDB Python SDK Documentation](https://singlestoredb-python.labs.singlestore.com/)
- [Ibis Documentation](https://ibis-project.org/)
- [MySQL Protocol Reference](https://dev.mysql.com/doc/internals/en/client-server-protocol.html)
