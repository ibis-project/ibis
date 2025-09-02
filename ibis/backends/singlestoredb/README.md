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
con = ibis.connect("singlestoredb://user:password@host:port/database?autocommit=true&local_infile=1")

# URL with special characters (use URL encoding)
from urllib.parse import quote_plus
password = "p@ssw0rd!"
encoded_password = quote_plus(password)
con = ibis.connect(f"singlestoredb://user:{encoded_password}@host:port/database")
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

### Creating Client from Existing Connection

You can create an Ibis client from an existing SingleStoreDB connection:

```python
import singlestoredb
import ibis

# Create connection using SingleStoreDB client directly
singlestore_con = singlestoredb.connect(
    host="localhost",
    user="root",
    password="password",
    database="my_database"
)

# Create Ibis client from existing connection
con = ibis.singlestoredb.from_connection(singlestore_con)
```

### Backend Properties and Methods

The SingleStoreDB backend provides additional properties and methods for advanced usage:

```python
# Get server version
print(con.version)

# Access SingleStoreDB-specific properties
print(con.show)           # Access to SHOW commands
print(con.globals)        # Global variables
print(con.locals)         # Local variables  
print(con.cluster_globals) # Cluster global variables
print(con.cluster_locals)  # Cluster local variables
print(con.vars)           # Variables accessor
print(con.cluster_vars)   # Cluster variables accessor

# Rename a table
con.rename_table("old_table_name", "new_table_name")

# Execute raw SQL and get cursor
cursor = con.raw_sql("SHOW TABLES")
tables = [row[0] for row in cursor.fetchall()]
cursor.close()

# Or use context manager
with con.raw_sql("SELECT COUNT(*) FROM users") as cursor:
    count = cursor.fetchone()[0]
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
- `JSON` - for storing JSON documents (with special handling for proper conversion)
- `GEOMETRY` - for geospatial data using MySQL-compatible spatial types
- `BLOB`, `MEDIUMBLOB`, `LONGBLOB` - for binary data storage

### Data Type Limitations
- **Complex Types**: Arrays, structs, and maps are **not supported** and will raise `UnsupportedBackendType` errors
- **Boolean Values**: Stored as `TINYINT(1)` and automatically converted by Ibis
- **Vector Types**: While SingleStoreDB supports VECTOR types, they are not currently mapped in the Ibis type system

## Technical Details

### SQL Dialect and Compilation
- **SQLGlot Dialect**: Uses `"singlestore"` dialect for SQL compilation  
- **Character Encoding**: UTF8MB4 (4-byte Unicode support)
- **Autocommit**: Enabled by default (`autocommit=True`)
- **Temporary Tables**: Fully supported for intermediate operations

### Type System Integration
- **Boolean Handling**: `TINYINT(1)` columns automatically converted to boolean
- **JSON Processing**: Special conversion handling for proper PyArrow compatibility
- **Decimal Precision**: Supports high-precision decimal arithmetic
- **Null Handling**: Proper NULL vs JSON null distinction

### Query Optimization Features
- **Shard Key Hints**: Compiler can add shard key hints for distributed queries
- **Columnstore Optimization**: Query patterns optimized for columnstore tables
- **Row Ordering**: Non-deterministic by default, use `ORDER BY` for consistent results

### Connection Management
- **Connection Pooling**: Uses SingleStoreDB Python client's connection handling
- **Transaction Support**: Full ACID transaction support with distributed consistency
- **Reconnection**: Automatic reconnection handling with parameter preservation

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

### SingleStoreDB-Specific Operations
- ✅ `FIND_IN_SET()` for searching in comma-separated lists
- ✅ `XOR` logical operator
- ✅ `RowID` support via `ROW_NUMBER()` implementation
- ✅ Advanced regex operations with POSIX compatibility

### Unsupported Operations
The following operations are **not supported** in the SingleStoreDB backend:

#### Hash and Digest Functions
- ❌ `HexDigest` - Hash digest functions not available
- ❌ `Hash` - Generic hash functions not supported

#### Aggregate Functions  
- ❌ `First` - First aggregate function not supported
- ❌ `Last` - Last aggregate function not supported
- ❌ `CumeDist` - Cumulative distribution window function not available

#### Array Operations
- ❌ `ArrayStringJoin` - No native array-to-string conversion (arrays not supported)
- ❌ All other array operations (arrays, structs, maps not supported)

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

### Database Management

```python
# Create and drop databases
con.create_database("new_database")
con.create_database("temp_db", force=True)  # CREATE DATABASE IF NOT EXISTS

# List all databases
databases = con.list_databases()
print(databases)

# Get current database
current_db = con.current_database
print(f"Connected to: {current_db}")

# Drop database
con.drop_database("temp_db")
con.drop_database("old_db", force=True)  # DROP DATABASE IF EXISTS
```

### Table Operations

```python
# List tables in current database
tables = con.list_tables()

# List tables in specific database  
other_tables = con.list_tables(database="other_db")

# List tables matching pattern
user_tables = con.list_tables(like="user_%")

# Get table schema
schema = con.get_schema("users")
print(schema)

# Drop table
con.drop_table("old_table")
con.drop_table("temp_table", force=True)  # DROP TABLE IF EXISTS
```

### Working with Temporary Tables

```python
import pandas as pd

# Create temporary table
temp_data = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
temp_table = con.create_table("temp_analysis", temp_data, temp=True)

# Use temporary table in queries
result = temp_table.aggregate(total=temp_table.value.sum())

# Temporary tables are automatically dropped when connection closes
```

### Raw SQL Execution

```python
# Execute raw SQL with cursor management
with con.raw_sql("SHOW PROCESSLIST") as cursor:
    processes = cursor.fetchall()
    for proc in processes:
        print(f"Process {proc[0]}: {proc[7]}")

# Insert data with raw SQL
with con.begin() as cursor:
    cursor.execute(
        "INSERT INTO users (name, email) VALUES (%s, %s)",
        ("John Doe", "john@example.com")
    )

# Batch operations
with con.begin() as cursor:
    data = [("Alice", "alice@example.com"), ("Bob", "bob@example.com")]
    cursor.executemany("INSERT INTO users (name, email) VALUES (%s, %s)", data)
```

### Advanced SingleStoreDB Features  

```python
# Use SingleStoreDB-specific functions
from ibis import _

# FIND_IN_SET function
table = con.table("products")
matching_products = table.filter(
    _.tags.find_in_set("electronics") > 0
)

# JSON path queries
json_table = con.table("events")
user_events = json_table.filter(
    json_table.data['user']['type'].cast('string') == 'premium'
)

# Geospatial queries (if using GEOMETRY types)
locations = con.table("locations")
nearby = locations.filter(
    locations.coordinates.st_distance_sphere(locations.coordinates) < 1000
)
```

## Known Limitations

### Architectural Limitations
- **No Catalog Support**: SingleStoreDB uses databases only, not catalogs
- **Complex Types**: Arrays, structs, and maps are not supported and will raise errors
- **Row Ordering**: Results may be non-deterministic without explicit `ORDER BY` clauses
- **Multi-byte Character Encoding**: Uses UTF8MB4 exclusively (4-byte Unicode characters)

### Unsupported Operations
Based on the current implementation, these operations are not supported:

#### Hash and Cryptographic Functions
- `HexDigest` - Hash digest functions not available in SingleStoreDB
- `Hash` - Generic hash functions not supported

#### Statistical and Analytical Functions
- `First` / `Last` - First/Last aggregate functions not supported
- `CumeDist` - Cumulative distribution window function not available

#### Array and Complex Data Operations
- `ArrayStringJoin` - No native array-to-string conversion
- All array, struct, and map operations (complex types not supported)

#### Advanced Window Functions
Some advanced window functions may not be available compared to other SQL databases

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

**Problem**: `JSON column issues` or `JSON conversion errors`
```
Solution: SingleStoreDB has special JSON handling requirements:
# Extract and cast JSON values properly
table.json_col['key'].cast('string')

# For PyArrow compatibility, JSON nulls vs SQL NULLs are handled differently
# The backend automatically converts JSON objects to strings for PyArrow

# When creating tables with JSON data, ensure valid JSON format
import json
df['json_col'] = df['json_col'].apply(lambda x: json.dumps(x) if x is not None else None)
```

**Problem**: `UnsupportedBackendType: Arrays/structs/maps not supported`
```
Solution: SingleStoreDB doesn't support complex types:
# Instead of arrays, use JSON arrays
df['array_col'] = df['array_col'].apply(json.dumps)  # Convert list to JSON string

# Instead of structs, use JSON objects  
df['struct_col'] = df['struct_col'].apply(json.dumps)  # Convert dict to JSON string

# Query JSON arrays/objects using JSON path expressions
table.json_col['$.array[0]'].cast('string')  # Access first array element
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
# Check container status and logs
docker ps | grep singlestore
docker logs <container_name>

# Common issues and solutions:
1. Port conflicts: Ensure port 3307 is not in use by another service
2. Memory limits: SingleStoreDB needs adequate memory (2GB+ recommended)  
3. License warnings: These are informational only and don't affect functionality
4. Initialization scripts: Check if /docker-entrypoint-initdb.d/init.sql ran successfully

# Restart container if needed
docker restart <container_name>

# Check if service is responding
mysql -h 127.0.0.1 -P 3307 -u root -p'ibis_testing' -e "SELECT 1"
```

**Problem**: `Connection timeout` or `Can't connect to server`
```
Solution:
# Verify container is running and port is accessible
docker ps | grep singlestore
netstat -tlnp | grep 3307

# Check if using correct connection parameters
- Host: 127.0.0.1 or localhost  
- Port: 3307 (not 3306)
- Database: ibis_testing
- Username: root
- Password: ibis_testing

# Test connection manually
mysql -h 127.0.0.1 -P 3307 -u root -p'ibis_testing' ibis_testing
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install -e '.[test,singlestoredb]'

# Start SingleStoreDB container (uses port 3307 to avoid MySQL conflicts)
just up singlestoredb

# Run SingleStoreDB-specific tests
pytest -m singlestoredb

# Run with explicit test configuration (these are the defaults)
IBIS_TEST_SINGLESTOREDB_HOST="127.0.0.1" \
IBIS_TEST_SINGLESTOREDB_PORT=3307 \
IBIS_TEST_SINGLESTOREDB_USER="root" \
IBIS_TEST_SINGLESTOREDB_PASSWORD="ibis_testing" \
IBIS_TEST_SINGLESTOREDB_DATABASE="ibis_testing" \
pytest -m singlestoredb

# Check container status
docker ps | grep singlestore

# View container logs (ignore capacity warnings - they don't affect functionality)
docker logs <singlestore_container_id>
```

### Test Environment Details

The test environment uses:
- **Docker Image**: `ghcr.io/singlestore-labs/singlestoredb-dev:latest`
- **Host Port**: 3307 (mapped to container port 3306)
- **Database**: `ibis_testing`
- **Username/Password**: `root`/`ibis_testing`
- **Test Configuration**: Found in `ibis/backends/singlestoredb/tests/conftest.py`

### Contributing

When contributing to the SingleStoreDB backend:

1. Follow the existing code patterns from other SQL backends
2. Add tests for new functionality
3. Update documentation for new features
4. Ensure compatibility with SingleStoreDB's MySQL protocol
5. Test with both rowstore and columnstore table types when relevant

## Resources

### SingleStoreDB Resources
- [SingleStoreDB Official Documentation](https://docs.singlestore.com/)
- [SingleStoreDB Python Client PyPI](https://pypi.org/project/singlestoredb/)
- [SingleStoreDB Python SDK Documentation](https://singlestoredb-python.labs.singlestore.com/)
- [SingleStoreDB Docker Images](https://github.com/singlestore-labs/singlestore-dev-image)
- [SingleStoreDB SQL Reference](https://docs.singlestore.com/managed-service/en/reference/sql-reference.html)

### Ibis Integration Resources  
- [Ibis Documentation](https://ibis-project.org/)
- [Ibis SQL Backend Guide](https://ibis-project.org/how-to/backends)
- [Ibis GitHub Repository](https://github.com/ibis-project/ibis)

### Development Resources
- [SQLGlot SingleStore Dialect](https://sqlglot.com/sql.html#singlestore)
- [MySQL Protocol Reference](https://dev.mysql.com/doc/internals/en/client-server-protocol.html) (SingleStoreDB is MySQL-compatible)
- [Docker Compose for Development](https://docs.docker.com/compose/)

### Community and Support
- [SingleStoreDB Community Forum](https://www.singlestore.com/forum/)
- [Ibis Community Discussions](https://github.com/ibis-project/ibis/discussions)
- [SingleStoreDB Discord](https://discord.gg/singlestore)
