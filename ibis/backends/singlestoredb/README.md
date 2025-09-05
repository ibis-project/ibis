# SingleStoreDB Backend for Ibis

This backend provides Ibis support for [SingleStoreDB](https://www.singlestore.com/),
a high-performance distributed SQL database designed for data-intensive applications.

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

### Additional Connection Options

SingleStoreDB supports additional connection parameters that can be passed as keyword arguments:

```python
con = ibis.singlestoredb.connect(
    host="localhost",
    user="root",
    password="password",
    database="my_db",
    # Additional options
    autocommit=False,
)
```

### Creating Client from Existing Connection

You can create an Ibis client from an existing SingleStoreDB connection:

```python
import singlestoredb as s2
import ibis

# Create connection using SingleStoreDB client directly
con = s2.connect(
    host="localhost",
    user="root",
    password="password",
    database="my_database"
)

# Create Ibis client from existing connection
ibis_con = ibis.singlestoredb.from_connection(con)
```

### Backend Properties and Methods

The SingleStoreDB backend provides additional properties and methods for advanced usage:

```python
# Get server version
print(ibis_con.version)

# Access SingleStoreDB-specific properties
print(ibis_con.show)             # Access to SHOW commands
print(ibis_con.globals)          # Global variables
print(ibis_con.locals)           # Local variables  
print(ibis_con.cluster_globals)  # Cluster global variables
print(ibis_con.cluster_locals)   # Cluster local variables
print(ibis_con.vars)             # Variables accessor
print(ibis_con.cluster_vars)     # Cluster variables accessor

# Rename a table
ibis_con.rename_table("old_table_name", "new_table_name")

# Execute raw SQL and get cursor
cursor = ibis_con.raw_sql("SHOW TABLES")
tables = [row[0] for row in cursor.fetchall()]
cursor.close()

# Or use context manager
with ibis_con.raw_sql("SELECT COUNT(*) FROM users") as cursor:
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

### Vector Types
- `VECTOR` - for vector data with element types of `F32` (float32), `F64` (float64),
  `I8` (int8), `I16` (int16), `I32` (int32), `I64` (int64).

Note that `VECTOR` types may be represented as binary or JSON dependeng on the
`vector_type_project_format` SingleStoreDB setting.

## Usage Examples

### Basic Query Operations

```python
import ibis

# Connect to SingleStoreDB
ibis_con = ibis.singlestoredb.connect(
    host="localhost",
    user="root",
    password="password",
    database="sample_db"
)

# Create a table reference
table = ibis_con.table('sales_data')

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
json_table = ibis_con.table('products')

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

tbl = ibis_con.create_table('new_products', schema=schema)

# Create table from query
expensive_products = tbl.filter(tbl.price > 100)

expensive_tbl = ibis_con.create_table('expensive_products', expensive_products)
```

### Database Management

```python
# Create and drop databases
ibis_con.create_database("new_database")
ibis_con.create_database("temp_db", force=True)  # CREATE DATABASE IF NOT EXISTS

# List all databases
databases = ibis_con.list_databases()
print(databases)

# Get current database
current_db = ibis_con.current_database
print(f"Connected to: {current_db}")

# Drop database
ibis_con.drop_database("temp_db")
ibis_con.drop_database("old_db", force=True)  # DROP DATABASE IF EXISTS
```

### Table Operations

```python
# List tables in current database
tables = ibis_con.list_tables()

# List tables in specific database  
other_tables = ibis_con.list_tables(database="other_db")

# List tables matching pattern
user_tables = ibis_con.list_tables(like="user_%")

# Get table schema
schema = ibis_con.get_schema("users")
print(schema)

# Drop table
ibis_con.drop_table("old_table")
ibis_con.drop_table("temp_table", force=True)  # DROP TABLE IF EXISTS
```

### Working with Temporary Tables

```python
import pandas as pd

# Create temporary table
temp_data = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
temp_table = ibis_con.create_table("temp_analysis", temp_data, temp=True)

# Use temporary table in queries
result = temp_table.aggregate(total=temp_table.value.sum())

# Temporary tables are automatically dropped when connection closes
```

### Raw SQL Execution

```python
# Execute raw SQL with cursor management
with ibis_con.raw_sql("SHOW PROCESSLIST") as cursor:
    processes = cursor.fetchall()
    for proc in processes:
        print(f"Process {proc[0]}: {proc[7]}")

# Insert data with raw SQL
with ibis_con.begin() as cursor:
    cursor.execute(
        "INSERT INTO users (name, email) VALUES (%s, %s)",
        ("John Doe", "john@example.com")
    )

# Batch operations
with ibis_con.begin() as cursor:
    data = [("Alice", "alice@example.com"), ("Bob", "bob@example.com")]
    cursor.executemany("INSERT INTO users (name, email) VALUES (%s, %s)", data)
```


### SingleStoreDB Resources
- [SingleStoreDB Official Documentation](https://docs.singlestore.com/)
- [SingleStoreDB Python SDK Documentation](https://singlestoredb-python.labs.singlestore.com/)
- [SingleStoreDB Docker Images](https://github.com/singlestore-labs/singlestore-dev-image)
- [SingleStoreDB SQL Reference](https://docs.singlestore.com/managed-service/en/reference/sql-reference.html)


### Community and Support
- [SingleStoreDB Community Forum](https://www.singlestore.com/forum/)
- [Ibis Community Discussions](https://github.com/ibis-project/ibis/discussions)
