# Configuring Ibis

Ibis configuration happens through the `ibis.options` attribute. Attributes can
be get and set like class attributes.

## Interactive mode

Ibis out of the box is in _developer mode_. Expressions display their internal
details when printed to the console. For a better interactive experience, set
the `interactive` option:

```python
ibis.options.interactive = True
```

This will cause expressions to be executed immediately when printed to the
console.

## SQL Query Execution

If an Ibis table expression has no row limit set using the `limit` API, a
default one is applied to prevent too much data from being retrieved from the
query engine. The default is currently 10000 rows, but this can be configured
with the `sql.default_limit` option:

```python
ibis.options.sql.default_limit = 100
```

Set this to `None` to retrieve all rows in all queries

!!! warning "Be careful with `None`"

    Setting the default limit to `None` will result in *all* rows from a query
    coming back to the client from the backend.

```python
ibis.options.sql.default_limit = None
```

## Verbose option and Logging

To see all internal Ibis activity (like queries being executed) set
`ibis.options.verbose`:

```python
ibis.options.verbose = True
```

By default this information is sent to `sys.stdout`, but you can set some
other logging function:

```python
def cowsay(msg):
    print(f"Cow says: {msg}")


ibis.options.verbose_log = cowsay
```

## Default backend

`ibis.options.default_backend` controls which backend is used by table
expressions returned by top-level functions such as `ibis.memtable`,
`ibis.read_csv` or `ibis.read_parquet`.

By default, it points to an instance of DuckDB backend. Assuming the [backend
dependencies have been installed](../install.md), it can be updated by passing
the name of the backend to `ibis.set_backend` as follows:

```python
import ibis

expr = ibis.memtable({"column": [0, 1, 2, 3, 4]})
ibis.get_backend(expr)
# <ibis.backends.duckdb.Backend at 0x12fa0fb50>

ibis.set_backend("sqlite")
ibis.get_backend(expr)
# <ibis.backends.sqlite.Backend at 0x158411d10>
```
