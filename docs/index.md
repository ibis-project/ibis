<figure markdown> 
  ![Image title](/static/img/ibis_sky.png){ width="300" }
  <figcaption>Write your analytics code once, run it everywhere.</figcaption>
</figure>

## Features

Ibis provides a standard way to write analytics code, that then can be run in
multiple engines.

- **Full coverage of SQL features**: Anything you can write in a `SELECT` statement you can wrtite in Ibis
- **Abstract over SQL differences**: Write standard code that translates to any SQL syntax
- **High performance execution**: Execute at the speed of your backend, not your local computer
- **Integration with community infrastructure**: Ibis works with existing Python tools

## Supported Backends

- Traditional DBMSs: [PostgreSQL](/backends/postgres), [MySQL](/backends/mysql), [SQLite](/backends/sqlite)
- Analytical DBMSs: [OmniSciDB](/backends/omnisci), [ClickHouse](/backends/clickhouse), [Datafusion](/backends/datafusion)
- Distributed DBMSs: [Impala](/backends/impala), [PySpark](/backends/pyspark), [BigQuery](/backends/bigquery)
- In memory analytics: [pandas](/backends/pandas), [Dask](/backends/dask)

## Example

Here's Ibis computing the number of citizens per squared kilometer in Asia:

```python
>>> import ibis
>>> db = ibis.sqlite.connect('geography.db')
>>> countries = db.table('countries')
>>> asian_countries = countries.filter(countries['continent'] == 'AS')
>>> density_in_asia = asian_countries['population'].sum() / asian_countries['area_km2'].sum()
>>> density_in_asia.execute()
130.7019141926602
```

!!! tip "Learn more!"

    Learn more about Ibis in [our tutorial](/tutorial/01-Introduction-to-Ibis).

## Comparison to other tools

=== "SQL"

    ??? tip "Coming from SQL?"

        Check out [Ibis for SQL Programmers](/user_guide/sql)!

    Ibis gives you the benefit of a programming language. You don't need to
    sacrifice maintainability to get to those insights!

    === "SQL"

        ``` sql title="docs/example.sql" linenums="1"
        --8<-- "docs/example.sql"
        ```

    === "Ibis"

        ``` py title="docs/example.py" linenums="1"
        --8<-- "docs/example.py"
        ```

=== "SQLAlchemy"

    Ibis aims to be more concise and composable than
    [SQLAlchemy](https://www.sqlalchemy.org/) when writing interactive
    analytics code.

    !!! success "Ibis :heart:'s SQLAlchemy"

        Ibis generates SQLAlchemy expressions for some of our backends
        including the [PostgreSQL](/backends/postgres) and
        [SQLite](/backends/sqlite) backends!

    === "SQLAlchemy"

        ``` python title="docs/sqlalchemy_example.py" "linenums="1"
        --8<-- "docs/sqlalchemy_example.py"
        ```

    === "Ibis"

        ``` py title="docs/example.py" linenums="1"
        --8<-- "docs/example.py"
        ```
