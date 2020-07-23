<div class="row">
    <div class="col">
        <section class="jumbotron text-center home-jumbotron">
            <p>
                Write your analytics code once, run it everywhere.
            </p>
        </section>
    </div>
</div>

## Main features

Ibis provides a standard way to write analytics code, that then can be run in
multiple engines.

- **Full coverage of SQL features**: You can code in Ibis anything you can implement in a SQL SELECT
- **Transparent to SQL implementation differences**: Write standard code that translate to any SQL syntax
- **High performance execution**: Execute at the speed of your backend, not your local computer
- **Integration with community data formats and tools** (e.g. pandas, Parquet, Avro...)

## Supported engines

- Standard DBMS: [PostgreSQL](/docs/backends/postgres.html), [MySQL](/docs/backends/mysql.html), [SQLite](/docs/backends/sqlite.html)
- Analytical DBMS: [OmniSciDB](/docs/backends/omnisci.html), [ClickHouse](/docs/backends/clickhouse.html)
- Distributed platforms: [Impala](/docs/backends/impala.html), [Spark](/docs/backends/spark.html), [BigQuery](/docs/backends/bigquery.html)
- In memory execution: [pandas](/docs/backends/pandas.html)

## Example

The next example is all the code you need to connect to a database with a
countries database, and compute the number of citizens per squared kilometer in Asia:

```python
>>> import ibis
>>> db = ibis.sqlite.connect('geography.db')
>>> countries = db.table('countries')
>>> asian_countries = countries.filter(countries['continent'] == 'AS')
>>> density_in_asia = asian_countries['population'].sum() / asian_countries['area_km2'].sum()
>>> density_in_asia.execute()
130.7019141926602
```

Learn more about Ibis in [our tutorial](/docs/tutorial/).

## Comparison to other tools

### Why not use [pandas](https://pandas.pydata.org/)?

pandas is great for many use cases. But pandas loads the data into the
memory of the local host, and performs the computations on it.

Ibis instead, leaves the data in its storage, and performs the computations
there. This means that even if your data is distributed, or it requires
GPU accelarated speed, Ibis code will be able to benefit from your storage
capabilities.

### Why not use SQL?

SQL is widely used and very convenient when writing simple queries. But as
the complexity of operations grow, SQL can become very difficult to deal with.

With Ibis, you can take fully advantage of software engineering techniques to
keep your code readable and maintainable, while writing very complex analytics
code.

### Why not use [SQLAlchemy](https://www.sqlalchemy.org/)?

SQLAlchemy is very convenient as an ORM (Object Relational Mapper), providing
a Python interface to SQL databases. Ibis uses SQLAlchemy internally, but aims
to provide a friendlier syntax for analytics code. And Ibis is also not limited
to SQL databases, but also can connect to distributed platforms and in-memory
representations.

### Why not use [Dask](https://dask.org/)?

Dask provides advanced parallelism, and can distribute pandas jobs. Ibis can
process data in a similar way, but for a different number of backends. For
example, given a Spark cluster, Ibis allows to perform analytics using it,
with a familiar Python syntax. Ibis plans to add support for a Dask backend
in the future.
