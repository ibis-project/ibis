<div class="row">
    <div class="col">
        <section class="jumbotron text-center">
            <h1>Ibis</h1>
            <p>
                Write your analytics code once, run it everywhere.
            </p>
            <p>
                <a class="btn btn-primary" href="{{ base_url }}/getting_started.html">Try Ibis now &raquo;</a>
            </p>
        </section>
    </div>
</div>

Ibis provides a standard way to write analytics code, that then can be run in
multiple engines.

## Main features

- Full coverage of SQL features: You can code in Ibis anything you can implement in a SQL SELECT
- Transparent to SQL implementation differences: Write standard code that translate to any SQL syntax
- High performance execution: Execute at the speed of your backend, not your local computer
- Integration with community data formats and tools (e.g. pandas, Parquet, Avro...)

## Supported engines

- Standard DBMS: PostgreSQL, MySQL, SQLite
- Hadoop based systems: Hive, Impala, Spark, PySpark, Kudu
- Other big data systems: Google BigQuery
- Analytical DBMS: ClickHouse
- GPU accelerated analytical DBMS: omniscidb
- Python dataframe libraries: pandas

## Example

The next example is all the code you need to connect to a database with a
countries database, and compute the number of citizens per squared kilometer in Asia:

```python
import ibis

db = ibis.sqlite.connect('geonames.db')

countries = geonames_db.table('countries')

asian_countries = countries[countries['continent'] == 'AS']

density_in_asia = asian_countries['population'].sum() / asian_countries['area_km2'].sum()

density_in_asia.execute()
```

## Comparison to other tools

### Why not use pandas?

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
keep your code readable and maintainable, while writing very complex analitics
code.

### Why not use SQLAlchemy?

SQLAlchemy is very convenient as an ORM (Object Relational Mapper), providing
a Python interface to SQL databases. But SQLAlchemy is focussed on access to
the data, and not to perform analytics on it. And it is mostly limited to
conventional SQL databases, and doesn't support big data platforms or specialized
analytical tools.
