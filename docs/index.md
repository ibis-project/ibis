---
hide:
  - toc
---

# :ibis-logo: Ibis

## Expressive analytics in Python at any scale.

<script 
    src="https://asciinema.org/a/yp5Ww4XKyjJsUCXkEz5or9rPq.js"
    data-autoplay="true"
    data-preload="true"
    data-loop="true"
    data-i="4"
    data-rows="20"
    id="asciicast-yp5Ww4XKyjJsUCXkEz5or9rPq"
    async>
</script>

## Installation

=== "pip"

    ```sh
    pip install ibis-framework
    ```

{% for mgr in ["conda", "mamba"] %}
=== "{{ mgr }}"

    ```sh
    {{ mgr }} install -c conda-forge ibis-framework
    ```

{% endfor %}

Try it out!

```python
python -c 'import ibis; print(ibis.__version__)'
```

## Features

### SQL Coverage

#### Anything you can write in a `SELECT` statement you can write in Ibis.

=== "Group By"

    ##### SQL

    ```sql
    SELECT f, sum(a + b) AS d
    FROM t
    GROUP BY f
    ```

    ##### Ibis

    ```python
    t.group_by("f").aggregate(d=(t.a + t.b).sum())
    ```

=== "Join"

    ##### SQL

    ```sql
    SELECT exp(t.a) AS d
    FROM t
    LEFT SEMI JOIN s
      ON t.x = t.y
    ```

    ##### Ibis

    ```python
    t.semi_join(s, t.x == t.y).select([lambda t: t.a.exp().name("d")])
    ```

=== "Window Functions"

    ##### SQL

    ```sql
    SELECT *, avg(x) OVER (PARTITION BY y) as z
    FROM t
    ```

    ##### Ibis

    ```python
    t.group_by("y").mutate(z=t.x.avg())
    ```

!!! tip "Coming from SQL?"

    Check out [Ibis for SQL Programmers](ibis-for-sql-programmers)

### Abstract Over SQL Dialects

#### No more rewrites when scaling up or down.

=== "SQLite"

    ```python
    con = ibis.sqlite.connect("my_sqlite.db")
    ```

=== "PostgreSQL"

    ```python
    con = ibis.postgres.connect(user="me", host="my_computer", port=9090)
    ```

=== "BigQuery"

    ```python
    con = ibis.bigquery.connect(project_id="my_project_id", dataset_id="my_dataset_id")
    ```

```python
t = con.table("t")
t.group_by("y").mutate(z=t.x.avg())
```

### Ecosystem

#### Ibis builds on top of and works with existing Python tools.

```python
expr = t.semi_join(s, t.x == t.y).select([lambda t: t.a.exp().name("d")]).head(2)
df = expr.execute()  # a pandas DataFrame!
```

## Example

Let's compute the number of citizens per squared kilometer in Asia:

```python
>>> import ibis
>>> db = ibis.sqlite.connect("geography.db")
>>> countries = db.table("countries")
>>> asian_countries = countries.filter(countries.continent == "AS")
>>> density_in_asia = asian_countries.population.sum() / asian_countries.area_km2.sum()
>>> density_in_asia.execute()
130.7019141926602
```

!!! tip "Learn more!"

    Learn more about Ibis in [our tutorial](tutorial/01-Introduction-to-Ibis).

## Comparison to other tools

=== "SQL"

    !!! tip "Coming from SQL?"

        Check out [Ibis for SQL Programmers](ibis-for-sql-programmers)!

    Ibis gives you the benefit of a programming language. You don't need to
    sacrifice maintainability to get to those insights!

    === "Ibis"

        ``` py title="docs/example.py" linenums="1"
        --8<-- "docs/example.py"
        ```

    === "SQL"

        ``` sql title="docs/example.sql" linenums="1"
        --8<-- "docs/example.sql"
        ```

=== "SQLAlchemy"

    Ibis aims to be more concise and composable than
    [SQLAlchemy](https://www.sqlalchemy.org/) when writing interactive
    analytics code.

    !!! success "Ibis :heart:'s SQLAlchemy"

        Ibis generates SQLAlchemy expressions for some of our backends
        including the [PostgreSQL](./backends/PostgreSQL.md) and
        [SQLite](./backends/SQLite.md) backends!

    === "Ibis"

        ``` py title="docs/example.py" linenums="1"
        --8<-- "docs/example.py"
        ```

    === "SQLAlchemy"

        ``` py title="docs/sqlalchemy_example.py" "linenums="1"
        --8<-- "docs/sqlalchemy_example.py"
        ```

<div class="download-button" markdown>
[:fontawesome-solid-cloud-arrow-down: Download the example data](https://storage.googleapis.com/ibis-testing-data/crunchbase.db){ .md-button .md-button--primary }
</div>

## What's Next?

!!! question "Need a specific backend?"

    Take a look at the [backends](./backends/index.md) documentation!

!!! tip "Interested in contributing?"

    Get started by [setting up a development environment](./contribute/01_environment.md)!
