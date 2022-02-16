---
hide:
  - toc
---

# :ibis-logo: Ibis

<div markdown>
## Expressive analytics in Python, whatever the scale.
</div>

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
    t.group_by("f").aggregate(d=lambda t: (t.a + t.b).sum())
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
    t.group_by("y").mutate(z=lambda t: t.x.avg())
    ```

!!! tip "Coming from SQL?"

    Check out [Ibis for SQL Programmers](/user_guide/sql)!

### Abstract Over SQL Dialects

#### No more rewrites when scaling up or down.

=== "BigQuery"

    ```python
    con = ibis.bigquery.connect(project_id=...)
    ```

=== "SQLite"

    ```python
    con = ibis.sqlite.connect("path/to/sqlite.db")
    ```

=== "PostgreSQL"

    ```python
    con = ibis.postgres.connect(user=..., host=..., port=...)
    ```

```python
t = con.table("t")
t.group_by("y").mutate(z=lambda t: t.x.avg())
```

### Ecosystem

#### Ibis builds on top of and works with existing Python tools.

```python
t.semi_join(s, t.x == t.y).select([lambda t: t.a.exp().name("d")]).head(2)
df = expr.execute()  # a pandas DataFrame!
```

## Example

Let's compute the number of citizens per squared kilometer in Asia:

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

    !!! tip "Coming from SQL?"

        Check out [Ibis for SQL Programmers](/user_guide/sql)!

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
        including the [PostgreSQL](/backends/postgres) and
        [SQLite](/backends/sqlite) backends!

    === "Ibis"

        ``` py title="docs/example.py" linenums="1"
        --8<-- "docs/example.py"
        ```

    === "SQLAlchemy"

        ``` python title="docs/sqlalchemy_example.py" "linenums="1"
        --8<-- "docs/sqlalchemy_example.py"
        ```
