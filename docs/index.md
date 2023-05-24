---
hide:
  - toc
  - navigation
  - footer
---

# <span style="font-size: 1.5em; margin: 0">:ibis-logo: The Ibis Project</span>

## The flexibility of Python analytics with the scale and performance of modern SQL.

---

<div class="install-tutorial-button" markdown>
[Getting Started](./getting_started.md){ .md-button .md-button--primary }
[Install](./install.md){ .md-button }
</div>

---

```python title="Write high-level Python code"
>>> import ibis
>>> movies = ibis.examples.ml_latest_small_movies.fetch()
>>> rating_by_year = movies.group_by('year').avg_rating.mean()
>>> q = rating_by_year.order_by(rating_by_year.year.desc())
```

```py title="Compile to SQL"
>>> con.compile(q)

SELECT year, avg(avg_rating)
FROM movies t1
GROUP BY t1.year
ORDER BY t1.year DESC
```

```py title="Execute on multiple backends"
>>> con.execute(q)

     year  mean(avg_rating)
0    2021          2.586362
1    2020          2.719994
2    2019          2.932275
3    2018          3.005046
4    2017          3.071669
```

---

## Features

- **Consistent syntax across backends**: Enjoy a uniform Python API, whether using [DuckDB](https://duckdb.org), [PostgreSQL](https://postgresql.org), [PySpark](https://spark.apache.org/docs/latest/api/python/index.html), [BigQuery](https://cloud.google.com/bigquery/), or [any other supported backend](./backends/index.md).
- **Performant**: Execute queries as fast as the database engine itself.
- **Interactive**: Explore data in a notebook or REPL.
- **Extensible**: Add new operations, optimizations, and custom APIs.
- **Free and open-source**: licensed under Apache 2.0, [available on Github](https://github.com/ibis-project/ibis/blob/master/README.md).
