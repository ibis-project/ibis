# [PostgreSQL](https://www.postgresql.org/)

## Install

Install dependencies for Ibis's PostgreSQL dialect:

```sh
pip install 'ibis-framework[postgres]'
```

or

```sh
conda install -c conda-forge ibis-postgres
```

## Connect

Create a client by passing a connection string to the `url` parameter or
individual parameters to `ibis.postgres.connect`:

```python
con = ibis.postgres.connect(
   url='postgresql://postgres:postgres@postgres:5432/ibis_testing'
)
con = ibis.postgres.connect(
   user='postgres',
   password='postgres',
   host='postgres',
   port=5432,
   database='ibis_testing',
)
```

## API

The PostgreSQL client is accessible through the `ibis.postgres` namespace.

Use `ibis.postgres.connect` with a SQLAlchemy-compatible connection string to
create a client.

<!-- prettier-ignore-start -->
::: ibis.backends.postgres.Backend
    rendering:
      heading_level: 3

<!-- prettier-ignore-end -->
