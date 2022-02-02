# [MySQL](https://www.mysql.com/)

## Install

Install dependencies for Ibis's MySQL dialect:

```sh
pip install 'ibis-framework[mysql]'
```

or

```sh
conda install -c conda-forge ibis-mysql
```

## Connect

Create a client by passing a connection string or individual parameters to
`ibis.mysql.connect`:

```python
con = ibis.mysql.connect(url='mysql+pymysql://ibis:ibis@mysql/ibis_testing')
con = ibis.mysql.connect(
    user='ibis',
    password='ibis',
    host='mysql',
    database='ibis_testing',
)
```

## API

The MySQL client is accessible through the `ibis.mysql` namespace.

Use `ibis.mysql.connect` with a SQLAlchemy-compatible connection string to
create a client.

<!-- prettier-ignore-start -->
::: ibis.backends.mysql.Backend
    rendering:
      heading_level: 3

<!-- prettier-ignore-end -->
