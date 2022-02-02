# [SQLite](https://www.sqlite.org/)

## Install

Install dependencies for Ibis's SQLite dialect:

```sh
pip install 'ibis-framework[sqlite]'
```

or

```sh
conda install -c conda-forge ibis-sqlite
```

## Connect

Create a client by passing a path to a SQLite database to `ibis.sqlite.connect`:

```python
import ibis

ibis.sqlite.connect('path/to/my/sqlite.db')
```

## API

The SQLite client is accessible through the `ibis.sqlite` namespace.

<!-- prettier-ignore-start -->
::: ibis.backends.sqlite.Backend
    rendering:
      heading_level: 3

<!-- prettier-ignore-end -->
