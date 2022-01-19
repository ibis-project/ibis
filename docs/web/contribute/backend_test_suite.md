# Working with the Backend Test Suite

!!! danger "Before you start"

    This section assumes you've already [set up a development environment](environment.md).

!!! info "You may be able to skip this section"

    If you haven't made changes to any core parts of ibis (e.g., `ibis/expr`)
    or any specific backends (`ibis/backends`) this material isn't necessary to
    follow to make a pull request.

One the primary challenges when working with the ibis codebase is testing.

Ibis supports a execution against a number of backends with wildly varying
deployment models:

- Single-node systems like SQLite
- Multi-node client-server systems like PostgreSQL and MySQL
- Systems that span a variety of different models like ClickHouse
- Systems that are designed to run on premises, like Impala

This presents quite a challenge for testing. This page is all about how to work
with the backend test suite.

## Systems that can be tested without any additional infrastructure

Many backends can be tested without `docker-compose`.

The `sqlite`, `datafusion`, and any `pandas`-based backends can all be tested
without needing to do anything except loading data.

## Backend Testing with Compose

Here is the list of backends that can be tested using `docker-compose`.

| Backend    | Docker Compose Services |
| ---------- | ----------------------- |
| ClickHouse | `clickhouse`            |
| PostgreSQL | `ibis-postgres`         |
| impala     | `impala`                |
| kudu       | `kudu`, `impala`        |
| mysql      | `mysql`                 |

### Testing a Compose Service

!!! check "Check your current directory"

    Make sure you're inside of your clone of the ibis GitHub repository

Let's fire up a PostgreSQL server and run tests against it.

#### Start the Service

Open a new shell and run

```sh
docker-compose up --build ibis-postgres
```

Test the connection in a different shell using

```sh
export PGPASSWORD=postgres
psql -t -A -h localhost -U postgres -d ibis_testing -c "select 'success'"
```

You should see this output:

```console
success
```

!!! warning "PostgreSQL doesn't start up instantly"

    It takes a few seconds for postgres to start, so if the previous
    command fails wait a few seconds and try again

Congrats, you now have a PostgreSQL server running and are ready to run tests!

First we need to load some data:

#### Load Data

1.  Download the data

    ```sh
    python ci/datamgr.py download
    ```

2.  In the original terminal, run

    ```sh
    python ci/datamgr.py postgres
    ```

    You should see a bit of logging, and the command should complete shortly thereafter.

#### Run the test suite

You're now ready to run the test suite for the postgres backend:

```sh
export PYTEST_BACKENDS=postgres
pytest ibis/backends/postgres/tests ibis/backends/tests
```

Running the tests may take some time, but eventually they should finish successfully.

Please report any failures upstream, even if you're not sure if the failure is bug.
