# Working with the Backend Test Suite

!!! danger "Before you start"

    This section assumes you have a working [development environment](01_environment.md).

!!! info "You may be able to skip this section"

    If you haven't made changes to the core of ibis (e.g., `ibis/expr`)
    or any specific backends (`ibis/backends`) this material isn't necessary to
    follow to make a pull request.

## Motivation

One the primary challenges when developing against the ibis codebase is testing
backends that require non-trivial setup.

Moreover, many of the backends that ibis works with have very different
deployment deployment models:

- **In-process** systems like SQLite
- **Client-server** systems like PostgreSQL and MySQL
- Systems that **run the gamut** of deployment models like ClickHouse
- Systems that run **on-premises**, like Impala

This section of the docs is describes how to work with the backend test suite.

## Backend Testing with Compose

Here is the list of backends that can be tested using `docker-compose`.

| Backend    | Docker Compose Services |
| ---------- | ----------------------- |
| ClickHouse | `clickhouse`            |
| PostgreSQL | `postgres`              |
| impala     | `impala`, `kudu`        |
| mysql      | `mysql`                 |

### Testing a Compose Service

!!! check "Check your current directory"

    Make sure you're inside of your clone of the ibis GitHub repository

Let's fire up a PostgreSQL server and run tests against it.

#### Start the `postgres` Service

Open a new shell and run

```sh
docker-compose up --build postgres
```

Test the connection in the original shell using

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

#### Load Data

The backend needs to be populated with test data:

1.  Download the data

    ```sh
    python ci/datamgr.py download
    ```

2.  In the original terminal, run

    ```sh
    python ci/datamgr.py load postgres
    ```

    You should see a bit of logging, and the command should complete shortly thereafter.

#### Run the test suite

You're now ready to run the test suite for the postgres backend:

```sh
pytest -m postgres
```

Please [file an issue](https://github.com/ibis-project/ibis/issues/new) if the
test suite fails for any reason.
