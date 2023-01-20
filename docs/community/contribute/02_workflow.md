# Contribute to the Ibis Codebase

## Getting started

First, set up a [development environment](01_environment.md).

## Taking Issues

If you find an issue you want to work on, write a comment with the text
`/take` on the issue. GitHub will then assign the issue to you.

## Running the test suite

To run tests that do not require a backend:

```sh
pytest -m core
```

### Backend Test Suites

!!! info "You may be able to skip this section"

    If you haven't made changes to the core of ibis (e.g., `ibis/expr`)
    or any specific backends (`ibis/backends`) this material isn't necessary to
    follow to make a pull request.

First, we need to download example data to run the tests successfully:

```sh
just download-data
```

To run the tests for a specific backend (e.g. sqlite):

```sh
pytest -m sqlite
```

## Setting up non-trivial backends

These client-server backends need to be started before testing them.
They can be started with `docker-compose` directly, or using the `just` tool.

- ClickHouse: `just up clickhouse`
- PostgreSQL: `just up postgres`
- MySQL: `just up mysql`
- impala: `just up impala`

### Test the backend locally

If anything seems amiss with a backend, you can of course test it locally:

```sh
export PGPASSWORD=postgres
psql -t -A -h localhost -U postgres -d ibis_testing -c "select 'success'"
```

## Writing the commit

Ibis follows the [Conventional Commits](https://www.conventionalcommits.org/) structure.
In brief, the commit summary should look like:

    fix(types): make all floats doubles

The type (e.g. `fix`) can be:

- `fix`: A bug fix. Correlates with PATCH in SemVer
- `feat`: A new feature. Correlates with MINOR in SemVer
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
  `
  If the commit fixes a Github issue, add something like this to the bottom of the description:

      fixes #4242

## Submit a PR

Ibis follows the standard Git Pull Request process. The team will review the PR and merge when it's ready.
