# Feldera ↔ Ibis backend — status & findings

## What works

`ibis.feldera.connect(host=..., pipeline=...)` returns a working Ibis connection
to a running Feldera pipeline. Tested against Feldera 0.316.0 (server) /
0.317.0 (Python SDK), Ibis main, Python 3.13.

End-to-end flows verified against a live Docker Feldera:
- `list_tables`, `get_schema`, `table()`
- `compile()` → valid Feldera (Calcite/Postgres-flavored) SQL
- `execute()` → pandas DataFrame (snapshot via `pipeline.query_arrow`)
- `to_pyarrow()` → PyArrow Table
- `create_table(obj=df)` → declares a materialized table + pushes data
- Complex expressions: filter → group_by → agg → order_by, window funcs, joins

## Test-suite coverage (Ibis backend-test harness)

Ran the standard Ibis `ibis/backends/tests/` suites against Feldera:

```
687 passed, 813 failed, 28 skipped, 31 xfailed  (15 core suites)
```

Suites: test_aggregation, test_generic, test_string, test_temporal,
test_numeric, test_column, test_conditionals, test_join, test_window,
test_array, test_map, test_uuid, test_client, test_export, test_binary.

### Important caveat: row ordering

A large fraction of the "failures" are **row-ordering mismatches**, not
semantic bugs. Feldera is an incremental/streaming engine; materialized-table
output order is unspecified unless the query has an explicit `ORDER BY`.
The Ibis test harness compares result rows positionally, so any reorder shows
as a failure even when the value multiset is identical.

Verified by spot-check: e.g. `test_simple_math_functions_columns[sign]` fails
positionally, but `sorted(result) == sorted(expected)` passes. The compiler
correctly lowers `sign` → `SIGNUM` (Calcite spelling) and the values are right.

The `force_sort` TestConf flag exists for this, but it over-sorts on all
columns and breaks on NULL/NaN/dtype-mismatch columns, so it is **not** a net
win here. A more surgical ordering strategy (sort on a stable key column when
the expression has one) is future work.

## Failure taxonomy

Buckets (from the 813 failures), largest first:

| Bucket | Count | Nature |
|---|---|---|
| `FelderaAPIError` / HTTP 400 — SQL Feldera rejects | ~300 | real gaps |
| `AssertionError` (mostly ordering, see caveat) | ~193 | mostly false-negatives |
| `NotImplementedError: In-memory tables not supported` | ~168 | by design |
| `AttributeError` (test infra assuming in-memory) | ~62 | by design |
| `OperationNotDefinedError` (compiler rule missing) | ~54 | expected, mark `notimpl` |
| `ValueError` | ~36 | mixed |
| `KeyError` | ~13 | test infra |
| `UnsupportedOperationError` | ~11 | e.g. sample correlation |

### Specific Calcite/Feldera function gaps (SQL that compiles but Feldera rejects)

| Ibis op → emitted fn | Calcite has | Occurrences |
|---|---|---|
| `sign` → `SIGN` | `SIGNUM` ✅ fixed in our dialect | (was 114) |
| interval construction → `make_interval` | not available | 46 |
| `typeof` → `pg_typeof` | `arrow_typeof` only | 40 |
| date extraction → `date(...)` | `to_date` | 26 |
| timestamp construction → `make_timestamp` | `to_timestamp` | 22 |
| `jsonb_object_agg` | not available | 18 |
| `first` / `last` aggregate | not available | 28 |
| `regexp_split_to_array` | not available | 6 |
| `percentile_disc` | not available | 6 |
| `array(...)` constructor | `array_agg`/literals | 4 |

These are genuine Calcite/Feldera function-name or capability gaps. Most are
addressable either by:
1. dialect-level function remapping (like `SIGN`→`SIGNUM`), or
2. marking the op `notimpl`/`notyet` on Feldera in the shared test files.

## By-design limitations (not bugs)

- **No in-memory tables.** Feldera tables must be declared in the pipeline's
  SQL program and materialized. `create_table(obj=df)` works (it emits
  `CREATE TABLE ... WITH ('materialized'='true')` + `input_pandas`), but
  registering a Python object as a queryable table without a pipeline is not
  possible. ~168 test failures are tests that assume in-memory table support;
  these should be marked `never("feldera")`.
- **Pipeline fixed at connect time.** Feldera has pipelines, not catalogs.
  `connect(pipeline=...)` binds to one pipeline; cross-pipeline queries are
  out of scope. Matches the dbt-feldera convention (schema = pipeline,
  relation = table/view).
- **`execute()` = snapshot.** Like the Flink backend, `execute()` compiles to
  a `SELECT`, runs `pipeline.query_arrow()`, and returns a pandas DataFrame.
  Streaming/changelog consumption is out of scope for this prototype.
- **Materialized requirement.** `CREATE TABLE` needs
  `WITH ('materialized' = 'true')` and views need `CREATE MATERIALIZED VIEW`
  to be queryable by ad-hoc `SELECT`. This is a Feldera constraint, surfaced
  through `create_table`.

## Architecture

- `ibis/backends/sql/dialects.py` — `Feldera(Postgres)` sqlglot dialect
  (FLOAT→REAL, INT→INTEGER, `SIGN`→`SIGNUM`).
- `ibis/backends/sql/datatypes.py` — `FelderaType(PostgresType)` (non-string
  MAP key support).
- `ibis/backends/sql/compilers/feldera.py` —
  `FelderaCompiler(PostgresCompiler)`.
- `ibis/backends/feldera/__init__.py` — `Backend(SQLBackend)` with `connect`,
  `do_connect`, `raw_sql`, `execute`, `to_pyarrow`, `list_tables`, `get_schema`,
  `create_table`, `version`, `_fetch_from_cursor` (handles empty results).
- `ibis/backends/feldera/tests/conftest.py` — `BackendTest` subclass that
  builds a pipeline SQL program from the parquet test-data dtypes, pushes data
  via `input_pandas`, yields an Ibis connection.

Follows the RisingWave precedent (subclass PostgresCompiler) and the Flink
precedent (snapshot execute(), no in-memory tables, test conf builds pipeline).

## Files changed in ibis repo

- `ibis/backends/sql/dialects.py` (Feldera dialect)
- `ibis/backends/sql/datatypes.py` (FelderaType)
- `ibis/backends/sql/compilers/feldera.py` (new — compiler)
- `ibis/backends/sql/compilers/__init__.py` (register compiler)
- `ibis/backends/feldera/__init__.py` (new — backend)
- `ibis/backends/feldera/tests/conftest.py` (new — test harness)
- `ibis/backends/feldera/tests/smoke_e2e.py` (new — smoke test)
- `pyproject.toml` (entry point + `feldera` test marker)

## Next steps (for the PR / community)

1. Add `notimpl`/`never` markers for Feldera on the relevant shared test
   functions (the 813 will split into true-fails vs. known-unsupported).
2. Dialect remapping for the cheap function-name gaps (`make_timestamp`→
   `to_timestamp`, etc.) where semantics allow.
3. Decide ordering strategy with Ibis maintainers: a backend hook for
   "results are unordered" so the harness sorts on a stable key.
4. Open draft PR to ibis-project/ibis (issue #10609).
