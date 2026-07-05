# Feldera Backend Test Notes

This file documents the known failure taxonomy for the Feldera backend in the
shared Ibis test suite.  It is intended to help contributors distinguish
semantic bugs from expected limitations.

## Architecture Reminder

- **Pipeline SQL** is parsed by Apache **Calcite** (Postgres-flavoured).
- **Ad-hoc SQL** (`execute()`, `to_pyarrow()`, `raw_sql()`) is parsed by
  Apache **DataFusion**.
- The compiler targets DataFusion for ad-hoc queries; see
  `test_adhoc_surface.py` for the validated function matrix.

## Failure Categories

### 1. Row Ordering (benign)

Feldera is an incremental engine; output order is unspecified without an
explicit `ORDER BY`.  The test harness sets `force_sort = True`, which sorts
both the actual and expected results before comparison.  A large fraction of
historical failures were ordering mismatches that disappeared after enabling
sorting.

**Action:** None — already handled.

### 2. Unsupported Operations

The compiler declares an `UNSUPPORTED_OPS` tuple that maps the following Ibis
operations to `OperationNotDefinedError`:

- `Sample`, `RandomScalar`, `RandomUUID`
- `Arbitrary`, `Mode`, `Kurtosis`
- `Quantile`, `MultiQuantile`, `ApproxMultiQuantile`
- `First`, `Last`
- `RegexSplit`
- `TimestampBucket`
- `TypeOf`

**Action:** Add compiler overrides or upstream support in Feldera/DataFusion,
then remove from `UNSUPPORTED_OPS`.

### 3. DDL Limitations

Feldera tables must be declared in the pipeline SQL program; they cannot be
created ad hoc at runtime.

- `create_table(schema=...)` → `NotImplementedError`
- `create_table(obj=ir.Table)` → `NotImplementedError` (only `obj=pd.DataFrame` works)
- `drop_table(...)` → `NotImplementedError`
- `_register_in_memory_table(...)` → `NotImplementedError`

Shared tests that require temporary tables or memtables are skipped via
`pytest.skip` in `test_client.py` and `test_string.py`.

**Action:** None by design.

### 4. Type Support

The following complex types are conservatively disabled until validated against
a live Feldera instance:

- **Arrays** (`supports_arrays = False`)
- **Structs** (`supports_structs = False`)
- **Maps** (`supports_map = False`)
- **JSON** (`supports_json = False`)

Calcite supports ARRAY and MAP; once the schema conversion and compiler
lowering are validated, flip the flags and remove the `pytest.mark.never`
entries in `ibis/backends/tests/conftest.py`.

### 5. Decimal Edge Cases

Feldera rejects `Infinity` and `NaN` casts to DECIMAL.  These are correctly
marked `notyet` with `raises=FelderaAPIError` in `test_numeric.py`.

### 6. UDFs

Python UDFs are not supported.  Marked `notyet` in `test_udf.py`.

### 7. Expression Caching

Feldera does not support the expression-caching contract used by some shared
tests.  Marked alongside RisingWave in `test_expr_caching.py`.

## Running the Tests Locally

```bash
# Start the Feldera container
docker compose up feldera -d

# Run the ad-hoc surface validation suite
uv run pytest ibis/backends/feldera/tests/test_adhoc_surface.py -p no:cacheprovider -v

# Run the smoke E2E suite
uv run pytest ibis/backends/feldera/tests/test_smoke_e2e.py -p no:cacheprovider -v

# Run the shared backend tests (serially — one pipeline-manager instance)
just test feldera
```

## Next Steps

1. Validate ARRAY / STRUCT / MAP support and flip feature flags.
2. Validate `Clip` NULL-propagation workaround against newer DataFusion
   releases; remove if fixed upstream.
3. Add compiler support for `TimestampBucket` once Feldera/DataFusion supports
   it.
4. Investigate remaining semantic failures in the shared suite and add targeted
   compiler overrides or xfail marks.
