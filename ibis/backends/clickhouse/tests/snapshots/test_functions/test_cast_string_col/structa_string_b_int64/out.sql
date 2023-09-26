SELECT
  CAST(t0.string_col AS Tuple(a Nullable(String), b Nullable(Int64))) AS "Cast(string_col, !struct<a: string, b: int64>)"
FROM functional_alltypes AS t0