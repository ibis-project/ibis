SELECT
  CAST("t0"."string_col" AS Map(String, Nullable(Int64))) AS "Cast(string_col, !map<string, int64>)"
FROM "functional_alltypes" AS "t0"