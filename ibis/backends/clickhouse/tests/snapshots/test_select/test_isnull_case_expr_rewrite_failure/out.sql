SELECT
  SUM(CASE WHEN isNull(t0.string_col) THEN 1 ELSE 0 END) AS "Sum(Where(IsNull(string_col), 1, 0))"
FROM ibis_testing.functional_alltypes AS t0