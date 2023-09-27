SELECT
  SUM(CASE WHEN isNull(t0.string_col) THEN 1 ELSE 0 END) AS "Sum(IfElse(IsNull(string_col), 1, 0))"
FROM functional_alltypes AS t0