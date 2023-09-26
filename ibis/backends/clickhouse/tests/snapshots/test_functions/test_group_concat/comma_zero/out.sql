SELECT
  CASE
    WHEN empty(groupArrayIf(t0.string_col, t0.bool_col = 0))
    THEN NULL
    ELSE arrayStringConcat(groupArrayIf(t0.string_col, t0.bool_col = 0), ',')
  END AS "GroupConcat(string_col, ',', Equals(bool_col, 0))"
FROM functional_alltypes AS t0