SELECT
  CASE
    WHEN 2 >= 0
    THEN SUBSTRING(t0.string_col, 2 + 1)
    ELSE SUBSTRING(t0.string_col, LENGTH(t0.string_col) + 2 + 1)
  END AS "Substring(string_col, 2)"
FROM functional_alltypes AS t0