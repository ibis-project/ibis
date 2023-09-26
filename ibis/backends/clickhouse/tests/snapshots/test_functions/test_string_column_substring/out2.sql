SELECT
  CASE
    WHEN 0 >= 0
    THEN SUBSTRING(t0.string_col, 0 + 1, 3)
    ELSE SUBSTRING(t0.string_col, LENGTH(t0.string_col) + 0 + 1, 3)
  END AS "Substring(string_col, 0, 3)"
FROM functional_alltypes AS t0