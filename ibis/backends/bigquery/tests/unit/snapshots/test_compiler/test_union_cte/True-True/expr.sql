WITH t0 AS (
  SELECT `string_col`, sum(`double_col`) AS `metric`
  FROM functional_alltypes
  GROUP BY 1
)
SELECT *
FROM t0
UNION DISTINCT
SELECT `string_col`, sum(`double_col`) AS `metric`
FROM functional_alltypes
GROUP BY 1
UNION DISTINCT
SELECT `string_col`, sum(`double_col`) AS `metric`
FROM functional_alltypes
GROUP BY 1