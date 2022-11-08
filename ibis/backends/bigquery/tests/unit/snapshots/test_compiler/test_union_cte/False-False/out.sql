WITH t0 AS (
  SELECT `string_col`, sum(`double_col`) AS `metric`
  FROM functional_alltypes
  GROUP BY 1
)
SELECT *
FROM t0
UNION ALL
SELECT `string_col`, sum(`double_col`) AS `metric`
FROM functional_alltypes
GROUP BY 1
UNION ALL
SELECT `string_col`, sum(`double_col`) AS `metric`
FROM functional_alltypes
GROUP BY 1