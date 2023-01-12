WITH t0 AS (
  SELECT t1.`string_col`, sum(t1.`double_col`) AS `metric`
  FROM functional_alltypes t1
  GROUP BY 1
)
SELECT *
FROM t0
UNION DISTINCT
SELECT t1.`string_col`, sum(t1.`double_col`) AS `metric`
FROM functional_alltypes t1
GROUP BY 1
UNION DISTINCT
SELECT t1.`string_col`, sum(t1.`double_col`) AS `metric`
FROM functional_alltypes t1
GROUP BY 1