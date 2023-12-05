WITH t0 AS (
  SELECT t2.`float_col`, t2.`timestamp_col`, t2.`int_col`, t2.`string_col`
  FROM alltypes t2
  WHERE t2.`timestamp_col` < '2014-01-01T00:00:00'
)
SELECT count(t1.`foo`) AS `Count(foo)`
FROM (
  SELECT t0.`string_col`, sum(t0.`float_col`) AS `foo`
  FROM t0
  GROUP BY 1
) t1