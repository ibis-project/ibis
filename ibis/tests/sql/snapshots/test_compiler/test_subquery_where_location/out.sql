SELECT count(t0.`foo`) AS `Count(foo)`
FROM (
  SELECT t1.`string_col`, sum(t1.`float_col`) AS `foo`
  FROM alltypes t1
  WHERE t1.`timestamp_col` < '2014-01-01T00:00:00'
  GROUP BY 1
) t0