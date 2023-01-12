SELECT count(t0.`foo`) AS `Count(foo)`
FROM (
  SELECT t1.`string_col`, sum(t1.`float_col`) AS `foo`
  FROM (
    SELECT t2.`float_col`, t2.`timestamp_col`, t2.`int_col`, t2.`string_col`
    FROM alltypes t2
    WHERE t2.`timestamp_col` < '2014-01-01T00:00:00'
  ) t1
  GROUP BY 1
) t0