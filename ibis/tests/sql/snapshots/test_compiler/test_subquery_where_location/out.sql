SELECT count(`foo`)
FROM (
  SELECT `string_col`, sum(`float_col`) AS `foo`
  FROM (
    SELECT `float_col`, `timestamp_col`, `int_col`, `string_col`
    FROM alltypes
    WHERE `timestamp_col` < '2014-01-01T00:00:00'
  ) t1
  GROUP BY 1
) t0