SELECT count(`foo`) AS `Count(foo)`
FROM (
  SELECT `string_col`, sum(`float_col`) AS `foo`
  FROM (
    SELECT `float_col`, `timestamp_col`, `int_col`, `string_col`
    FROM alltypes
    WHERE `timestamp_col` < '20140101'
  ) t1
  GROUP BY 1
) t0