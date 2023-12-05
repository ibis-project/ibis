SELECT t0.*
FROM functional_alltypes t0
WHERE (t0.`double_col` > 3.14) AND
      (locate('foo', t0.`string_col`) - 1 >= 0) AND
      (((t0.`int_col` - 1) = 0) OR (t0.`float_col` <= 1.34))