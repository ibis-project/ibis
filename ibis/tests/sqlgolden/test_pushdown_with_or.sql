SELECT *
FROM functional_alltypes
WHERE (`double_col` > 3.14) AND
      (locate('foo', `string_col`) - 1 >= 0) AND
      (((`int_col` - 1) = 0) OR (`float_col` <= 1.34))
