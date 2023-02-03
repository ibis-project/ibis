SELECT t0.`key`
FROM (
  SELECT t1.`string_col` AS `key`, CAST(t1.`float_col` AS double) AS `value`
  FROM functional_alltypes t1
  WHERE t1.`int_col` > 0
  EXCEPT
  SELECT t1.`string_col` AS `key`, t1.`double_col` AS `value`
  FROM functional_alltypes t1
  WHERE t1.`int_col` <= 0
) t0