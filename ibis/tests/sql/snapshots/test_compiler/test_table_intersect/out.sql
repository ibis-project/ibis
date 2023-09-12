SELECT t0.`key`, t0.`value`
FROM (
  WITH t1 AS (
    SELECT t3.`string_col` AS `key`, t3.`double_col` AS `value`
    FROM functional_alltypes t3
    WHERE t3.`int_col` <= 0
  ),
  t2 AS (
    SELECT t3.`string_col` AS `key`, CAST(t3.`float_col` AS double) AS `value`
    FROM functional_alltypes t3
    WHERE t3.`int_col` > 0
  )
  SELECT *
  FROM t2
  INTERSECT
  SELECT *
  FROM t1
) t0