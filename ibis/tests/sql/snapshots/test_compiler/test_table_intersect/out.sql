WITH t0 AS (
  SELECT t2.`string_col` AS `key`, t2.`double_col` AS `value`
  FROM functional_alltypes t2
  WHERE t2.`int_col` <= 0
),
t1 AS (
  SELECT t2.`string_col` AS `key`, CAST(t2.`float_col` AS double) AS `value`
  FROM functional_alltypes t2
  WHERE t2.`int_col` > 0
)
SELECT *
FROM t1
INTERSECT
SELECT *
FROM t0