SELECT t0.`key`
FROM (
  SELECT t1.`key`, t1.`value`
  FROM (
    WITH t2 AS (
      SELECT t4.`string_col` AS `key`, t4.`double_col` AS `value`
      FROM functional_alltypes t4
      WHERE t4.`int_col` <= 0
    ),
    t3 AS (
      SELECT t4.`string_col` AS `key`, CAST(t4.`float_col` AS double) AS `value`
      FROM functional_alltypes t4
      WHERE t4.`int_col` > 0
    )
    SELECT *
    FROM t3
    UNION ALL
    SELECT *
    FROM t2
  ) t1
) t0