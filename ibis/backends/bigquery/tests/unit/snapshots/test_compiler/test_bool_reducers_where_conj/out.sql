SELECT
  sum(
    IF((
      t0.`month` > 6
    ) AND (
      t0.`month` < 10
    ), CAST(t0.`bool_col` AS INT64), NULL)
  ) AS `Sum_bool_col_ And_Greater_month_ 6_ Less_month_ 10`
FROM functional_alltypes AS t0