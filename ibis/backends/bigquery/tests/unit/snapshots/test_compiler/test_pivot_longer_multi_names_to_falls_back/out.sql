SELECT
  `t1`.`id`,
  `t1`.`__pivoted__`.`metric` AS `metric`,
  `t1`.`__pivoted__`.`grp` AS `grp`,
  `t1`.`__pivoted__`.`val` AS `val`
FROM (
  SELECT
    `t0`.`id`,
    IF(pos = pos_2, `__pivoted__`, NULL) AS `__pivoted__`
  FROM `t` AS `t0`
  CROSS JOIN UNNEST(GENERATE_ARRAY(
    0,
    GREATEST(
      ARRAY_LENGTH(
        [
          STRUCT('x' AS `metric`, 'a' AS `grp`, `t0`.`x_a` AS `val`),
          STRUCT('x' AS `metric`, 'b' AS `grp`, `t0`.`x_b` AS `val`)
        ]
      )
    ) - 1
  )) AS pos
  CROSS JOIN UNNEST([
    STRUCT('x' AS `metric`, 'a' AS `grp`, `t0`.`x_a` AS `val`),
    STRUCT('x' AS `metric`, 'b' AS `grp`, `t0`.`x_b` AS `val`)
  ]) AS `__pivoted__` WITH OFFSET AS pos_2
  WHERE
    pos = pos_2
    OR (
      pos > (
        ARRAY_LENGTH(
          [
            STRUCT('x' AS `metric`, 'a' AS `grp`, `t0`.`x_a` AS `val`),
            STRUCT('x' AS `metric`, 'b' AS `grp`, `t0`.`x_b` AS `val`)
          ]
        ) - 1
      )
      AND pos_2 = (
        ARRAY_LENGTH(
          [
            STRUCT('x' AS `metric`, 'a' AS `grp`, `t0`.`x_a` AS `val`),
            STRUCT('x' AS `metric`, 'b' AS `grp`, `t0`.`x_b` AS `val`)
          ]
        ) - 1
      )
    )
) AS `t1`