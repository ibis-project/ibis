WITH `t4` AS (
  SELECT
    `t3`.`file_date`,
    `t3`.`PARTITIONTIME`,
    `t3`.`val`,
    `t3`.`val` * 2 AS `XYZ`
  FROM (
    SELECT
      *
    FROM (
      SELECT
        CAST(`t1`.`file_date` AS DATE) AS `file_date`,
        `t1`.`PARTITIONTIME`,
        `t1`.`val`
      FROM (
        SELECT
          *
        FROM `unbound_table` AS `t0`
        WHERE
          `t0`.`PARTITIONTIME` < DATE(2017, 1, 1)
      ) AS `t1`
    ) AS `t2`
    WHERE
      `t2`.`file_date` < DATE(2017, 1, 1)
  ) AS `t3`
)
SELECT
  `t6`.`file_date`,
  `t6`.`PARTITIONTIME`,
  `t6`.`val`,
  `t6`.`XYZ`
FROM `t4` AS `t6`
INNER JOIN `t4` AS `t7`
  ON TRUE