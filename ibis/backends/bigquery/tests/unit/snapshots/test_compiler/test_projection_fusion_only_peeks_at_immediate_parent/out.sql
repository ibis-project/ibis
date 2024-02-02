WITH `t1` AS (
  SELECT
    CAST(`t0`.`file_date` AS DATE) AS `file_date`,
    `t0`.`PARTITIONTIME`,
    `t0`.`val`,
    `t0`.`val` * 2 AS `XYZ`
  FROM `unbound_table` AS `t0`
  WHERE
    `t0`.`PARTITIONTIME` < DATE(2017, 1, 1)
    AND CAST(`t0`.`file_date` AS DATE) < DATE(2017, 1, 1)
)
SELECT
  `t3`.`file_date`,
  `t3`.`PARTITIONTIME`,
  `t3`.`val`,
  `t3`.`XYZ`
FROM `t1` AS `t3`
INNER JOIN `t1` AS `t5`
  ON TRUE