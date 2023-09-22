WITH t0 AS (
  SELECT
    t5.*
  FROM unbound_table AS t5
  WHERE
    t5.`PARTITIONTIME` < CAST('2017-01-01' AS DATE)
), t1 AS (
  SELECT
    CAST(t0.`file_date` AS DATE) AS `file_date`,
    t0.`PARTITIONTIME`,
    t0.`val`
  FROM t0
), t2 AS (
  SELECT
    t1.*
  FROM t1
  WHERE
    t1.`file_date` < CAST('2017-01-01' AS DATE)
), t3 AS (
  SELECT
    t2.*,
    t2.`val` * 2 AS `XYZ`
  FROM t2
)
SELECT
  t3.*
FROM t3
INNER JOIN t3 AS t4