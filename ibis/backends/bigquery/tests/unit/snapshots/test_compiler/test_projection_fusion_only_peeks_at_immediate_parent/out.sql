WITH t0 AS (
  SELECT t4.*
  FROM unbound_table t4
  WHERE t4.`PARTITIONTIME` < DATE '2017-01-01'
),
t1 AS (
  SELECT CAST(t0.`file_date` AS DATE) AS `file_date`, t0.`PARTITIONTIME`,
         t0.`val`
  FROM t0
  WHERE t0.`file_date` < DATE '2017-01-01'
),
t2 AS (
  SELECT t1.*, t1.`val` * 2 AS `XYZ`
  FROM t1
)
SELECT t2.*
FROM t2
  INNER JOIN t2 t3