WITH t0 AS (
  SELECT *
  FROM unbound_table
  WHERE `PARTITIONTIME` < DATE '2017-01-01'
),
t1 AS (
  SELECT CAST(`file_date` AS DATE) AS `file_date`, `PARTITIONTIME`, `val`
  FROM t0
  WHERE `file_date` < DATE '2017-01-01'
),
t2 AS (
  SELECT *, `val` * 2 AS `XYZ`
  FROM t1
)
SELECT t2.*
FROM t2
  INNER JOIN t2 t3