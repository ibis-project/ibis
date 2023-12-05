WITH t0 AS (
  SELECT t4.`a`, t4.`b`, t4.`c` AS `C`
  FROM t t4
),
t1 AS (
  SELECT t0.*
  FROM t0
  WHERE t0.`C` = '2018-01-01T00:00:00'
),
t2 AS (
  SELECT t1.`a`, t1.`b`, '2018-01-01T00:00:00' AS `the_date`
  FROM t1
)
SELECT t3.*
FROM (
  SELECT t2.`a`
  FROM t2
    INNER JOIN s t4
      ON t2.`b` = t4.`b`
) t3
WHERE t3.`a` < 1.0