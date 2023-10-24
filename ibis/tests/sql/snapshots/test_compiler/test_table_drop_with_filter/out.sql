WITH t0 AS (
  SELECT t3.`a`, t3.`b`, t3.`c` AS `C`
  FROM t t3
),
t1 AS (
  SELECT t0.`a`, t0.`b`, '2018-01-01T00:00:00' AS `the_date`
  FROM t0
  WHERE t0.`C` = '2018-01-01T00:00:00'
)
SELECT t2.*
FROM (
  SELECT t1.`a`
  FROM t1
    INNER JOIN s t3
      ON t1.`b` = t3.`b`
) t2
WHERE t2.`a` < 1.0