WITH t0 AS (
  SELECT t2.`a`, t2.`b`, '2018-01-01T00:00:00' AS `the_date`
  FROM t t2
  WHERE t2.`c` = '2018-01-01T00:00:00'
)
SELECT t0.`a`
FROM t0
  INNER JOIN s t1
    ON t0.`b` = t1.`b`
WHERE t0.`a` < 1.0