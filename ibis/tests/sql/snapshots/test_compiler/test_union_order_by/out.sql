SELECT t0.`a`, t0.`b`
FROM (
  WITH t1 AS (
    SELECT t2.*
    FROM t t2
    ORDER BY t2.`b` ASC
  )
  SELECT *
  FROM t1
  UNION ALL
  SELECT *
  FROM t1
) t0