WITH t0 AS (
  SELECT `a`, `g`, sum(`f`) AS `metric`
  FROM alltypes
  GROUP BY 1, 2
),
t1 AS (
  SELECT t0.*
  FROM t0
    INNER JOIN t0 t3
      ON t0.`g` = t3.`g`
)
SELECT *
FROM t1
UNION ALL
SELECT t0.*
FROM t0
  INNER JOIN t0 t3
    ON t0.`g` = t3.`g`