WITH t0 AS (
  SELECT t2.`a`, t2.`g`, sum(t2.`f`) AS `metric`
  FROM alltypes t2
  GROUP BY 1, 2
),
t1 AS (
  SELECT t0.*
  FROM t0
    INNER JOIN t0 t2
      ON t0.`g` = t2.`g`
)
SELECT *
FROM t1
UNION ALL
SELECT t0.*
FROM t0
  INNER JOIN t0 t2
    ON t0.`g` = t2.`g`