WITH t0 AS (
  SELECT t3.`a`, t3.`g`, sum(t3.`f`) AS `metric`
  FROM alltypes t3
  GROUP BY 1, 2
),
t1 AS (
  SELECT t0.*
  FROM t0
    INNER JOIN t0 t3
      ON t0.`g` = t3.`g`
)
SELECT t2.`a`, t2.`g`, t2.`metric`
FROM (
  WITH t0 AS (
    SELECT t3.`a`, t3.`g`, sum(t3.`f`) AS `metric`
    FROM alltypes t3
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
) t2