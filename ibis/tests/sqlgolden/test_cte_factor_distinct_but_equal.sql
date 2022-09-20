WITH t0 AS (
  SELECT `g`, sum(`f`) AS `metric`
  FROM alltypes
  GROUP BY 1
)
SELECT t0.*
FROM t0
  INNER JOIN t0 t1
    ON t0.`g` = t1.`g`
