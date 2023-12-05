WITH t0 AS (
  SELECT t2.`g`, t2.`a`, t2.`b`, sum(t2.`f`) AS `total`
  FROM alltypes t2
  GROUP BY 1, 2, 3
)
SELECT t0.`g`, max(t0.`total` - `total`) AS `metric`
FROM (
  SELECT t0.`g`, t0.`a`, t0.`b`, t0.`total`, t2.`g` AS `g_right`,
         t2.`a` AS `a_right`, t2.`b` AS `b_right`,
         t2.`total` AS `total_right`
  FROM t0
    INNER JOIN t0 t2
      ON t0.`a` = t2.`b`
) t1
GROUP BY 1