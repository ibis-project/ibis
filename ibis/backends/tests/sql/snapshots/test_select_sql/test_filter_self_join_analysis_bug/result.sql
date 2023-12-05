WITH t0 AS (
  SELECT t2.`region`, t2.`kind`, sum(t2.`amount`) AS `total`
  FROM purchases t2
  WHERE t2.`kind` = 'bar'
  GROUP BY 1, 2
),
t1 AS (
  SELECT t2.`region`, t2.`kind`, sum(t2.`amount`) AS `total`
  FROM purchases t2
  WHERE t2.`kind` = 'foo'
  GROUP BY 1, 2
)
SELECT t1.`region`, t1.`total` - t0.`total` AS `diff`
FROM t1
  INNER JOIN t0
    ON t1.`region` = t0.`region`