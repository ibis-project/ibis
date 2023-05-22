WITH t0 AS (
  SELECT t3.`region`, t3.`kind`, sum(t3.`amount`) AS `total`
  FROM purchases t3
  GROUP BY 1, 2
),
t1 AS (
  SELECT t0.*
  FROM t0
  WHERE t0.`kind` = 'bar'
),
t2 AS (
  SELECT t0.*
  FROM t0
  WHERE t0.`kind` = 'foo'
)
SELECT t2.`region`, t2.`total` - t1.`total` AS `diff`
FROM t2
  INNER JOIN t1
    ON t2.`region` = t1.`region`