WITH t0 AS (
  SELECT t3.`region`, t3.`kind`, sum(t3.`amount`) AS `total`
  FROM purchases t3
  GROUP BY 1, 2
)
SELECT t1.`region`, t1.`total` - t2.`total` AS `diff`
FROM (
  SELECT t0.*
  FROM t0
  WHERE t0.`kind` = 'foo'
) t1
  INNER JOIN (
    SELECT t0.*
    FROM t0
    WHERE t0.`kind` = 'bar'
  ) t2
    ON t1.`region` = t2.`region`