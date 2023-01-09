WITH t0 AS (
  SELECT `region` AS `region`, `kind` AS `kind`, sum(`amount`) AS `total`
  FROM purchases
  GROUP BY 1, 2
)
SELECT t1.`region` AS `region`, t1.`total` - t2.`total` AS `diff`
FROM (
  SELECT *
  FROM t0
  WHERE `kind` = 'foo'
) t1
  INNER JOIN (
    SELECT *
    FROM t0
    WHERE `kind` = 'bar'
  ) t2
    ON t1.`region` = t2.`region`