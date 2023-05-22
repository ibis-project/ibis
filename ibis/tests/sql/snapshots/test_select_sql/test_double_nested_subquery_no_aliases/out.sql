WITH t0 AS (
  SELECT t2.`key1`, t2.`key2`, t2.`key3`, sum(t2.`value`) AS `total`
  FROM foo_table t2
  GROUP BY 1, 2, 3
)
SELECT t1.`key1`, sum(t1.`total`) AS `total`
FROM (
  SELECT t0.`key1`, t0.`key2`, sum(t0.`total`) AS `total`
  FROM t0
  GROUP BY 1, 2
) t1
GROUP BY 1