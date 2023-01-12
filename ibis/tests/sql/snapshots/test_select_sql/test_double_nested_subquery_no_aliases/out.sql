SELECT t0.`key1`, sum(t0.`total`) AS `total`
FROM (
  SELECT t1.`key1`, t1.`key2`, sum(t1.`total`) AS `total`
  FROM (
    SELECT t2.`key1`, t2.`key2`, t2.`key3`, sum(t2.`value`) AS `total`
    FROM foo_table t2
    GROUP BY 1, 2, 3
  ) t1
  GROUP BY 1, 2
) t0
GROUP BY 1