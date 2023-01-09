SELECT `key1` AS `key1`, sum(`total`) AS `total`
FROM (
  SELECT `key1` AS `key1`, `key2` AS `key2`, sum(`total`) AS `total`
  FROM (
    SELECT `key1` AS `key1`, `key2` AS `key2`, `key3` AS `key3`,
           sum(`value`) AS `total`
    FROM foo_table
    GROUP BY 1, 2, 3
  ) t1
  GROUP BY 1, 2
) t0
GROUP BY 1