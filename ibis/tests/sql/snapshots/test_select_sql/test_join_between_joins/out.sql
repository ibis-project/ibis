WITH t0 AS (
  SELECT t2.*, t3.`value4`
  FROM third t2
    INNER JOIN fourth t3
      ON t2.`key3` = t3.`key3`
),
t1 AS (
  SELECT t2.*, t3.`value2`
  FROM first t2
    INNER JOIN second t3
      ON t2.`key1` = t3.`key1`
)
SELECT t1.*, t0.`value3`, t0.`value4`
FROM t1
  INNER JOIN t0
    ON t1.`key2` = t0.`key2`