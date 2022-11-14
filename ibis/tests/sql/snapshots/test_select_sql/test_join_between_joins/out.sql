SELECT t0.*, t1.`value3`, t1.`value4`
FROM (
  SELECT t2.*, t3.`value2`
  FROM `first` t2
    INNER JOIN second t3
      ON t2.`key1` = t3.`key1`
) t0
  INNER JOIN (
    SELECT t2.*, t3.`value4`
    FROM third t2
      INNER JOIN fourth t3
        ON t2.`key3` = t3.`key3`
  ) t1
    ON t0.`key2` = t1.`key2`