SELECT
  CAST(FROM_UNIXTIME(CAST(`t0`.`c` AS INT), 'yyyy-MM-dd HH:mm:ss') AS TIMESTAMP) AS `TimestampFromUNIX(c, SECOND)`
FROM `alltypes` AS `t0`