SELECT
  FROM_UNIXTIME(UNIX_TIMESTAMP(CAST(`t0`.`timestamp_col` AS STRING)), 'MM/dd/yyyy') AS `Strftime(timestamp_col, '%m/%d/%Y')`
FROM `functional_alltypes` AS `t0`