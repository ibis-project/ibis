SELECT count(1) AS `count`
FROM functional_alltypes
WHERE (`timestamp_col` < date_add(cast('2010-01-01T00:00:00' as timestamp), INTERVAL 3 MONTH)) AND
      (`timestamp_col` < date_add(cast(now() as timestamp), INTERVAL 10 DAY))