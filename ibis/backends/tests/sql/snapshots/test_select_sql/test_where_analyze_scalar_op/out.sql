SELECT count(1) AS `CountStar()`
FROM functional_alltypes t0
WHERE (t0.`timestamp_col` < date_add(cast('2010-01-01T00:00:00' as timestamp), INTERVAL 3 MONTH)) AND
      (t0.`timestamp_col` < date_add(cast(now() as timestamp), INTERVAL 10 DAY))