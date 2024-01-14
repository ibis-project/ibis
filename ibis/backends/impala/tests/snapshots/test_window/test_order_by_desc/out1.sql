SELECT
  `t0`.`f`,
  ROW_NUMBER() OVER (ORDER BY `t0`.`f` DESC) - 1 AS `revrank`
FROM `alltypes` AS `t0`