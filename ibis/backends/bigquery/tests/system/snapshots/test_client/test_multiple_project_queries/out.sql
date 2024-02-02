SELECT
  `t2`.`title`
FROM `bigquery-public-data`.`stackoverflow`.`posts_questions` AS `t2`
INNER JOIN `nyc-tlc`.`yellow`.`trips` AS `t3`
  ON `t2`.`tags` = `t3`.`rate_code`