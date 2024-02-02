SELECT
  `t0`.`title`,
  `t0`.`tags`
FROM `bigquery-public-data`.`stackoverflow`.`posts_questions` AS `t0`
WHERE
  strpos(`t0`.`tags`, 'ibis') > 0