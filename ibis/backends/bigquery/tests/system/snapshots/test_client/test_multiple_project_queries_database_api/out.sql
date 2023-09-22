SELECT
  t0.`title`
FROM `bigquery-public-data`.stackoverflow.posts_questions AS t0
INNER JOIN `nyc-tlc`.yellow.trips AS t1
  ON t0.`tags` = t1.`rate_code`