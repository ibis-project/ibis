SELECT
  t0.`title`,
  t0.`tags`
FROM (
  SELECT
    t1.*
  FROM `bigquery-public-data`.stackoverflow.posts_questions AS t1
  WHERE
    STRPOS(t1.`tags`, 'ibis') - 1 >= 0
) AS t0