WITH t0 AS (
  SELECT t2.*
  FROM bar t2
  WHERE t2.`id` < 3
),
t1 AS (
  SELECT t2.*
  FROM foo t2
  WHERE t2.`id` < 2
)
SELECT t1.`id` AS `left_id`, t1.`desc` AS `left_desc`, t0.`id` AS `right_id`,
       t0.`desc` AS `right_desc`
FROM t1
  LEFT OUTER JOIN t0
    ON (t1.`id` = t0.`id`) AND
       (t1.`desc` = t0.`desc`)