SELECT t0.`id` AS `left_id`, t0.`desc` AS `left_desc`, t1.`id` AS `right_id`,
       t1.`desc` AS `right_desc`
FROM (
  SELECT t2.*
  FROM foo t2
  WHERE t2.`id` < 2
) t0
  LEFT OUTER JOIN (
    SELECT t2.*
    FROM bar t2
    WHERE t2.`id` < 3
  ) t1
    ON (t0.`id` = t1.`id`) AND
       (t0.`desc` = t1.`desc`)