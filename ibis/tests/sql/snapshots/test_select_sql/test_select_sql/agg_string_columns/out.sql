SELECT `foo_id` AS `foo_id`, `bar_id` AS `bar_id`, sum(`f`) AS `total`
FROM star1
GROUP BY 1, 2