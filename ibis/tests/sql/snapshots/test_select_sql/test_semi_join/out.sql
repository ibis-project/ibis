SELECT t0.*
FROM star1 t0
  LEFT SEMI JOIN star2 t1
    ON t0.`foo_id` = t1.`foo_id`