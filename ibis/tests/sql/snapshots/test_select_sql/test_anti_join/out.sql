SELECT t0.*
FROM star1 t0
  LEFT ANTI JOIN star2 t1
    ON t0.`foo_id` = t1.`foo_id`