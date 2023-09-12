SELECT t0.*
FROM `t` t0
  INNER JOIN `t` t1
    ON (t0.`a` = t1.`a`) AND
       (((t0.`a` != t1.`b`) OR (t0.`b` != t1.`a`)) AND NOT ((t0.`a` != t1.`b`) AND (t0.`b` != t1.`a`)))