SELECT
  t0.id IN (
    SELECT
      arrayJoin(t0.ids) AS ids
    FROM way_view AS t0
  ) AS "Contains(id, ids)"
FROM node_view AS t0