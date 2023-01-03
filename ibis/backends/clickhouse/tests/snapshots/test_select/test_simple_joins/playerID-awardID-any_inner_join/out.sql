SELECT
  t0.*
FROM batting AS t0
ANY JOIN awards_players AS t1
  ON t0.playerID = t1.awardID