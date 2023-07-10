SELECT
  t0.playerID AS playerID,
  t0.yearID AS yearID,
  t0.stint AS stint,
  t0.teamID AS teamID,
  t0.lgID AS lgID,
  t0.G AS G,
  t0.AB AS AB,
  t0.R AS R,
  t0.H AS H,
  t0.X2B AS X2B,
  t0.X3B AS X3B,
  t0.HR AS HR,
  t0.RBI AS RBI,
  t0.SB AS SB,
  t0.CS AS CS,
  t0.BB AS BB,
  t0.SO AS SO,
  t0.IBB AS IBB,
  t0.HBP AS HBP,
  t0.SH AS SH,
  t0.SF AS SF,
  t0.GIDP AS GIDP
FROM batting AS t0
LEFT ANY JOIN awards_players AS t2
  ON t0.playerID = t2.playerID