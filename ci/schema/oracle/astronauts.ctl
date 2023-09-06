options (SKIP=1)
load data
  infile '/opt/oracle/data/astronauts.csv'
  into table "astronauts"
  fields csv with embedded
  ("id",
    "number",
    "nationwide_number",
    "name",
    "original_name",
    "sex",
    "year_of_birth",
    "nationality",
    "military_civilian",
    "selection",
    "year_of_selection",
    "mission_number",
    "total_number_of_missions",
    "occupation",
    "year_of_mission",
    "mission_title",
    "ascend_shuttle",
    "in_orbit",
    "descend_shuttle",
    "hours_mission",
    "total_hrs_sum",
    "field21",
    "eva_hrs_mission",
    "total_eva_hrs")
