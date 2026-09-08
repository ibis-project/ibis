-- IBM Db2 schema for ibis test suite
-- Executed statement-by-statement by conftest._load_data()
-- All column names are double-quoted to preserve exact case for inserts.

DROP TABLE IF EXISTS diamonds;

CREATE TABLE diamonds (
    "carat" DOUBLE,
    "cut" VARCHAR(255),
    "color" VARCHAR(255),
    "clarity" VARCHAR(255),
    "depth" DOUBLE,
    "table" DOUBLE,
    "price" BIGINT,
    "x" DOUBLE,
    "y" DOUBLE,
    "z" DOUBLE
);

DROP TABLE IF EXISTS astronauts;

CREATE TABLE astronauts (
    "id" BIGINT,
    "number" BIGINT,
    "nationwide_number" BIGINT,
    "name" VARCHAR(255),
    "original_name" VARCHAR(255),
    "sex" VARCHAR(255),
    "year_of_birth" BIGINT,
    "nationality" VARCHAR(255),
    "military_civilian" VARCHAR(255),
    "selection" VARCHAR(255),
    "year_of_selection" BIGINT,
    "mission_number" BIGINT,
    "total_number_of_missions" BIGINT,
    "occupation" VARCHAR(255),
    "year_of_mission" BIGINT,
    "mission_title" VARCHAR(255),
    "ascend_shuttle" VARCHAR(255),
    "in_orbit" VARCHAR(255),
    "descend_shuttle" VARCHAR(255),
    "hours_mission" DOUBLE,
    "total_hrs_sum" DOUBLE,
    "field21" BIGINT,
    "eva_hrs_mission" DOUBLE,
    "total_eva_hrs" DOUBLE
);

DROP TABLE IF EXISTS batting;

CREATE TABLE batting (
    "playerID" VARCHAR(255),
    "yearID" BIGINT,
    "stint" BIGINT,
    "teamID" VARCHAR(7),
    "lgID" VARCHAR(7),
    "G" BIGINT,
    "AB" BIGINT,
    "R" BIGINT,
    "H" BIGINT,
    "X2B" BIGINT,
    "X3B" BIGINT,
    "HR" BIGINT,
    "RBI" BIGINT,
    "SB" BIGINT,
    "CS" BIGINT,
    "BB" BIGINT,
    "SO" BIGINT,
    "IBB" BIGINT,
    "HBP" BIGINT,
    "SH" BIGINT,
    "SF" BIGINT,
    "GIDP" BIGINT
);

DROP TABLE IF EXISTS awards_players;

CREATE TABLE awards_players (
    "playerID" VARCHAR(255),
    "awardID" VARCHAR(255),
    "yearID" BIGINT,
    "lgID" VARCHAR(7),
    "tie" VARCHAR(7),
    "notes" VARCHAR(255)
);

DROP TABLE IF EXISTS functional_alltypes;

CREATE TABLE functional_alltypes (
    "id" INTEGER,
    "bool_col" SMALLINT,
    "tinyint_col" SMALLINT,
    "smallint_col" SMALLINT,
    "int_col" INTEGER,
    "bigint_col" BIGINT,
    "float_col" REAL,
    "double_col" DOUBLE,
    "date_string_col" VARCHAR(255),
    "string_col" VARCHAR(255),
    "timestamp_col" TIMESTAMP,
    "year" INTEGER,
    "month" INTEGER
);

DROP TABLE IF EXISTS win;

CREATE TABLE win ("g" VARCHAR(8), "x" BIGINT NOT NULL, "y" BIGINT);

INSERT INTO win VALUES ('a', 0, 3);
INSERT INTO win VALUES ('a', 1, 2);
INSERT INTO win VALUES ('a', 2, 0);
INSERT INTO win VALUES ('a', 3, 1);
INSERT INTO win VALUES ('a', 4, 1);

DROP TABLE IF EXISTS topk;

CREATE TABLE topk ("x" BIGINT);

INSERT INTO topk VALUES (1);
INSERT INTO topk VALUES (1);
INSERT INTO topk VALUES (NULL);
