DROP TABLE IF EXISTS ibis_testing.dbo.diamonds;

CREATE TABLE ibis_testing.dbo.diamonds (
    carat FLOAT,
    cut VARCHAR(MAX),
    color VARCHAR(MAX),
    clarity VARCHAR(MAX),
    depth FLOAT,
    "table" FLOAT,
    price BIGINT,
    x FLOAT,
    y FLOAT,
    z FLOAT
);


-- /data is a volume mount to the ibis testing data
-- used for snappy test data loading
-- DataFrame.to_sql is unusably slow for loading CSVs
BULK INSERT ibis_testing.dbo.diamonds
FROM '/data/diamonds.csv'
WITH (FORMAT = 'CSV', FIELDTERMINATOR = ',', ROWTERMINATOR = '\n', FIRSTROW = 2)

DROP TABLE IF EXISTS ibis_testing.dbo.astronauts;

CREATE TABLE ibis_testing.dbo.astronauts (
    "id" BIGINT,
    "number" BIGINT,
    "nationwide_number" BIGINT,
    "name" VARCHAR(MAX),
    "original_name" VARCHAR(MAX),
    "sex" VARCHAR(MAX),
    "year_of_birth" BIGINT,
    "nationality" VARCHAR(MAX),
    "military_civilian" VARCHAR(MAX),
    "selection" VARCHAR(MAX),
    "year_of_selection" BIGINT,
    "mission_number" BIGINT,
    "total_number_of_missions" BIGINT,
    "occupation" VARCHAR(MAX),
    "year_of_mission" BIGINT,
    "mission_title" VARCHAR(MAX),
    "ascend_shuttle" VARCHAR(MAX),
    "in_orbit" VARCHAR(MAX),
    "descend_shuttle" VARCHAR(MAX),
    "hours_mission" DOUBLE PRECISION,
    "total_hrs_sum" DOUBLE PRECISION,
    "field21" BIGINT,
    "eva_hrs_mission" DOUBLE PRECISION,
    "total_eva_hrs" DOUBLE PRECISION
);

BULK INSERT ibis_testing.dbo.astronauts
FROM '/data/astronauts.csv'
WITH (FORMAT = 'CSV', FIELDTERMINATOR = ',', ROWTERMINATOR = '\n', FIRSTROW = 2)

DROP TABLE IF EXISTS ibis_testing.dbo.batting;

CREATE TABLE ibis_testing.dbo.batting (
    "playerID" VARCHAR(MAX),
    "yearID" BIGINT,
    stint BIGINT,
    "teamID" VARCHAR(MAX),
    "lgID" VARCHAR(MAX),
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

BULK INSERT ibis_testing.dbo.batting
FROM '/data/batting.csv'
WITH (FORMAT = 'CSV', FIELDTERMINATOR = ',', ROWTERMINATOR = '\n', FIRSTROW = 2)

DROP TABLE IF EXISTS ibis_testing.dbo.awards_players;

CREATE TABLE ibis_testing.dbo.awards_players (
    "playerID" VARCHAR(MAX),
    "awardID" VARCHAR(MAX),
    "yearID" BIGINT,
    "lgID" VARCHAR(MAX),
    tie VARCHAR(MAX),
    notes VARCHAR(MAX)
);

BULK INSERT ibis_testing.dbo.awards_players
FROM '/data/awards_players.csv'
WITH (FORMAT = 'CSV', FIELDTERMINATOR = ',', ROWTERMINATOR = '\n', FIRSTROW = 2)

DROP TABLE IF EXISTS ibis_testing.dbo.functional_alltypes;

CREATE TABLE ibis_testing.dbo.functional_alltypes (
    id INTEGER,
    bool_col BIT,
    tinyint_col SMALLINT,
    smallint_col SMALLINT,
    int_col INTEGER,
    bigint_col BIGINT,
    float_col REAL,
    double_col DOUBLE PRECISION,
    date_string_col VARCHAR(MAX),
    string_col VARCHAR(MAX),
    timestamp_col DATETIME2,
    year INTEGER,
    month INTEGER
);

BULK INSERT ibis_testing.dbo.functional_alltypes
FROM '/data/functional_alltypes.csv'
WITH (FORMAT = 'CSV', FIELDTERMINATOR = ',', ROWTERMINATOR = '\n', FIRSTROW = 2)

DROP TABLE IF EXISTS ibis_testing.dbo.win;

CREATE TABLE ibis_testing.dbo.win (g VARCHAR(MAX), x BIGINT NOT NULL, y BIGINT);
INSERT INTO ibis_testing.dbo.win VALUES
    ('a', 0, 3),
    ('a', 1, 2),
    ('a', 2, 0),
    ('a', 3, 1),
    ('a', 4, 1);

DROP TABLE IF EXISTS ibis_testing.dbo.topk;

CREATE TABLE ibis_testing.dbo.topk (x BIGINT);
INSERT INTO ibis_testing.dbo.topk VALUES (1), (1), (NULL);
