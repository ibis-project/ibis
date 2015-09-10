import ibis
import os
import pandas


def wrangle_csvs():
    years = range(1987, 2009)

    for year in years:
        path = '%d.csv.bz2' % year
        outpath = os.path.expanduser('~/data/%d_clean.csv' % year)

        print 'Working on %s' % path

        df = pandas.read_csv(path, compression='bz2')
        df.to_csv(outpath, header=False, index=False,
                  float_format='%g', na_rep='\N')



schema = ibis.schema([
    ('year', 'int32'),
    ('month', 'int8'),
    ('day', 'int8'),
    ('dayofweek', 'int8'),
    ('dep_time', 'int32'),
    ('crs_dep_time', 'int32'),
    ('arr_time', 'int32'),
    ('crs_arr_time', 'int32'),
    ('carrier', 'string'),
    ('flight_num', 'int32'),
    ('tail_num', 'int32'),
    ('actual_elapsed_time', 'int32'),
    ('crs_elapsed_time', 'int32'),
    ('airtime', 'int32'),
    ('arrdelay', 'int32'),
    ('depdelay', 'int32'),
    ('origin', 'string'),
    ('dest', 'string'),
    ('distance', 'int32'),
    ('taxi_in', 'int32'),
    ('taxi_out', 'int32'),
    ('cancelled', 'int8'),
    ('cancellation_code', 'string'),
    ('diverted', 'int8'),
    ('carrier_delay', 'int32'),
    ('weather_delay', 'int32'),
    ('nas_delay', 'int32'),
    ('security_delay', 'int32'),
    ('late_aircraft_delay', 'int32')
])
