import fnmatch
import os
import sys

collect_ignore = ['setup.py']

if sys.version_info.major == 2:
    this_directory = os.path.dirname(__file__)
    bigquery_udf = os.path.join(this_directory, 'ibis', 'bigquery', 'udf')
    for root, _, filenames in os.walk(bigquery_udf):
        for filename in filenames:
            if fnmatch.fnmatch(filename, '*.py'):
                collect_ignore.append(os.path.join(root, filename))
