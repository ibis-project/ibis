import os

import hypothesis as h

# setup hypothesis profiles
h.settings.register_profile('ci', max_examples=1000)
h.settings.register_profile('dev', max_examples=50)
h.settings.register_profile('debug', max_examples=10, verbosity=h.Verbosity.verbose)

# load default hypothesis profile, either set HYPOTHESIS_PROFILE environment
# variable or pass --hypothesis-profile option to pytest, to see the generated
# examples try:
# pytest pyarrow -sv --hypothesis-profile=debug
h.settings.load_profile(os.environ.get('HYPOTHESIS_PROFILE', 'dev'))
