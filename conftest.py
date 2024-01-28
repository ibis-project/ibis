from __future__ import annotations

import os

import hypothesis as h

# setup hypothesis profiles
h.settings.register_profile(
    "ci",
    max_examples=1000,
    suppress_health_check=[h.HealthCheck.too_slow],
    deadline=None,
)
h.settings.register_profile(
    "dev",
    max_examples=50,
    suppress_health_check=[h.HealthCheck.too_slow],
    deadline=None,
)
h.settings.register_profile(
    "debug",
    max_examples=10,
    verbosity=h.Verbosity.verbose,
    suppress_health_check=[h.HealthCheck.too_slow],
    deadline=None,
)

# load default hypothesis profile, either set HYPOTHESIS_PROFILE environment
# variable or pass --hypothesis-profile option to pytest, to see the generated
# examples try:
# pytest pyarrow -sv --hypothesis-profile=debug
h.settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
