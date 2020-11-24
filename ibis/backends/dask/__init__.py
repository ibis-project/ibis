from __future__ import absolute_import

import ibis.config

with ibis.config.config_prefix('dask'):
    ibis.config.register_option(
        'enable_trace',
        False,
        'Whether enable tracing for dask execution. '
        'See ibis.dask.trace for details.',
        validator=ibis.config.is_bool,
    )
