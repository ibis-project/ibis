# Copyright 2014 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file may adapt small portions of https://github.com/mtth/hdfs (MIT
# license), see the LICENSES directory.

from __future__ import annotations

from typing import Any

import fsspec

from ibis import util


@util.deprecated(
    instead="Use fsspec.filesystem(\"webhdfs\", ...) directly",
    version="3.0.0",
)
def hdfs_connect(
    *args: Any,
    protocol: str = "webhdfs",
    **kwargs: Any,
) -> fsspec.spec.AbstractFileSystem:
    """Connect to HDFS using `fsspec`.

    This function is a thing wrapper around `fsspec.filesystem`. All arguments
    are forwarded to that API.
    """

    return fsspec.filesystem(protocol, *args, **kwargs)
