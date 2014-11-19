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

import traceback


class Task(object):
    """
    Prototype

    Run task in a thread, capture tracebacks or other problems.
    """
    def __init__(self, worker):
        self.worker = worker
        self.complete = False

    def start(self):
        pass

    def run(self):
        raise NotImplementedError

    def done(self):
        pass


#----------------------------------------------------------------------
# UDA execution tasks


class AggregationTask(Task):
    """

    """
    pass


class MergeTask(Task):
    pass


class FinalizeTask(Task):
    pass
