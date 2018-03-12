#!/usr/bin/env python
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Utility for creating well-formed pull request merges and pushing them to
# Apache.
#   usage: ./apache-pr-merge.py    (see config env vars below)
#
# Lightly modified from version of this script in incubator-parquet-format

from __future__ import print_function

from requests.auth import HTTPBasicAuth
import requests

import os
import subprocess
import sys
import textwrap

from six.moves import input
import six

if __name__ == '__main__':
    IBIS_HOME = os.path.abspath(__file__).rsplit("/", 2)[0]
    PROJECT_NAME = 'ibis'
    print("IBIS_HOME = " + IBIS_HOME)

    # Remote name with the PR
    PR_REMOTE_NAME = os.environ.get("PR_REMOTE_NAME", "upstream")

    # Remote name where results pushed
    PUSH_REMOTE_NAME = os.environ.get("PUSH_REMOTE_NAME", "upstream")

    GITHUB_BASE = "https://github.com/pandas-dev/" + PROJECT_NAME + "/pull"
    GITHUB_API_BASE = "https://api.github.com/repos/pandas-dev/" + PROJECT_NAME

    # Prefix added to temporary branches
    BRANCH_PREFIX = "PR_TOOL"

    os.chdir(IBIS_HOME)

    auth_required = False

    if auth_required:
        GITHUB_USERNAME = os.environ['GITHUB_USER']
        import getpass
        GITHUB_PASSWORD = getpass.getpass('Enter github.com password for %s:'
                                          % GITHUB_USERNAME)

        def get_json_auth(url):
            auth = HTTPBasicAuth(GITHUB_USERNAME, GITHUB_PASSWORD)
            req = requests.get(url, auth=auth)
            return req.json()

        get_json = get_json_auth
    else:
        def get_json_no_auth(url):
            req = requests.get(url)
            return req.json()

        get_json = get_json_no_auth

    def fail(msg):
        print(msg)
        clean_up()
        sys.exit(-1)

    def run_cmd(cmd):
        if isinstance(cmd, six.string_types):
            cmd = cmd.split(' ')

        try:
            output = subprocess.check_output(cmd)
        except subprocess.CalledProcessError as e:
            # this avoids hiding the stdout / stderr of failed processes
            print('Command failed: %s' % cmd)
            print('With output:')
            print('--------------')
            print(e.output)
            print('--------------')
            raise e

        if isinstance(output, six.binary_type):
            output = output.decode('utf-8')
        return output

    def continue_maybe(prompt):
        result = input("\n%s (y/n): " % prompt)
        if result.lower() != "y":
            fail("Okay, exiting")

    original_head = run_cmd("git rev-parse HEAD")[:8]

    def clean_up():
        print("Restoring head pointer to %s" % original_head)
        run_cmd("git checkout %s" % original_head)

        branches = run_cmd("git branch").replace(" ", "").split("\n")

        for branch in filter(lambda x: x.startswith(BRANCH_PREFIX), branches):
            print("Deleting local branch %s" % branch)
            run_cmd("git branch -D %s" % branch)

    # merge the requested PR and return the merge hash
    def merge_pr(pr_num, target_ref):
        pr_branch_name = "%s_MERGE_PR_%s" % (BRANCH_PREFIX, pr_num)
        target_branch_name = "%s_MERGE_PR_%s_%s" % (BRANCH_PREFIX, pr_num,
                                                    target_ref.upper())
        run_cmd("git fetch %s pull/%s/head:%s" % (PR_REMOTE_NAME, pr_num,
                                                  pr_branch_name))
        run_cmd("git fetch %s %s:%s" % (PUSH_REMOTE_NAME, target_ref,
                                        target_branch_name))
        run_cmd("git checkout %s" % target_branch_name)

        had_conflicts = False
        try:
            run_cmd(['git', 'merge', pr_branch_name, '--squash'])
        except Exception as e:
            msg = ("Error merging: %s\nWould you like to "
                   "manually fix-up this merge?" % e)
            continue_maybe(msg)
            msg = ("Okay, please fix any conflicts and 'git add' "
                   "conflicting files... Finished?")
            continue_maybe(msg)
            had_conflicts = True

        commit_authors = run_cmd(['git', 'log', 'HEAD..%s' % pr_branch_name,
                                 '--pretty=format:%an <%ae>']).split("\n")
        distinct_authors = sorted(set(commit_authors),
                                  key=lambda x: commit_authors.count(x),
                                  reverse=True)
        primary_author = distinct_authors[0]
        commits = run_cmd(['git', 'log', 'HEAD..%s' % pr_branch_name,
                          '--pretty=format:%h [%an] %s']).split("\n\n")

        merge_message_flags = []

        merge_message_flags += ["-m", title]
        if body is not None:
            merge_message_flags += ["-m", '\n'.join(textwrap.wrap(body))]

        authors = "\n".join(["Author: %s" % a for a in distinct_authors])

        merge_message_flags += ["-m", authors]

        if had_conflicts:
            committer_name = run_cmd("git config --get user.name").strip()
            committer_email = run_cmd("git config --get user.email").strip()
            message = ("This patch had conflicts when merged, "
                       "resolved by\nCommitter: %s <%s>" %
                       (committer_name, committer_email))
            merge_message_flags += ["-m", message]

        # The string "Closes #%s" string is required for GitHub to correctly
        # close the PR
        merge_message_flags += [
            "-m",
            "Closes #%s from %s and squashes the following commits:"
            % (pr_num, pr_repo_desc)]
        for c in commits:
            merge_message_flags += ["-m", c]

        run_cmd(['git', 'commit',
                 '--no-verify',  # do not run commit hooks
                 '--author="%s"' % primary_author] +
                merge_message_flags)

        continue_maybe("Merge complete (local ref %s). Push to %s?" % (
            target_branch_name, PUSH_REMOTE_NAME))

        try:
            run_cmd('git push %s %s:%s' % (
                PUSH_REMOTE_NAME, target_branch_name, target_ref))
        except Exception as e:
            clean_up()
            fail("Exception while pushing: %s" % e)

        merge_hash = run_cmd("git rev-parse %s" % target_branch_name)[:8]
        clean_up()
        print("Pull request #%s merged!" % pr_num)
        print("Merge hash: %s" % merge_hash)
        return merge_hash

    branches = get_json("%s/branches" % GITHUB_API_BASE)
    branch_names = filter(lambda x: x.startswith("branch-"),
                          [x['name'] for x in branches])

    pr_num = input("Which pull request would you like to merge? (e.g. 34): ")
    pr = get_json("%s/pulls/%s" % (GITHUB_API_BASE, pr_num))

    url = pr["url"]
    title = pr["title"]
    body = pr["body"]
    target_ref = pr["base"]["ref"]
    user_login = pr["user"]["login"]
    base_ref = pr["head"]["ref"]
    pr_repo_desc = "%s/%s" % (user_login, base_ref)

    if pr["merged"] is True:
        print("Pull request {0} has already been merged, assuming "
              "you want to backport".format(pr_num))
        merge_commit_desc = run_cmd([
            'git', 'log', '--merges', '--first-parent',
            '--grep=pull request #%s' % pr_num, '--oneline']).split("\n")[0]
        if merge_commit_desc == "":
            fail("Couldn't find any merge commit for #{0}"
                 ", you may need to update HEAD.".format(pr_num))

        merge_hash = merge_commit_desc[:7]
        message = merge_commit_desc[8:]

        print("Found: %s" % message)
        sys.exit(0)

    if not bool(pr["mergeable"]):
        msg = ("Pull request {0} is not mergeable in its current form.\n"
               "Continue? (experts only!)".format(pr_num))
        continue_maybe(msg)

    print("\n=== Pull Request #%s ===" % pr_num)
    print("title\t%s\nsource\t%s\ntarget\t%s\nurl\t%s" % (
        title, pr_repo_desc, target_ref, url))
    continue_maybe("Proceed with merging pull request #%s?" % pr_num)

    merged_refs = [target_ref]

    merge_hash = merge_pr(pr_num, target_ref)
