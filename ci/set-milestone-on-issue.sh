#!/usr/bin/env bash

set -euo pipefail

top="$(dirname "$(readlink -f "$0")")"

# find all pull requests associated with commit
linked_issues_query='.data.repository.pullRequest.closingIssuesReferences.nodes[].number'

gh pr list --search "$1" --state merged --json number --jq '.[].number' |
  sed '/^$/d' |
  while read -r pr; do
    milestone="$(gh pr view "${pr}" --json milestone --jq '.milestone.title')"

    if [ -n "${milestone}" ]; then
      # find all issues associated with said pull requests
      # taken from https://github.com/cli/cli/discussions/7097#discussioncomment-5229031
      readarray -t issues < <(
        gh api graphql -F owner=ibis-project -F repo=ibis -F pr="${pr}" -F query="@${top}/linked-issues.gql" \
        --jq "${linked_issues_query}" | sed '/^$/d')

      if [ "${#issues[@]}" -gt 0 ]; then
        gh issue edit "${issues[@]}" --milestone "${milestone}"
      fi
    fi
  done
