#!/usr/bin/env bash
set -euo pipefail

STARROCKS_FLAKE_REF="${STARROCKS_FLAKE_REF:-github:ascii-supply-networks/flaky-stars-on-the-rocks/dc5391f572431e7151e2120898be112bb5ed29d0}"

root="${RUNNER_TEMP:-/tmp}/ibis-starrocks"
fe_state="$root/fe"
be_state="$root/be-0"
mysql_args=(--connect-timeout=2 -h 127.0.0.1 -P9030 -uroot)

show_logs() {
  if [[ -f "$root/fe.log" ]]; then
    echo "::group::StarRocks FE log"
    tail -n 200 "$root/fe.log" || true
    echo "::endgroup::"
  fi

  if [[ -f "$root/be.log" ]]; then
    echo "::group::StarRocks BE log"
    tail -n 200 "$root/be.log" || true
    echo "::endgroup::"
  fi
}

ensure_running() {
  local pid_file="$1"
  local name="$2"

  if [[ -f "$pid_file" ]] && ! kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$name exited before becoming ready" >&2
    show_logs
    return 1
  fi
}

wait_for_sql() {
  local sql="$1"
  local description="$2"

  for attempt in $(seq 1 120); do
    if mysql "${mysql_args[@]}" --skip-column-names --batch -e "$sql" >/dev/null 2>&1; then
      return 0
    fi

    ensure_running "$root/fe.pid" "StarRocks FE"

    if [[ "$attempt" -eq 120 ]]; then
      echo "Timed out waiting for StarRocks: $description" >&2
      show_logs
      return 1
    fi

    sleep 2
  done
}

wait_for_backend() {
  for attempt in $(seq 1 120); do
    if mysql "${mysql_args[@]}" --skip-column-names --batch -e "SHOW BACKENDS;" \
      | grep -F "9050" >/dev/null; then
      return 0
    fi

    ensure_running "$root/be.pid" "StarRocks BE"

    if [[ "$attempt" -eq 120 ]]; then
      echo "Timed out waiting for StarRocks BE registration" >&2
      show_logs
      return 1
    fi

    sleep 2
  done
}

rm -rf "$root"
mkdir -p "$fe_state/log" "$fe_state/meta" "$be_state/log" "$be_state/storage"

pkg="$(nix build --no-link --print-out-paths "$STARROCKS_FLAKE_REF#starrocks")"

"$pkg/bin/starrocks-prepare-runtime" fe "$fe_state"
"$pkg/bin/starrocks-prepare-runtime" be "$be_state"

cat > "$fe_state/home/conf/fe.conf" <<EOF
LOG_DIR = $fe_state/log
DATE = "\$(date +%Y%m%d-%H%M%S)"
JAVA_OPTS="-Dlog4j2.formatMsgNoLookups=true -Xmx1024m -XX:+UseG1GC -XX:ErrorFile=$fe_state/log/hs_err_pid%p.log -Djava.security.policy=$fe_state/home/conf/udf_security.policy"

sys_log_level = INFO
http_port = 8030
rpc_port = 9020
query_port = 9030
edit_log_port = 9010
mysql_service_nio_enabled = true
meta_dir = $fe_state/meta
sys_log_dir = $fe_state/log
audit_log_dir = $fe_state/log
EOF

cat > "$be_state/home/conf/be.conf" <<EOF
sys_log_level = INFO
be_port = 9060
be_http_port = 8040
heartbeat_service_port = 9050
brpc_port = 8060
starlet_port = 9070
storage_root_path = $be_state/storage
sys_log_dir = $be_state/log
EOF

(
  cd "$fe_state/home"
  nohup "$pkg/bin/starrocks-fe" --host_type FQDN --logconsole >"$root/fe.log" 2>&1 &
  echo "$!" >"$root/fe.pid"
)

wait_for_sql "SELECT 1;" "FE MySQL protocol"

mysql "${mysql_args[@]}" -e 'ALTER SYSTEM ADD BACKEND "127.0.0.1:9050";' || true

(
  cd "$be_state/home"
  nohup "$pkg/bin/starrocks-be" --be --logconsole >"$root/be.log" 2>&1 &
  echo "$!" >"$root/be.pid"
)

wait_for_backend
mysql "${mysql_args[@]}" -e "CREATE DATABASE IF NOT EXISTS ibis_testing;"

echo "StarRocks is ready on 127.0.0.1:9030"
