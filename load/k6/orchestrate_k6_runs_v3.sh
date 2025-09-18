#!/usr/bin/env bash
set -euo pipefail

# ========== Config ==========
K6_BIN="${K6_BIN:-k6}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
SCRIPTS_DIR="$ROOT_DIR/load/k6"
LOG_DIR="$ROOT_DIR/k6_runs_logs"
mkdir -p "$LOG_DIR"

ts() { date -u +"%Y%m%dT%H%M%SZ"; }

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") run A|B|C|D|all
  $(basename "$0") tail <name|all>
  $(basename "$0") stop <name|all>

Names:
  A           benign-breeze (standalone)
  B           attack-burst in parallel with benign-breeze (combo)
  C           benign-surge (standalone)
  D           combo benign-breeze + attack-burst (like B) with own labels
  all         run D and C (concurrently)

Environment (common):
  CF_DOMAIN or BASE_URL; BENIGN_ROUTES, ATTACKY_ROUTES (comma separated)
  WARM_MINS/BREEZE_MINS/COOL_MINS and WARM_RPS/BREEZE_RPS/COOL_RPS
  ATT_* (see scripts), SURGE_* (see scripts)
  ATT_START_OFFSET_MINS (for combos B/D), default: 15
  VU_SAFETY_FACTOR, VU_MAX_CAP, EST_LAT_MS, THINK_MS

Examples:
  CF_DOMAIN=example.cloudfront.net $(basename "$0") run A
  BASE_URL=https://my-alb.example.com ATT_START_OFFSET_MINS=15 $(basename "$0") run D
USAGE
}

start_bg() {
  local label="$1" script="$2"
  local now="$(ts)"
  local log="$LOG_DIR/${label}_${now}.log"
  local json="$LOG_DIR/${label}_${now}.json"
  local pidf="$LOG_DIR/${label}.pid"

  echo "[*] Starting $label → $script"
  echo "    Logs: $log"
  echo "    JSON: $json"
  set -x
  nohup "$K6_BIN" run --out "json=$json" "$script" >"$log" 2>&1 &
  set +x
  local pid=$!
  echo "$pid" > "$pidf"
  echo "[*] PID of $label: $pid (stored in $pidf)"
}

run_A() { start_bg "A_benign_breeze" "$SCRIPTS_DIR/benign-breeze.js"; }

run_C() { start_bg "C_benign_surge" "$SCRIPTS_DIR/benign-surge.js"; }

# Combo: benign now + attack after offset
run_combo() {
  local label="$1" # AB or D
  local offset="${ATT_START_OFFSET_MINS:-15}"
  echo "[*] Combo $label: starting benign-breeze now, attack-burst after ${offset}m"
  start_bg "${label}_A_benign_breeze" "$SCRIPTS_DIR/benign-breeze.js"
  # human-friendly wait banner
  echo "[*] Sleeping ${offset} minutes before starting attack..."
  sleep "$((offset*60))"
  start_bg "${label}_B_attack_burst" "$SCRIPTS_DIR/attack-burst.js"
}

run_B_combo() { run_combo "B"; }
run_D_combo() { run_combo "D"; }

cmd_run() {
  case "${1:-}" in
    A) run_A ;;
    B) run_B_combo ;;
    C) run_C ;;
    D) run_D_combo ;;
    all)
      # Warning: concurrent overall load; ensure safe thresholds/limits!
      run_D_combo
      run_C
      ;;
    *) usage; exit 1 ;;
  esac
}

cmd_tail() {
  local name="${1:-}"
  if [[ "$name" == "all" ]]; then
    tail -n +0 -F "$LOG_DIR"/*.log
  else
    local files=( "$LOG_DIR/${name}"*.log )
    if ls "${files[@]}" >/dev/null 2>&1; then
      tail -n +0 -F "${files[@]}"
    else
      echo "No logs match ${name}" >&2; exit 2
    fi
  fi
}

cmd_stop() {
  local name="${1:-}"
  stop_file() {
    local f="$1"
    if [[ -f "$f" ]]; then
      local pid; pid="$(cat "$f" || true)"
      if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
        echo "[*] Killing PID $pid from $(basename "$f")"
        kill "$pid" || true
      fi
      rm -f "$f"
    fi
  }
  if [[ "$name" == "all" ]]; then
    for f in "$LOG_DIR"/*.pid; do stop_file "$f"; done
  else
    for f in "$LOG_DIR/${name}"*.pid "$LOG_DIR/${name}.pid"; do
      [[ -e "$f" ]] && stop_file "$f"
    done
  fi
}

main() {
  local sub="${1:-}"; shift || true
  case "$sub" in
    run)  cmd_run "${1:-}";;
    tail) cmd_tail "${1:-}";;
    stop) cmd_stop "${1:-}";;
    *) usage; exit 1;;
  esac
}
main "$@"
