#!/usr/bin/env bash
set -euo pipefail

# Параметри
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
ORCH="$ROOT_DIR/load/k6/orchestrate_k6_runs_v3.sh"
LOG_DIR="$ROOT_DIR/k6_runs_logs"
mkdir -p "$LOG_DIR"

# ===== Часові утиліти =====
iso_utc_now() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
iso_local_now() { date +"%Y-%m-%dT%H:%M:%S%z"; }

# EPOCH(sec) -> ISO-8601 UTC (GNU/Linux і macOS)
iso_utc_from_epoch() {
  local epoch="$1"
  if date -u -d "@$epoch" +"%Y-%m-%dT%H:%M:%SZ" >/dev/null 2>&1; then
    date -u -d "@$epoch" +"%Y-%m-%dT%H:%M:%SZ"
  else
    date -u -r "$epoch" +"%Y-%m-%dT%H:%M:%SZ"
  fi
}

# Мінімальні/дефолтні тривалості (хвилини), якщо не виставлені ззовні
num() { awk "BEGIN{printf \"%d\", $1}"; } # ціле
geti() {
  local name="$1" def="$2"
  local val="${!name:-}"
  [[ -n "$val" ]] && echo "$val" || echo "$def"
}

# A (benign-breeze)
WARM_MINS=$(geti WARM_MINS 5)
BREEZE_MINS=$(geti BREEZE_MINS 30)
COOL_MINS=$(geti COOL_MINS 5)
TOT_A_MIN=$(( WARM_MINS + BREEZE_MINS + COOL_MINS ))

# B/D (attack-burst + benign фоном)
ATT_START_OFFSET_MINS=$(geti ATT_START_OFFSET_MINS 15)
ATT_RAMP_MINS=$(geti ATT_RAMP_MINS 2)
ATT_PEAK_MINS=$(geti ATT_PEAK_MINS 8)
ATT_COOL_MINS=$(geti ATT_COOL_MINS 2)
TOT_ATTACK_MIN=$(( ATT_RAMP_MINS + ATT_PEAK_MINS + ATT_COOL_MINS ))
# Загальна тривалість combo = max(TOT_A_MIN, ATT_START_OFFSET_MINS + TOT_ATTACK_MIN)
combo_total_minutes() {
  local max1="$TOT_A_MIN"
  local max2=$(( ATT_START_OFFSET_MINS + TOT_ATTACK_MIN ))
  (( max1 > max2 )) && echo "$max1" || echo "$max2"
}
TOT_B_MIN=$(combo_total_minutes)
TOT_D_MIN="$TOT_B_MIN"

# C (benign-surge)
SURGE_RAMP_MINS=$(geti SURGE_RAMP_MINS 3)
SURGE_PEAK_MINS=$(geti SURGE_PEAK_MINS 30)
SURGE_COOL_MINS=$(geti SURGE_COOL_MINS 7)
TOT_C_MIN=$(( SURGE_RAMP_MINS + SURGE_PEAK_MINS + SURGE_COOL_MINS ))

banner() { printf "\n========== %s ==========\n" "$*"; }

# Чекати завершення всіх PID, що збігаються з шаблоном
wait_for_label() {
  local pattern="$1"
  shopt -s nullglob
  local pidfiles=( "$LOG_DIR"/${pattern}*.pid )
  shopt -u nullglob

  if (( ${#pidfiles[@]} == 0 )); then
    echo "[WARN] Немає PID-файлів за шаблоном: ${pattern}*.pid"
    return 0
  fi

  for pf in "${pidfiles[@]}"; do
    local pid; pid="$(cat "$pf" 2>/dev/null || true)"
    if [[ -z "${pid:-}" ]]; then
      echo "[WARN] Порожній PID у ${pf}, видаляю"; rm -f "$pf"; continue
    fi
    echo "[*] Очікую завершення PID ${pid} (${pf}) ..."
    while kill -0 "$pid" 2>/dev/null; do sleep 5; done
    rm -f "$pf" || true
  done
}

timeline_file="$LOG_DIR/timeline_$(date -u +%Y%m%dT%H%M%SZ).txt"
print_and_log() { echo -e "$*" | tee -a "$timeline_file" >/dev/null; }

# Попередній розрахунок плану (UTC + локальний)
plan_timeline() {
  local start_epoch end_epoch atks

  start_epoch="$(date -u +%s)"
  print_and_log "=== TIMELINE ==="
  print_and_log "Now (UTC):   $(iso_utc_now)"
  print_and_log "Now (Local): $(iso_local_now)"
  print_and_log ""

  # 1) A
  local A_start="$start_epoch"
  local A_end=$(( A_start + TOT_A_MIN*60 ))
  print_and_log "1) A benign-breeze:"
  print_and_log "   start_utc:  $(iso_utc_from_epoch "$A_start")"
  print_and_log "   end_utc:    $(iso_utc_from_epoch "$A_end")"
  print_and_log "   duration:   ${TOT_A_MIN}m"
  print_and_log ""

  # 2) B combo
  local B_start="$A_end"
  local B_attack_start=$(( B_start + ATT_START_OFFSET_MINS*60 ))
  local B_attack_end=$(( B_attack_start + TOT_ATTACK_MIN*60 ))
  local B_benign_end=$(( B_start + TOT_A_MIN*60 ))
  local B_end="$B_benign_end"
  if (( B_attack_end > B_benign_end )); then B_end="$B_attack_end"; fi
  print_and_log "2) B combo (benign + attack-burst):"
  print_and_log "   start_utc:        $(iso_utc_from_epoch "$B_start")"
  print_and_log "   benign_end_utc:   $(iso_utc_from_epoch "$B_benign_end")"
  print_and_log "   attack_start_utc: $(iso_utc_from_epoch "$B_attack_start")  (+${ATT_START_OFFSET_MINS}m)"
  print_and_log "   attack_end_utc:   $(iso_utc_from_epoch "$B_attack_end")    (attack duration ${TOT_ATTACK_MIN}m)"
  print_and_log "   end_utc:          $(iso_utc_from_epoch "$B_end")"
  print_and_log ""

  # 3) C
  local C_start="$B_end"
  local C_end=$(( C_start + TOT_C_MIN*60 ))
  print_and_log "3) C benign-surge:"
  print_and_log "   start_utc:  $(iso_utc_from_epoch "$C_start")"
  print_and_log "   end_utc:    $(iso_utc_from_epoch "$C_end")"
  print_and_log "   duration:   ${TOT_C_MIN}m"
  print_and_log ""

  # 4) D combo
  local D_start="$C_end"
  local D_attack_start=$(( D_start + ATT_START_OFFSET_MINS*60 ))
  local D_attack_end=$(( D_attack_start + TOT_ATTACK_MIN*60 ))
  local D_benign_end=$(( D_start + TOT_A_MIN*60 ))
  local D_end="$D_benign_end"
  if (( D_attack_end > D_benign_end )); then D_end="$D_attack_end"; fi
  print_and_log "4) D combo (benign + attack-burst):"
  print_and_log "   start_utc:        $(iso_utc_from_epoch "$D_start")"
  print_and_log "   benign_end_utc:   $(iso_utc_from_epoch "$D_benign_end")"
  print_and_log "   attack_start_utc: $(iso_utc_from_epoch "$D_attack_start")  (+${ATT_START_OFFSET_MINS}m)"
  print_and_log "   attack_end_utc:   $(iso_utc_from_epoch "$D_attack_end")    (attack duration ${TOT_ATTACK_MIN}m)"
  print_and_log "   end_utc:          $(iso_utc_from_epoch "$D_end")"
  print_and_log ""

  print_and_log "TOTAL planned duration: $(( TOT_A_MIN + TOT_B_MIN + TOT_C_MIN + TOT_D_MIN )) minutes"
  print_and_log "Timeline file: $timeline_file"
}

run_and_wait() {
  local pretty="$1" orch_arg="$2" wait_pattern="$3"
  banner "Старт ${pretty} @ UTC $(iso_utc_now)"
  "$ORCH" run "$orch_arg"
  banner "Очікування завершення ${pretty} @ UTC $(iso_utc_now)"
  wait_for_label "$wait_pattern"
  banner "Завершено ${pretty} @ UTC $(iso_utc_now)"
}

main() {
  plan_timeline

  run_and_wait "1) A (benign-breeze)" "A" "A_benign_breeze"
  run_and_wait "2) B (combo: A+attack-burst)" "B" "B_"
  run_and_wait "3) C (benign-surge)" "C" "C_benign_surge"
  run_and_wait "4) D (combo: A+attack-burst)" "D" "D_"

  banner "ПОСЛІДОВНІСТЬ 1→2→3→4 ЗАВЕРШЕНА @ UTC $(iso_utc_now)"
  echo "Див. план: $timeline_file"
}

main "$@"
