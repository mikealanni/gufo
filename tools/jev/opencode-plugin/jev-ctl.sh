#!/usr/bin/env bash
# Control script for the /jev opencode command. pause/resume act on the calling session only
# (OPENCODE_SESSION_ID); "pause all" / "resume all" act on every session.
set -u
DIR="${JEV_LOG_DIR:-$HOME/.gufo/jev-log}"
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
SID="${OPENCODE_SESSION_ID:-}"
SAFE="$(printf '%s' "$SID" | tr -c 'A-Za-z0-9._-' '_')"
ALL="$DIR/plugin-pause"
ONE="$DIR/plugin-pause.$SAFE"
ARG="${1:-}"; SCOPE="${2:-}"
[ "$ARG" = "pause" ] || [ "$ARG" = "resume" ] || [ "$ARG" = "status" ] || [ "$ARG" = "autoeffort" ] || [ "$ARG" = "feature" ] || [ "$ARG" = "goal" ] || { set -- $ARG; ARG="${1:-}"; SCOPE="${2:-}"; }
case "$ARG" in
  pause)
    mkdir -p "$DIR"
    if [ "$SCOPE" = "all" ]; then : > "$ALL"; echo "jev plugin PAUSED for ALL sessions."
    elif [ -n "$SID" ]; then : > "$ONE"; echo "jev plugin PAUSED for this session ($SID) only."
    else echo "no session id available here; use: /jev pause all"; fi ;;
  resume)
    if [ "$SCOPE" = "all" ]; then rm -f "$ALL" "$DIR"/plugin-pause.*; echo "jev plugin RESUMED for ALL sessions."
    elif [ -n "$SID" ]; then rm -f "$ONE"; echo "jev plugin RESUMED for this session ($SID)."
      [ -e "$ALL" ] && echo "note: a global pause is still active (/jev resume all)."
    else echo "no session id available here; use: /jev resume all"; fi ;;
  feature)
    case "$SCOPE" in command|effort|goal|finish|pre) ;; *) echo "usage: /jev feature command|effort|goal|finish|pre on|off"; exit 0 ;; esac
    F="$DIR/feature-off.$SCOPE"
    case "${3:-}" in
      off) mkdir -p "$DIR"; : > "$F"; echo "feature '$SCOPE' OFF." ;;
      on) rm -f "$F"; echo "feature '$SCOPE' ON." ;;
      *) [ -e "$F" ] && echo "feature '$SCOPE': off" || echo "feature '$SCOPE': on" ;;
    esac ;;
  autoeffort)
    EF="$DIR/autoeffort-off"
    case "$SCOPE" in
      off) LEVEL="${3:-low}"; mkdir -p "$DIR"; printf '%s' "$LEVEL" > "$EF"
        echo "autoeffort OFF: every gufo request without its own effort is pinned to '$LEVEL' (all sessions)." ;;
      on) rm -f "$EF"; echo "autoeffort ON: jev picks the reasoning effort per request." ;;
      *) [ -e "$EF" ] && echo "autoeffort: OFF, pinned to $(cat "$EF")" || echo "autoeffort: ON (default)"
         echo "usage: /jev autoeffort on | off [none|minimal|low|medium|high|xhigh|max]" ;;
    esac ;;
  status)
    [ -n "$SID" ] && { [ -e "$ONE" ] && echo "this session ($SID): PAUSED" || echo "this session ($SID): active"; }
    python3 "$HERE/status.py" 2>&1 | head -14 ;;
  goal)
    shift
    mkdir -p "$DIR"
    if [ "${1:-}" = "off" ]; then : > "$DIR/goal-off"; echo "goal cancelled for the next agent turn."
    else printf '%s' "$*" > "$DIR/goal-pending"; echo "goal armed${*:+: $*}"; echo "it takes effect when the agent next stops."; fi ;;
  ""|help)
    cat <<'HELP'
jev control (type jev <option>)
  status                      plugin state, pause, features, autoeffort, guard (model, memory)
  pause [all]                 pause jev for this session, or all sessions
  resume [all]                resume jev
  autoeffort on|off [level]   let jev pick reasoning effort, or pin it (none = no thinking|minimal|low|medium|high|xhigh|max)
  goal [text]                 keep the agent working until the task is done (no text = current task)
  goal off                    cancel the goal
  feature <name> on|off       switch one feature: command, effort, goal, finish, pre
  help                        this list
HELP
    ;;
  *)
    echo "unknown option: $ARG (run: jev help)" ;;
esac
