#!/usr/bin/env bash
# Jev-gate stage 0 measurement: matched wall time, tokens and retained answers
# for the Flash-Next reasoning pre-gate against its two baselines.
#
#   ./tests/functional/jev_gate_measure.sh                 # all three arms
#   JEV_ONLY=trivial ./tests/functional/jev_gate_measure.sh
#   JEV_ARMS="gate_on gate_off" ./tests/functional/jev_gate_measure.sh
#
# Results land in one directory: summary.tsv, summary.txt, per-arm server logs
# with the recorded gate decisions, and every retained response body.
set -uo pipefail
# monitor mode: every background job (the server) becomes its own process group,
# so $! is also the PGID that stop_server() can signal.
set -m

# The server is launched through the operator's own run-gufo.sh so every arm
# inherits the production settings (quant, DFlash2 draft, context, cache tier,
# NPU clock hold and its RAM guard). Only the reasoning controls change per arm.
RUNNER=${JEV_RUNNER:-/home/majed/run-gufo.sh}
GUFO=${JEV_GUFO:-/home/majed/gufo-jevgate/build/release/gufo}
PORT=${JEV_PORT:-8901}
MODEL_ID=${JEV_MODEL_ID:-Qwen3.8 Flash Next}
OUT=${JEV_OUT:-/home/majed/jev-gate-run}
MAX_TOKENS=${JEV_MAX_TOKENS:-16384}
TIMEOUT=${JEV_TIMEOUT:-900}
ONLY=${JEV_ONLY:-all}
ARMS=${JEV_ARMS:-"gate_on gate_off effort_medium"}
LOAD_TIMEOUT=${JEV_LOAD_TIMEOUT:-3600}
CACHE_BYTES=${JEV_CACHE_BYTES:-32212254720}
WAIT_FREE=${JEV_WAIT_FREE:-10800}
STATUS="$OUT/STATUS"

mkdir -p "$OUT"
echo "RUNNING started=$(date -Is)" >"$STATUS"

if [[ ! -x $GUFO ]]; then
  echo "== $GUFO missing; building it (this takes a while)"
  (cd "$(dirname "$(dirname "$(dirname "$GUFO")")")" &&
      cmake --preset release &&
      cmake --build --preset release --target gufo --parallel 4) \
    >"$OUT/build.log" 2>&1 || {
    echo "FAILED build (see $OUT/build.log)" >"$STATUS"
    exit 1
  }
fi
if [[ ! -f $RUNNER ]]; then
  echo "server runner not found: $RUNNER (set JEV_RUNNER)" >&2
  echo "FAILED runner-missing" >"$STATUS"
  exit 1
fi
command -v jq >/dev/null || {
  echo "jq is required" >&2
  echo "FAILED no-jq" >"$STATUS"
  exit 1
}

# Gufo's weights are pinned in RAM (GTT) and cannot swap: a second server
# started while one is resident OOM-kills the box mid-load. Wait for the
# production server to be stopped rather than refusing, so this can be left
# running unattended and started before the server is brought down.
waited=0
while pgrep -f "gufo serve llm" >/dev/null; do
  if [[ ${JEV_FORCE:-0} == 1 ]]; then
    break
  fi
  if ((waited == 0)); then
    echo "== a gufo serve is running; waiting up to ${WAIT_FREE}s for it to stop"
  fi
  if ((waited >= WAIT_FREE)); then
    echo "FAILED gufo-never-freed (stop the production server first)" >"$STATUS"
    exit 1
  fi
  sleep 10
  waited=$((waited + 10))
done

mkdir -p "$OUT/prompts"
echo "== runner: $RUNNER with GUFO_BIN=$GUFO port=$PORT"
echo "== out:    $OUT"

PROMPT_IDS=()
PROMPT_CLASSES=()
PROMPT_BODIES=()

read_prompts() {
  local line id="" body=""
  while IFS= read -r line; do
    if [[ $line == '#@ '* ]]; then
      if [[ -n $id ]]; then
        PROMPT_BODIES+=("$body")
      fi
      read -r _ id class <<<"$line"
      body=""
      PROMPT_IDS+=("$id")
      PROMPT_CLASSES+=("$class")
    else
      body+="$line"$'\n'
    fi
  done
  if [[ -n $id ]]; then
    PROMPT_BODIES+=("$body")
  fi
}

read_prompts <<'PROMPTS'
#@ t01 trivial
What does the -j flag do in make -j8?
#@ t02 trivial
What does HTTP status 425 mean?
#@ t03 trivial
What is git stash pop used for?
#@ t04 trivial
Say in one sentence what this line reports: [INFO] cache restore 128 tokens in 41 ms
#@ t05 trivial
Translate good morning into Turkish.
#@ t06 trivial
What is 19 times 7?
#@ t07 trivial
Which cargo command builds an optimized Rust binary?
#@ t08 trivial
What does TS stand for in TypeScript error codes?
#@ t09 trivial
Which command shows the current git branch?
#@ t10 trivial
How many kibibytes are in one mebibyte?
#@ t11 trivial
In one line: difference between rm and unlink?
#@ t12 trivial
Thanks! Quick one: what does nproc print?
#@ h01 hard
This python script raised and I need it working:

Traceback (most recent call last):
  File "/srv/ingest/load.py", line 84, in flush
    batch = [normalize(row) for row in rows if row["ts"] > cutoff]
  File "/srv/ingest/load.py", line 61, in normalize
    return {"ts": float(row["ts"]), "v": row["v"] * scale}
TypeError: unsupported operand type(s) for *: 'NoneType' and 'float'

Rows come from a JSON API that sometimes sends "v": null. Fix normalize so
nulls are dropped rather than crashing, without scanning twice.
#@ h02 hard
Fix this bug. The function returns the wrong result for the third case:

int CeilingDiv(int a, int b) {
  return (a + b - 1) / b;
}

CHECK(CeilingDiv(7, 2) == 4);
CHECK(CeilingDiv(-7, 2) == -3);
CHECK(CeilingDiv(9, 3) == 3);
CHECK(CeilingDiv(-9, 3) == -3);

Why is the negative case wrong and what is the correct implementation for both
signs in C++17 where / truncates toward zero?
#@ h03 hard
Two files changed and the suite broke. Which change is responsible and why?

diff --git a/src/cache.cpp b/src/cache.cpp
@@ -120,7 +120,7 @@ void Cache::Evict() {
-  while (bytes_ > limit_) {
+  while (bytes_ >= limit_) {
     DropOldest();
   }
diff --git a/src/session.cpp b/src/session.cpp
@@ -44,6 +44,7 @@ Session::Session(std::size_t slot) : slot_(slot) {
+  bytes_held_ = 0;
   Attach(slot_);
 }

The failing test asserts that an empty cache reports bytes_ == limit_ after a
fill/drain cycle, and it does not. My guess was session.cpp but reverting it
does not help. Reason it through.
#@ h04 hard
My worker pool deadlocks only under load, never in a unit test:

  std::mutex m;
  std::condition_variable cv;
  bool stop = false;
  void Worker() {
    while (true) {
      std::unique_lock<std::mutex> lk(m);
      cv.wait(lk, [] { return stop || !queue.empty(); });
      if (stop) return;
      auto job = queue.front(); queue.pop();
      lk.unlock();
      job();
    }
  }
  void Shutdown() { stop = true; /* no notify under lock */ }

Explain the exact interleaving that loses the wakeup and give the fix.
#@ h05 hard
Process died in production, here is the dmesg tail:

kernel: python3[2211]: segfault at 0x7f3a1b2c0000 ip 0x7f3a20113aa0 error 4
kernel: python3[2211]: memory: cgroup out of memory: Killed process 2211
systemd-coredump: Process 2211 (python3) of user 1000 dumped core.

It only dies after ~40 minutes of steady traffic and RSS looks flat before it.
What are the three most likely causes for a flat-RSS OOM kill and how do I
confirm each one?
#@ h06 hard
Off-by-one somewhere; it fails on the second element of a 2-element array:

int BinarySearch(const int* a, int n, int key) {
  int lo = 0, hi = n;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (a[mid] < key) lo = mid; else hi = mid;
  }
  return lo;
}

Find the bug, state the inputs that expose it, and give the corrected loop.
#@ h07 hard
AssertionError: expected 200 got 500 on POST /ingest

The handler validates with this schema and the payload is
{"events":[{"ts":"2026-11-01T10:00:00Z","v":1}]}.

  events = fields.List(fields.Str(), required=True)

I cannot find where the 500 comes from - there is no traceback in the log even
though debug logging is on. Where do I look and what is the likely cause?
#@ h08 hard
This HIP kernel gives different results per launch on gfx1151 but is stable on
my desktop GPU:

__global__ void Reduce(const float* in, float* out, int n) {
  __shared__ float s[256];
  s[threadIdx.x] = in[blockIdx.x * blockDim.x + threadIdx.x];
  for (int off = 128; off > 0; off >>= 1) {
    if (threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x + off];
  }
  if (threadIdx.x == 0) atomicAdd(out, s[0]);
}

What is missing, and why can the missing piece be invisible on one GPU?
#@ h09 hard
Query got 40x slower after adding the index. Plan:

  -> Index Scan using idx_events_ts (cost=0.42..9120.19 rows=988712 width=41)
     Index Cond: (ts >= '2026-11-01')
     Rows Removed by Filter: 4213008

The index is btree on (ts). Why is so much data filtered after an index scan on
a range condition, and which index definition would fix it for this query?
#@ h10 hard
During an interactive rebase I resolved a conflict, continued, and now the
commit is empty but the branch is one commit behind develop. I want the branch
to contain develop's work plus my two commits without duplicating changes.
Show the exact commands and explain what happens to the resolved commit.
#@ h11 hard
Implement a rate limiter that must be correct under concurrency and then fix the
flaw in this sketch:

  tokens = capacity
  loop { request -> if tokens > 0 { tokens -= 1; allow } else { deny } }

State the invariants, the failure mode under two threads, and the corrected
algorithm with the data structures you would actually deploy.
#@ h12 hard
Memory grows without bound in this service; gc shows:

  12,340,112 dict objects
   9,887,431 tuple objects
     120,331 _cython_function_or_method objects

Only the request handler changed:

  CACHE = {}
  def handle(req):
      key = (req.tenant, req.path, req.ts)
      if key not in CACHE:
          CACHE[key] = expensive(req)
      return CACHE[key]

Why is gc counting dicts, why does the tuple key never get freed, and what is
the minimal fix that keeps cache hits?
PROMPTS

TOTAL=${#PROMPT_IDS[@]}
if [[ $TOTAL -eq 0 ]]; then
  echo "prompt table is empty" >&2
  exit 1
fi

SELECTED=()
for i in "${!PROMPT_IDS[@]}"; do
  if [[ $ONLY == all || $ONLY == "${PROMPT_CLASSES[$i]}" ]]; then
    SELECTED+=("$i")
  fi
done

printf '%-14s %-6s %-5s %9s %8s %10s %9s %8s %s\n' \
  arm id class wall_s prompt_tok compl_tok think_ch http gate >"$OUT/summary.tsv"

SERVER_PID=""

stop_server() {
  if [[ -n $SERVER_PID ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    # setsid makes the runner a process group leader: the wrapper, the gufo
    # it spawns and the tee all die together.
    kill -TERM -- -"$SERVER_PID" 2>/dev/null
    for _ in $(seq 120); do
      kill -0 -- -"$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- -"$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
  fi
  SERVER_PID=""
}
completed=0
trap 'rc=$?; if [[ $completed != 1 ]]; then
        echo "INTERRUPTED rc=$rc at=$(date -Is)" >"$STATUS"
      fi
      stop_server' EXIT

# effort xhigh is the production default: the pre-gate replaces it per request,
# the baselines pin it at startup instead.
arm_settings() {
  case $1 in
    gate_on) echo "xhigh on" ;;
    gate_off) echo "xhigh off" ;;
    effort_medium) echo "medium off" ;;
    *) echo "unknown arm: $1" >&2; return 1 ;;
  esac
}

failed_arms=""
for arm in $ARMS; do
  read -r effort gate_flag <<<"$(arm_settings "$arm")" || exit 1
  log="$OUT/$arm.server.log"
  echo "== $arm: GUFO_EFFORT=$effort --jev-gate=$gate_flag (log $log)"
  echo "RUNNING arm=$arm started=$(date -Is)" >"$STATUS"
  env GUFO_KIND=flash-next GUFO_BIN="$GUFO" PORT="$PORT" \
    JOBS="${JEV_JOBS:-1}" GUFO_EFFORT="$effort" GUFO_LOG="$log" \
    GUFO_CACHE_DIR="$OUT/cache-$arm" GUFO_CACHE_BYTES="$CACHE_BYTES" \
    bash "$RUNNER" --jev-gate "$gate_flag" --log-level debug \
      >/dev/null 2>&1 &
  SERVER_PID=$!

  ready=0
  for _ in $(seq "$LOAD_TIMEOUT"); do
    if curl -sf -o /dev/null --max-time 5 "http://127.0.0.1:$PORT/health"; then
      ready=1
      break
    fi
    kill -0 "$SERVER_PID" 2>/dev/null || {
      echo "server died during load; see $log" >&2
      break
    }
    sleep 1
  done
  if [[ $ready -ne 1 ]]; then
    failed_arms="$failed_arms $arm"
    echo "arm $arm produced no healthy server; skipping" >&2
    stop_server
    continue
  fi

  mkdir -p "$OUT/responses/$arm"
  mark=$(wc -l <"$log")
  for i in "${SELECTED[@]}"; do
    id=${PROMPT_IDS[$i]}
    class=${PROMPT_CLASSES[$i]}
    body=${PROMPT_BODIES[$i]}
    printf '%s' "$body" >"$OUT/prompts/$id.txt"
    payload=$(printf '%s' "$body" | jq -Rs \
      --arg model "$MODEL_ID" --argjson mt "$MAX_TOKENS" \
      '{model: $model, max_tokens: $mt, messages: [{role: "user", content: .}]}')

    raw=$(curl -sS --max-time "$TIMEOUT" -X POST \
      -H 'Content-Type: application/json' \
      -w $'\n%{http_code} %{time_total}' \
      "http://127.0.0.1:$PORT/v1/chat/completions" --data-binary @- <<<"$payload")
    status_line=${raw##*$'\n'}
    resp=${raw%$'\n'*}
    http=${status_line%% *}
    wall=${status_line##* }

    echo "$resp" >"$OUT/responses/$arm/$id.json"
    compl=$(jq -r '.usage.completion_tokens // 0' <<<"$resp" 2>/dev/null)
    ptok=$(jq -r '.usage.prompt_tokens // 0' <<<"$resp" 2>/dev/null)
    think=$(jq -r '(.choices[0].message.reasoning_content // "") | length' \
      <<<"$resp" 2>/dev/null)
    gate=$(tail -n "+$((mark + 1))" "$log" |
      grep -o 'jev-gate.*' | tail -1 | sed 's/^jev-gate\] *//')
    mark=$(wc -l <"$log")
    printf '%-14s %-6s %-5s %9s %8s %10s %9s %8s %s\n' \
      "$arm" "$id" "$class" "${wall:-0}" "${ptok:-0}" "${compl:-0}" \
      "${think:-0}" "${http:-0}" "${gate:-none}" | tee -a "$OUT/summary.tsv"
    # The labeled set: it is what "speed delta and correctness delta on the
    # same labeled set" is measured on, what decides Part A step 5
    # (heuristics vs jev zero-shot), and what the finish detector is tuned
    # against. `correct`/`note` are filled in by hand after grading.
    jq -cn --arg arm "$arm" --arg id "$id" --arg class "$class" \
      --arg gate "${gate:-none}" --rawfile text "$OUT/prompts/$id.txt" \
      --argjson wall "${wall:-0}" --argjson ptok "${ptok:-0}" \
      --argjson compl "${compl:-0}" --argjson think "${think:-0}" \
      --argjson http "${http:-0}" \
      '{arm: $arm, id: $id, class: $class, gate: $gate, wall_s: $wall,
        prompt_tokens: $ptok, completion_tokens: $compl,
        reasoning_chars: $think, http: $http, correct: null, note: "",
        prompt: $text}' >>"$OUT/labels.jsonl"
    if kill -0 "$SERVER_PID" 2>/dev/null; then :; else
      failed_arms="$failed_arms $arm(mid)"
      echo "server died mid-arm; see $log" >&2
      break
    fi
  done
  stop_server
done

{
  echo "== summary by arm and class (mean wall, mean completion tokens,"
  echo "   thinking count over ${#SELECTED[@]} prompts)"
  awk -F'[ \t]+' 'NR>1 {
    key=$1" "$3; n[key]++; w[key]+=$4; c[key]+=$6; t[key]+=($7>0?1:0)
  } END {
    for (k in n)
      printf "%-22s n=%-3d wall=%8.1fs  compl=%8.0f  thinking=%d\n",
        k, n[k], w[k]/n[k], c[k]/n[k], t[k]
  }' "$OUT/summary.tsv" | sort
  echo
  echo "Grade by hand: diff $OUT/responses/gate_on against gate_off"
  echo "(correctness) and against effort_medium (the cheaper baseline)."
  echo "Archive with: tar -C $(dirname "$OUT") -czf ${OUT}.tar.gz $(basename "$OUT")"
} | tee "$OUT/summary.txt"

completed=1
if [[ -n ${failed_arms// /} ]]; then
  echo "DONE-PARTIAL failed_arms=$failed_arms finished=$(date -Is)" >"$STATUS"
else
  echo "DONE finished=$(date -Is)" >"$STATUS"
fi
