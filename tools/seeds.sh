#!/bin/zsh
# Launch N seeds of one train.py config SERIALLY (one at a time: 8 GB machine), with
# per-dataset + per-series dumps for tools/compare.py.
#
#   tools/seeds.sh NAME N [VAR=value ...]
#   tools/seeds.sh base 3 TSAR_MINUTES=4
#   tools/seeds.sh latent 3 TSAR_MINUTES=4 TSAR_LATENT=0.3
#   SEEDS="3 4" tools/seeds.sh base 0 TSAR_MINUTES=4     # explicit seed list (N ignored)
#
# Writes runs/NAME_s<seed>.log and runs/dumps/NAME_s<seed>/ for each seed, then prints the
# multi-seed summary. Existing logs are skipped (re-run with FORCE=1), so an interrupted
# sweep can be resumed by re-issuing the same command.
# Never sets TSAR_RUN: a checkpoint name would make later seeds resume the first one.
# Never sets TSAR_SPLIT: confirmatory test/holdout runs are done by hand, once (docs/eval_rigor.md).
set -u
cd "${0:A:h}/.."
[[ $# -ge 2 ]] || { sed -n '2,15p' "$0"; exit 2; }
NAME=$1; N=$2; shift 2
for kv in "$@"; do
  [[ $kv == TSAR_RUN=* || $kv == TSAR_SPLIT=* || $kv == TSAR_SEED=* || $kv == TSAR_DUMP=* ]] && {
    echo "seeds.sh: do not pass ${kv%%=*}; it is managed per seed" >&2; exit 2; }
done
SEEDS=${SEEDS:-$(seq 0 $((N - 1)))}
mkdir -p runs/dumps
for s in ${=SEEDS}; do
  log=runs/${NAME}_s${s}.log
  if [[ -e $log && -z ${FORCE:-} ]] && grep -q '^val_wql_rel' $log; then
    echo "=== $NAME seed $s: exists, skipping ($log)"; continue
  fi
  echo "=== $NAME seed $s  $*  ($(date +%H:%M:%S))"
  env "$@" TSAR_SEED=$s TSAR_DUMP=runs/dumps/${NAME}_s${s} uv run tools/run.py > $log 2>&1
  grep -E '^val_wql_rel|^zs4' $log || { echo "seed $s FAILED, tail of $log:"; tail -5 $log; }
done
echo "=== summary"
uv run tools/compare.py seeds "runs/dumps/${NAME}_s*"
