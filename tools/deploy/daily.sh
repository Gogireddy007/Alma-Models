#!/usr/bin/env bash
# Daily inference-only deployment + independent audit of the frozen Alma models (the "constantly checking" loop).
#   tools/deploy/daily.sh            # -> runs/deploy_daily/<UTC date>/, LEDGER.md, EXANTE_LEDGER.md, a commit
# Works from a cold container: it builds the venv and re-downloads (and sha256-checks) all pinned data if missing.
# The audit's exit code decides the commit message: a failed check is committed as FAILED, never hidden.
set -uo pipefail
cd "$(dirname "$0")/../.."
TODAY=$(date -u +%F)
export ALMA_DEPLOY_OUT="runs/deploy_daily/$TODAY"
mkdir -p "$ALMA_DEPLOY_OUT"
LOG="$ALMA_DEPLOY_OUT/daily.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== Alma daily deployment $TODAY ($(date -u +%FT%TZ))"

# ---- environment (cold start only)
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv -q
  uv pip install --python .venv/bin/python -q --index-url https://download.pytorch.org/whl/cpu torch
  uv pip install --python .venv/bin/python -q numpy==2.5.4 pandas==3.1.0 pyarrow==26.0.0 scipy scikit-learn \
    statsmodels xlrd openpyxl lightgbm reportlab matplotlib pypdf
  uv pip install --python .venv/bin/python -q 'chronos-forecasting>=2.0' 'statsforecast>=2.0' 'tirex-ts>=1.4.2' \
    --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu
fi
PY=.venv/bin/python

# ---- data (pinned history verified, then new days)
$PY tools/deploy/refresh_data.py --today "$TODAY" || echo "!! some data sources failed (see above); continuing"

# ---- inference only
$PY tools/deploy/deploy_oil.py
$PY tools/deploy/deploy_power.py
$PY tools/deploy/deploy_gas.py
$PY tools/deploy/deploy_fin.py --variant corrected
$PY tools/deploy/exante_power.py || true          # skips itself if D-2 fundamentals are incomplete or too late

# ---- audit, reports, grades
$PY tools/deploy/audit_deploy.py; AUDIT=$?
$PY tools/deploy/make_report.py
$PY tools/deploy/score_exante.py
$PY tools/deploy/daily_ledger.py

# ---- publish
STATUS=$([ "$AUDIT" -eq 0 ] && echo "audit passed" || echo "AUDIT FAILED")
git add "$ALMA_DEPLOY_OUT" runs/deploy_daily/LEDGER.md runs/deploy_daily/LEDGER.json \
  runs/deploy_daily/EXANTE_LEDGER.md runs/deploy_daily/EXANTE_LEDGER.json runs/deploy_daily/manifests/"$TODAY"
git commit -q -m "Daily deployment $TODAY: $STATUS" -m "Inference only, frozen checkpoints; see runs/deploy_daily/LEDGER.md." \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || echo "nothing to commit"
echo "=== done: $STATUS"
exit "$AUDIT"
