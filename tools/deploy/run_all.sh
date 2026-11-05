#!/usr/bin/env bash
# Deploy every frozen Alma model on Oct/Nov 2026, audit it, and build the results tables. Stops on the first failure;
# the audit exits non-zero if any integrity check fails. See docs/deploy_2026_10_11.md for the data rebuild steps.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
$PY tools/deploy/deploy_oil.py
$PY tools/deploy/deploy_power.py
$PY tools/deploy/gas_daily_noweather.py       # only if the Open-Meteo weather files are unavailable
$PY tools/deploy/deploy_gas.py
# crypto: data/fin/*.parquet must hold the build being scored (pinned raw set, then the corrected set)
$PY tools/deploy/deploy_fin.py --variant "${FIN_VARIANT:-corrected}"
$PY tools/deploy/funding_effect.py || echo "funding_effect needs both fin variants"
$PY tools/deploy/audit_deploy.py
$PY tools/deploy/make_report.py
$PY tools/deploy/check_doc_numbers.py
$PY tools/deploy/make_pdf_report.py
