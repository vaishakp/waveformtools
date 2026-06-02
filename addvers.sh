#!/usr/bin/env bash
set -euo pipefail

python .generate_version.py

git add waveformtools/__init__.py
git add setup.py
git add public/version
git add docs/vers_badge.svg
