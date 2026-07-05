#!/usr/bin/env bash
# Regenerate the pre-rendered figures embedded in the docs.
#
# Each docs/**/*.py figure script prints its output (SVG plot, or an HTML
# table for tfidf-demo) to stdout; the result is committed next to the script
# and referenced from the markdown. Figures are NOT generated at deploy time —
# rerun this script after editing a figure script, then commit the outputs.
#
# Requires the scientific stack: pip install matplotlib numpy pandas scikit-learn
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-./env/bin/python}

for script in docs/2026.2/classes/*/*.py; do
    stem="${script%.py}"
    case "$script" in
        */tfidf-demo.py) out="${stem}.html" ;;
        *)               out="${stem}.svg" ;;
    esac
    echo "→ $script  >  $out"
    "$PY" "$script" > "$out"
done
echo "done."
