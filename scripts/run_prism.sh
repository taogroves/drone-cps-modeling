#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-prism/models/city_drone_risk.prism}"
PROPS_PATH="${2:-prism/properties/risk_queries.props}"

echo "Running PRISM model check..."
echo "  Model:      ${MODEL_PATH}"
echo "  Properties: ${PROPS_PATH}"

prism "${MODEL_PATH}" "${PROPS_PATH}"
