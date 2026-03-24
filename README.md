# Drone CPS Risk Modeling (PRISM)

This repository contains a PRISM project for analyzing risks to a medical delivery drone flying through an urban environment.

## Project goals

- Model mission success and failure behaviors in a city route.
- Quantify safety and security risks with probabilistic model checking.
- Compare route, threat, and mitigation scenarios.

## Directory layout

- `prism/models/`: PRISM model files (`.prism`)
- `prism/properties/`: Property queries (`.props`)
- `scenarios/`: Scenario assumptions and parameter sets
- `data/`: Input datasets and city/threat references
- `results/`: Generated analysis outputs (ignored except `.gitkeep`)
- `scripts/`: Helper scripts for running and post-processing analyses
- `docs/`: Notes, design decisions, and experiment logs

## Quick start

1. Install [PRISM](https://www.prismmodelchecker.org/).
2. Update `prism/models/city_drone_risk.prism` with your mission assumptions.
3. Add or refine properties in `prism/properties/risk_queries.props`.
4. Run:

```bash
prism prism/models/city_drone_risk.prism prism/properties/risk_queries.props
```

## Next setup steps

- Define baseline urban mission parameters in `scenarios/baseline.yaml`.
- Add threat and mitigation variants in `scenarios/`.
- Track experiment outcomes in `docs/experiment-log.md`.
