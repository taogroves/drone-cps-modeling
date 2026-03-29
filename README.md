# Drone CPS Risk Modeling (PRISM)

This repository contains a PRISM project for analyzing risks to a medical delivery drone flying through an urban environment. 
Specifically, it focuses on having regions that are inaccessible (via jamming). 

## Project goals

- Model probability of successful delivery within a certain time, given a discrete risk distribution on a 2D route
- Compare route, threat, and mitigation scenarios.

## Model Details
- 2D grid (height abstracted)
- Each flight has a discrete "mission priority"
- Each tile has a discrete value for a set of risk-related qualities
    - Time to cross
    - Collateral damage risk (pedestrian density, important sites, etc.)
    - Crash risk (high wind, storms, jamming, etc.)
- Drone can only see qualities of adjacent tiles and must choose a route dynamically
- Model outputs the probability of successful delivery within given constraints

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
