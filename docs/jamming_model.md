# Jamming Threat Model Documentation

This document explains the adversarial jamming model for the medical delivery drone project. The model is written in the PRISM model-checking language and represents a drone navigating a 2D city grid while an adversary jams its sensors/communications in specific zones.

## File Overview

| File | Purpose |
|------|---------|
| `prism/models/jamming.prism` | The PRISM MDP model — all state, transitions, and probabilities |
| `prism/properties/jamming_queries.props` | PCTL queries to verify against the model |
| `scenarios/jamming_baseline.yaml` | Human-readable documentation of the baseline scenario parameters |

---

## The Scenario

A drone must deliver a time-sensitive organ from cell `(0,0)` to cell `(9,9)` on a 10x10 grid. An adversary has set up jamming equipment in specific zones of the city. The drone **cannot detect jamming before entering a cell** — it only discovers it's being jammed once it arrives. When jammed, the drone must decide how to respond.

The organ has a viability deadline of 40 time steps. If the drone doesn't deliver by then, the mission fails.

---

## The Grid and Jamming Zones

```
Y
9 . . . . . . . . . D 
8 . . . . C C C C . .       C = Comms jam zone
7 . . . . C C C C . .       G = GPS jam zone
6 . . . . C C C C . .       A = Camera jam zone
5 . . . A A A . . . .       
4 . . . A A A G G G .       Zones can overlap in
3 . . . A A A G G G .       custom scenarios but
2 . . . . . . G G G .       don't in this baseline.
1 . . . . . . . . . .
0 S . . . . . . . . .       S = Start, D = Destination
  0 1 2 3 4 5 6 7 8 9  X
```

*(Destination D is at (9,9), top-right corner.)*

Each zone has an **activation probability** — the chance that jamming is actually active when the drone enters a cell inside that zone. This models the fact that the adversary's equipment isn't perfectly reliable, and it means the drone truly can't plan around jamming even if it "knows" the zone boundaries (the MDP strategy can't exploit deterministic zone info because activation is probabilistic).

---

## Three Jamming Types

Each type is an independent, toggleable module. They differ in **what they break** and **what it costs the drone**:

### Camera Jamming
- **What happens:** The drone loses its obstacle-avoidance cameras. It can't see buildings, wires, birds, etc.
- **Effect:** Increased crash probability. No extra time cost (the drone doesn't know what it can't see).
- **Zone:** `(3,3)` to `(5,5)` — 3x3 block in the center of the grid
- **Activation probability:** 80%

### GPS Jamming
- **What happens:** The drone's position fix degrades. It knows roughly where it is but may miscalculate.
- **Effect:** Increased crash probability **plus** extra time for recalibration (the drone has to slow down to cross-reference other sensors).
- **Zone:** `(6,2)` to `(8,4)` — 3x3 block on the east side
- **Activation probability:** 70%

### Comms Jamming
- **What happens:** The drone loses its link to the base station. It can't receive updated routing, weather, or emergency abort commands.
- **Effect:** Small crash risk increase (no emergency support). When slowing down, extra time penalty because the drone falls back to its **preloaded route** — a conservative, pre-planned path it carries onboard for exactly this situation.
- **Zone:** `(4,6)` to `(7,8)` — 4x3 block in the upper-middle area
- **Activation probability:** 75%

---

## Drone Actions

The drone makes two types of decisions (these are the MDP nondeterministic choices):

### 1. Movement Direction
At each cell, the drone chooses North, South, East, or West. It can't stay in place — it must keep moving toward the destination.

### 2. Jamming Response
When the drone enters a jammed cell, it chooses one of two responses:

| | **Push Through** | **Slow Down** |
|---|---|---|
| **Philosophy** | "The organ is critical — I can handle this" | "Better safe than sorry — reduce risk" |
| **Time cost** | Normal (1 step) + GPS recal if GPS jammed | Slow (2 steps) + GPS recal + comms conservative routing |
| **Crash risk** | Higher | Lower |
| **Comms behavior** | Continue optimizing route without base input | Fall back to preloaded safe route |

The crash probabilities stack additively when multiple jam types are active:

| Jam Type | Push Through | Slow Down |
|----------|:-----------:|:---------:|
| Baseline (no jam) | 0.1% | 0.1% |
| + Camera | +15% | +5% |
| + GPS | +10% | +3% |
| + Comms | +6% | +2% |
| **All three active** | **31.1%** | **10.1%** |

Time costs also depend on which jams are active:

| Response | Base | + GPS | + Comms | + Both |
|----------|:----:|:-----:|:-------:|:------:|
| Push through | 1 | 2 | 1 | 2 |
| Slow down | 2 | 3 | 3 | 4 |

---

## How the Model Works (Phase System)

Each step of the drone's journey follows a three-phase cycle:

```
Phase 0: CHOOSE MOVE          Phase 1: JAM REVEAL           Phase 2: RESPOND
(drone decides direction)     (environment resolves jam)     (drone decides response)
                                                             
  move_n / move_s /    ──►    [check_jam] syncs all    ──►    !jammed → [proceed]
  move_e / move_w             three jam modules.              jammed  → [push_through]
                              Each independently                     OR [slow_down]
                              flips its coin.                        
                                                              at_dest → [arrive] ✓
                                                              
     ◄──────────────────────── (phase resets to 0) ─────────────────┘
```

**Phase 0** — The drone picks a direction. This is an MDP nondeterministic choice (PRISM will find the optimal strategy). If time has expired, the only available action is `[timeout]`.

**Phase 1** — The three jam modules synchronize on `[check_jam]`. Each module independently and probabilistically sets its variable (`cam`, `gps`, or `comms`) to 0 or 1 based on whether the drone is in that module's zone and the activation probability. PRISM automatically computes the joint distribution (product of independent coin flips).

**Phase 2** — The drone observes the jam state and decides:
- If **at the destination**: deliver (mission success).
- If **not jammed**: proceed normally (small base crash risk).
- If **jammed**: choose `push_through` or `slow_down` (MDP nondeterministic choice). The outcome is then resolved probabilistically (crash or survive).

---

## Module Architecture

The model uses PRISM's module system for clean separation:

```
┌──────────────────────────────┐
│         drone module         │  Owns: x, y, t, status, phase
│  (movement, response, time)  │  Reads: cam, gps, comms
└──────────────┬───────────────┘
               │ synchronizes on [check_jam]
       ┌───────┼───────┐
       ▼       ▼       ▼
┌──────────┐ ┌──────┐ ┌───────┐
│camera_jam│ │gps_  │ │comms_ │  Each owns its own
│          │ │jam   │ │jam    │  variable (cam/gps/comms)
└──────────┘ └──────┘ └───────┘
```

Each jam module owns its own state variable. The drone module reads them cross-module (PRISM allows this). This means you can toggle each jam type independently without touching the drone logic.

---

## How to Toggle Jamming Types

Each jam type has an `ENABLE` flag at the top of the model. To disable one, set it to `0`:

```
const int ENABLE_CAM_JAM = 1;    // 1 = on, 0 = off
const int ENABLE_GPS_JAM = 1;
const int ENABLE_COMMS_JAM = 1;
```

When disabled, the zone formula always evaluates to false, so the module never activates the jam. The variable still exists (always 0) and all formulas remain valid. No other changes needed.

You can also adjust the **activation probability** to fine-tune:
- `p_cam_active = 1.0` — jamming is guaranteed inside the zone (deterministic)
- `p_cam_active = 0.5` — 50/50 chance per cell entry
- `p_cam_active = 0.0` — functionally the same as disabling

---

## Properties File (`jamming_queries.props`)

These are the PCTL (Probabilistic Computation Tree Logic) queries you run against the model. They ask questions like:

### Probability Queries
| Query | What it asks |
|-------|-------------|
| `Pmax=? [ F "delivered" ]` | What's the **best** delivery probability the drone can achieve? |
| `Pmin=? [ F "delivered" ]` | What's the **worst** (adversarial strategy against the drone)? |
| `Pmax=? [ F "crashed" ]` | Worst-case crash probability |
| `Pmin=? [ F "crashed" ]` | Best-case crash probability |
| `Pmax=? [ F<=60 "delivered" ]` | Can the drone deliver within 60 transitions? (= 20 grid moves) |

### Threshold Checks
| Query | What it asks |
|-------|-------------|
| `P>=0.90 [ F "delivered" ]` | Is there a strategy with >= 90% delivery success? (true/false) |
| `P<=0.05 [ F "crashed" ]` | Can crash risk be kept below 5%? (true/false) |

### Reward Queries
| Query | What it asks |
|-------|-------------|
| `R{"time"}min=? [ F status>0 ]` | Expected mission time under the fastest strategy |
| `R{"jam_exposure"}max=? [ F status>0 ]` | Worst-case number of jammed cells encountered |

---

## Scenario File (`jamming_baseline.yaml`)

This YAML file is **not read by PRISM** — it's documentation for humans. It records the parameter choices for the baseline scenario so you can:
- Quickly see the setup without reading PRISM syntax
- Compare against other scenarios you create later
- Share with teammates who aren't reading the `.prism` file

When you change parameters in the PRISM model, update this file to match.

---

## State Space

| Variable | Range | Values |
|----------|-------|--------|
| `x` | 0–9 | 10 |
| `y` | 0–9 | 10 |
| `t` | 0–40 | 41 |
| `status` | 0–3 | 4 |
| `phase` | 0–2 | 3 |
| `cam` | 0–1 | 2 |
| `gps` | 0–1 | 2 |
| `comms` | 0–1 | 2 |
| **Total** | | **~394K states** |

This is well within PRISM's capabilities (it can handle millions).

---

## Integration with the City Model

The jamming model is designed to compose with the city/environment model. Integration points:

- **Grid constants** (`GW`, `GH`): Replace with shared constants from the city model
- **Movement labels** (`move_n`, etc.): Can synchronize with a city module that applies terrain-based time costs or building obstacles
- **Time variable** (`t`): Could be shared or replaced with a global clock module
- **Start/end points** (`SX`, `SY`, `EX`, `EY`): Should match the city model's designated hospitals

The simplest integration is to keep the models separate and run them against different grid configurations. A tighter integration would synchronize movement actions so that terrain costs (from the city model) and jamming costs (from this model) stack on the same transitions.

---

## Running the Model

```bash
# Check all properties
prism prism/models/jamming.prism prism/properties/jamming_queries.props

# Check a single property (e.g., line 7: max delivery probability)
prism prism/models/jamming.prism prism/properties/jamming_queries.props -prop 1

# Export the strategy (optimal policy)
prism prism/models/jamming.prism prism/properties/jamming_queries.props -prop 1 -exportstrat stdout
```
