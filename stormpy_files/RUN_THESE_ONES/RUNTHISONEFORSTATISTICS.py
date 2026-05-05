#!/usr/bin/env python3
"""
Batch-run randomized UAV city environments and collect bounded Pmax statistics
for three properties × 6 organ types × 2 jamming conditions.

Properties (bounded to K steps):
  avoidance:    Pmax=? [ !"crashed" U<=K "goal_<organ>" ]
  reachability: Pmax=? [ F<=K "goal_<organ>" ]
  safety:       Pmax=? [ G<=K !"crashed" ]  (organ-independent)

Outputs (in --out-dir):
  batch_results.csv  — per (seed, organ, jamming): unbounded Pmax values
  bounded_by_k.csv   — per (seed, organ, jamming, k): bounded Pmax curves
"""

from __future__ import annotations

import random
import argparse
import csv
import math
import os
import statistics
import sys
from contextlib import contextmanager
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import stormpy

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from Propertiesv1 import compute_global_safety, compute_all_bounded_curves_per_organ
from jamming_overlay import baseline_for_grid, patch_base_for_jamming, emit_jamming_block


# ── Organ types ───────────────────────────────────────────────────────────────

ORGAN_TYPES = ["Heart", "Lungs", "Liver", "Intestines", "Pancreas", "Kidneys"]


def _assign_organs(goal_positions, seed=None):
    """
    Assign organ types to hospital cells using a separate RNG so city-gen
    seeding is not affected. Each organ gets at least one cell when possible;
    remaining cells are assigned randomly.
    """
    rng = random.Random(seed * 31337 if seed is not None else None)
    goals = list(goal_positions)
    rng.shuffle(goals)
    assignments = {}
    for i, goal in enumerate(goals[: len(ORGAN_TYPES)]):
        assignments[goal] = ORGAN_TYPES[i]
    for goal in goals[len(ORGAN_TYPES) :]:
        assignments[goal] = rng.choice(ORGAN_TYPES)
    return assignments  # {(x, y): organ_name}


# ── City / PRISM generation ───────────────────────────────────────────────────

def make_building(M, N, x, y, w, h):
	return {
		(i, j)
		for i in range(x, x + w)
		for j in range(y, y + h)
		if 1 <= i <= M and 1 <= j <= N
	}


def generate_city_with_hospital(M, N, num_buildings, seed=None):
	if seed is not None:
		random.seed(seed)

	buildings = []
	hospital = None
	occupied = set()

	def can_place(candidate):
		for x, y in candidate:
			for dx in (-1, 0, 1):
				for dy in (-1, 0, 1):
					if (x + dx, y + dy) in occupied:
						return False
		return True

	def in_inner_region(x, y, M, N, margin=0.1):
		x_min = int(M * margin)
		x_max = int(M * (1 - margin))
		y_min = int(N * margin)
		y_max = int(N * (1 - margin))
		return (x_min <= x <= x_max) and (y_min <= y <= y_max)

	def place_building(is_hospital=False):
		nonlocal hospital
		max_w = min(6, max(2, M - 1))
		max_h = min(6, max(2, N - 1))
		if max_w < 2 or max_h < 2:
			raise ValueError("Grid is too small to place a building")
		for _ in range(1000):
			if random.random() < 0.5 and max_w >= 3:
				w = random.randint(3, max_w)
				h = 2
			else:
				w = 2
				h = random.randint(3, max_h)
			x = random.randint(2, M - w + 1)
			y = random.randint(2, N - h + 1)
			candidate = make_building(M, N, x, y, w, h)
			if is_hospital and any(in_inner_region(cx, cy, M, N) for cx, cy in candidate):
				continue
			if can_place(candidate):
				buildings.append(candidate)
				occupied.update(candidate)
				if is_hospital:
					hospital = candidate
				return True
		return False

	place_building(is_hospital=True)
	for _ in range(num_buildings):
		place_building()

	building_cells = set().union(*buildings) if buildings else set()
	hospital_cells = hospital if hospital is not None else set()
	all_cells = {(x, y) for x in range(1, M + 1) for y in range(1, N + 1)}
	free_spaces = all_cells - (building_cells | hospital_cells)
	return building_cells, hospital_cells, free_spaces


def generate_random_prism(M: int = 5, N: int = 6, num_obstacles: int = 3, seed=None):
	"""
	Generate a PRISM file with per-organ goal labels.
	Returns (filename, organ_assignments) where organ_assignments is {(x,y): organ_name}.
	"""
	if seed is not None:
		random.seed(seed)

	buildings, hospital, free_spaces = generate_city_with_hospital(
		M=M, N=N, num_buildings=num_obstacles, seed=seed
	)

	obstacle_positions = sorted(buildings - hospital)
	goal_positions = sorted(hospital)
	organ_assignments = _assign_organs(goal_positions, seed=seed)

	# Per-organ goal formulas
	organ_formula_lines = []
	for organ in ORGAN_TYPES:
		cells = sorted(pos for pos, org in organ_assignments.items() if org == organ)
		if cells:
			parts = [f"(x = {cx} & y = {cy})" for cx, cy in cells]
			organ_formula_lines.append(
				f'formula goal_{organ.lower()} = {" | ".join(parts)};'
			)
		else:
			organ_formula_lines.append(f"formula goal_{organ.lower()} = false;")
	organ_formulas_str = "\n".join(organ_formula_lines)

	organ_label_lines = "\n".join(
		f'label "goal_{o.lower()}" = goal_{o.lower()};' for o in ORGAN_TYPES
	)

	def far_enough(x, y, targets, min_dist=10):
		return all(abs(x - tx) + abs(y - ty) >= min_dist for tx, ty in targets)

	def sample_start(free_spaces, hospital_cells, min_dist=10):
		free_list = list(free_spaces)
		for _ in range(10000):
			x, y = random.choice(free_list)
			if far_enough(x, y, hospital_cells, min_dist):
				return x, y
		raise ValueError("Could not find valid start position")

	start_x, start_y = sample_start(free_spaces, hospital, min_dist=10)

	goal_strings = [f"(x = {gx} & y = {gy})" for gx, gy in goal_positions]
	goal_formula = f"formula goal = {' | '.join(goal_strings)};"
	obs_strings = [f"(x = {ox} & y = {oy})" for ox, oy in obstacle_positions]
	crashed_formula = f"formula crashed = {' | '.join(obs_strings)};"

	prism_template = f"""mdp

// Model parameters
const double p = 0.01;
const double e = 0.01;
formula a = e;

const int M = {M};
const int N = {N};
const double eps = 0.0001;

// --- DYNAMICALLY GENERATED ELEMENTS ---
{goal_formula}
{crashed_formula}
{organ_formulas_str}
// --------------------------------------

module env
	x : [1..M] init {start_x};
	y : [1..N] init {start_y};
	[up]    !crashed & y < N -> [max(eps,1-3*(p+a)), min(1-3*(p-a),1)] : (y'=min(y+1,N))
			  + [max(eps,p-a), min(p+a,1)] : (y'=min(y+1,N)) & (x'=max(x-1,1))
			  + [max(eps,p-a), min(p+a,1)] : (y'=min(y+1,N)) & (x'=min(x+1,M))
			  + [max(eps,p-a), min(p+a,1)] : true;
	[down]  !crashed & y > 1 -> [max(eps,1-3*(p+a)), min(1-3*(p-a),1)] : (y'=max(y-1,1))
			  + [max(eps,p-a), min(p+a,1)] : (y'=max(y-1,1)) & (x'=max(x-1,1))
			  + [max(eps,p-a), min(p+a,1)] : (y'=max(y-1,1)) & (x'=min(x+1,M))
			  + [max(eps,p-a), min(p+a,1)] : true;
	[right] !crashed & x < M -> [max(eps,1-3*(p+a)), min(1-3*(p-a),1)] : (x'=min(x+1,M))
			  + [max(eps,p-a), min(p+a,1)] : (x'=min(x+1,M)) & (y'=min(y+1,N))
			  + [max(eps,p-a), min(p+a,1)] : (x'=max(x-1,1))
			  + [max(eps,p-a), min(p+a,1)] : true;
	[left] 	!crashed & x > 1 -> [max(eps,1-3*(p+a)), min(1-3*(p-a),1)] : (x'=max(x-1,1))
			  + [max(eps,p-a), min(p+a,1)] : (x'=max(x-1,1)) & (y'=min(y+1,N))
			  + [max(eps,p-a), min(p+a,1)] : (x'=min(x+1,M))
			  + [max(eps,p-a), min(p+a,1)] : true;
endmodule

label "crashed" = crashed;
label "goal" = goal;
{organ_label_lines}
"""

	filename = f"uav_random_seed_{seed}.prism"
	with open(filename, "w", encoding="utf-8") as prism_file:
		prism_file.write(prism_template)

	return filename, organ_assignments


# ── Helpers ───────────────────────────────────────────────────────────────────

@contextmanager
def working_directory(path: str) -> Any:
    prev = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _to_float(x: Any) -> float:
    if isinstance(x, (float, int)):
        return float(x)
    if hasattr(x, "to_double"):
        return float(x.to_double())
    return float(x)


def _initial_state(model: Any) -> int:
    labeling = model.labeling
    for state in model.states:
        s_id = int(state.id) if hasattr(state, "id") else int(state)
        if "init" in labeling.get_labels_of_state(s_id):
            return s_id
    raise RuntimeError("Could not resolve an initial state for the model.")


def _robust_pmax_init(model: Any, prism_program: Any, formula_str: str) -> float:
    properties = stormpy.parse_properties(formula_str, prism_program)
    task = stormpy.CheckTask(properties[0].raw_formula)
    task.set_produce_schedulers(False)
    task.set_uncertainty_resolution_mode(stormpy.UncertaintyResolutionMode.ROBUST)
    result = stormpy.check_interval_mdp(model, task, stormpy.Environment())
    return _to_float(result.at(_initial_state(model)))


def _apply_jamming(prism_path: str, jamming: dict, workspace: str) -> str:
    """Patch base PRISM with jamming overlay. Returns path to the patched file."""
    with open(prism_path, encoding="utf-8") as f:
        base = f.read()
    patched = patch_base_for_jamming(base) + emit_jamming_block(jamming)
    jam_path = os.path.join(
        workspace,
        os.path.basename(prism_path).replace(".prism", "_jam.prism"),
    )
    with open(jam_path, "w", encoding="utf-8") as f:
        f.write(patched)
    return jam_path


# ── Seed runner ───────────────────────────────────────────────────────────────

def run_seed(
    seed: int,
    m: int,
    n: int,
    num_obstacles: int,
    workspace: str,
    keep_prism: bool,
    bounded_k_max: int | None,
    run_jamming: bool,
) -> list[dict[str, Any]]:
    """
    Process one seed for all organs × jamming conditions.
    Returns a list of result dicts (one per organ × condition).
    """
    prism_name = f"uav_random_seed_{seed}.prism"
    prism_path = os.path.join(workspace, prism_name)
    with working_directory(workspace):
        _, organ_assignments = generate_random_prism(
            M=m, N=n, num_obstacles=num_obstacles, seed=seed
        )

    jamming_scenario = baseline_for_grid(m, n) if run_jamming else None
    jam_path = _apply_jamming(prism_path, jamming_scenario, workspace) if jamming_scenario else None

    conditions: list[tuple[int, str]] = [(0, prism_path)]
    if jam_path:
        conditions.append((1, jam_path))

    rows: list[dict[str, Any]] = []

    for jam_flag, path in conditions:
        # Unbounded safety — organ-independent, compute once per condition
        try:
            pmax_safety_global = compute_global_safety(path)
        except Exception as e:
            pmax_safety_global = math.nan
            print(f"    [seed={seed} jam={jam_flag}] safety VI failed: {e}", flush=True)

        # Bounded curves — all organs in one efficient pass
        bounded_results: dict | None = None
        if bounded_k_max and bounded_k_max > 0:
            try:
                bounded_results = compute_all_bounded_curves_per_organ(
                    path, max_k=bounded_k_max, organ_types=ORGAN_TYPES
                )
            except Exception as e:
                print(
                    f"    [seed={seed} jam={jam_flag}] bounded VI failed: {e}", flush=True
                )

        # Unbounded avoidance + reachability — build model once, query per organ
        prism_program = stormpy.parse_prism_program(path)
        options = stormpy.BuilderOptions()
        options.set_build_state_valuations()
        model = stormpy.build_sparse_interval_model_with_options(prism_program, options)
        n_states = int(getattr(model, "nr_states", len(list(model.states))))

        for organ in ORGAN_TYPES:
            goal_label = f"goal_{organ.lower()}"
            row: dict[str, Any] = {
                "seed": seed,
                "organ": organ,
                "jamming": jam_flag,
                "pmax_avoidance": math.nan,
                "pmax_reachability": math.nan,
                "pmax_safety": pmax_safety_global,
                "n_states": n_states,
                "avoidance_curve": None,
                "reach_curve": None,
                "safety_curve": None,
                "error": "",
            }
            try:
                row["pmax_avoidance"] = _robust_pmax_init(
                    model, prism_program,
                    f'Pmax=? [ !"crashed" U "{goal_label}" ]',
                )
                row["pmax_reachability"] = _robust_pmax_init(
                    model, prism_program,
                    f'Pmax=? [ F "{goal_label}" ]',
                )
            except Exception as e:
                row["error"] = str(e)

            if bounded_results and organ in bounded_results:
                avoid_c, reach_c, safe_c = bounded_results[organ]
                row["avoidance_curve"] = avoid_c[1:]   # drop K=0
                row["reach_curve"]     = reach_c[1:]
                row["safety_curve"]    = safe_c[1:]

            rows.append(row)

    if not keep_prism:
        for p in filter(None, [prism_path, jam_path]):
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    return rows


# ── Statistics / plotting ─────────────────────────────────────────────────────

def summarize(values: list[float]) -> dict[str, float]:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return {k: float("nan") for k in ("n", "mean", "stdev", "min", "max", "median", "q1", "q3", "iqr")}
    clean.sort()
    try:
        qs = statistics.quantiles(clean, n=4, method="inclusive")
    except TypeError:
        qs = statistics.quantiles(clean, n=4)
    q1, _, q3 = qs[0], qs[1], qs[2]
    return {
        "n": float(len(clean)),
        "mean": statistics.fmean(clean),
        "stdev": statistics.pstdev(clean) if len(clean) > 1 else 0.0,
        "min": clean[0],
        "max": clean[-1],
        "median": statistics.median(clean),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }


def plot_organ_curves(
    bounded_rows: list[dict],
    jam_flag: int,
    property_key: str,
    property_label: str,
    k_max: int,
    outfile: str | None,
    show: bool = True,
) -> None:
    """
    Plot median K-step curve per organ for one jamming condition and one property.
    bounded_rows: list of rows from bounded_by_k.csv filtered to one jam_flag.
    """
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    fig, ax = plt.subplots(figsize=(10, 5))

    for organ, color in zip(ORGAN_TYPES, colors):
        organ_rows = [r for r in bounded_rows if r["organ"] == organ and r["jamming"] == jam_flag]
        if not organ_rows:
            continue
        # Group by K
        by_k: dict[int, list[float]] = {}
        for r in organ_rows:
            k = int(r["k"])
            val = float(r[property_key])
            by_k.setdefault(k, []).append(val)
        ks = sorted(by_k)
        medians = [statistics.median(by_k[k]) for k in ks]
        ax.plot(ks, medians, label=organ, color=color, linewidth=1.8, marker="o", markersize=3)

    jam_str = "with jamming" if jam_flag else "no jamming"
    ax.set_title(f"{property_label} — {jam_str}")
    ax.set_xlabel("Bound limit (K steps)")
    ax.set_ylabel("Maximum probability")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {outfile}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect bounded Pmax statistics per organ and jamming condition."
    )
    parser.add_argument("--num-seeds", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--M", type=int, default=12, dest="M")
    parser.add_argument("--N", type=int, default=12, dest="N")
    parser.add_argument("--obstacles", type=int, default=30)
    parser.add_argument("--bounded-k-max", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default="organ_batch_results")
    parser.add_argument("--keep-prism", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-jamming", action="store_true", help="Skip jamming runs")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    workspace = os.path.join(out_dir, "prism_workspace")
    os.makedirs(workspace, exist_ok=True)

    seeds = [args.base_seed + i for i in range(args.num_seeds)]
    run_jamming = not args.no_jamming

    all_rows: list[dict[str, Any]] = []
    for i, seed in enumerate(seeds):
        label = f"[{i+1}/{len(seeds)}] seed={seed}"
        print(f"{label} ...", flush=True)
        try:
            rows = run_seed(
                seed=seed,
                m=args.M,
                n=args.N,
                num_obstacles=args.obstacles,
                workspace=workspace,
                keep_prism=args.keep_prism,
                bounded_k_max=args.bounded_k_max,
                run_jamming=run_jamming,
            )
            all_rows.extend(rows)
            for r in rows:
                if r["error"]:
                    print(
                        f"    organ={r['organ']} jam={r['jamming']} ERROR: {r['error']}",
                        flush=True,
                    )
                else:
                    print(
                        f"    organ={r['organ']:12s} jam={r['jamming']}"
                        f"  avoid={r['pmax_avoidance']:.4f}"
                        f"  reach={r['pmax_reachability']:.4f}"
                        f"  safe={r['pmax_safety']:.4f}",
                        flush=True,
                    )
        except Exception as e:
            print(f"    SEED FAILED: {e}", flush=True)

    # Write batch_results.csv
    batch_csv = os.path.join(out_dir, "batch_results.csv")
    with open(batch_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "seed", "organ", "jamming",
                "pmax_avoidance", "pmax_reachability", "pmax_safety",
                "n_states", "error",
            ],
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {batch_csv}")

    # Write bounded_by_k.csv
    bounded_csv = os.path.join(out_dir, "bounded_by_k.csv")
    bounded_rows_flat: list[dict] = []
    k_max = args.bounded_k_max
    for r in all_rows:
        if not r.get("avoidance_curve"):
            continue
        for k_idx in range(len(r["avoidance_curve"])):
            bounded_rows_flat.append({
                "seed":        r["seed"],
                "organ":       r["organ"],
                "jamming":     r["jamming"],
                "k":           k_idx + 1,
                "avoidance":   r["avoidance_curve"][k_idx],
                "reachability": r["reach_curve"][k_idx],
                "safety":      r["safety_curve"][k_idx],
            })

    with open(bounded_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["seed", "organ", "jamming", "k", "avoidance", "reachability", "safety"],
        )
        w.writeheader()
        w.writerows(bounded_rows_flat)
    print(f"Wrote {bounded_csv}")

    # Plots — one per (property × jamming condition)
    show = not args.no_show
    for jam_flag in ([0, 1] if run_jamming else [0]):
        for prop_key, prop_label in [
            ("avoidance",    "Avoidance  Pmax [ !crashed U<=K goal ]"),
            ("reachability", "Reachability  Pmax [ F<=K goal ]"),
            ("safety",       "Safety  Pmax [ G<=K !crashed ]"),
        ]:
            jam_str = "jam" if jam_flag else "nojam"
            fig_path = os.path.join(out_dir, f"{prop_key}_{jam_str}.png")
            plot_organ_curves(
                bounded_rows_flat,
                jam_flag=jam_flag,
                property_key=prop_key,
                property_label=prop_label,
                k_max=k_max,
                outfile=fig_path,
                show=show,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
