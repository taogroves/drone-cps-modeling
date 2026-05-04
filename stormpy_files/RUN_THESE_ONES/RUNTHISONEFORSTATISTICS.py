#!/usr/bin/env python3
"""
Run many PRISM random-grid (randomized city) seeds and summarize variability of
robust maximum reachability Pmax=? [ !"crashed" U "goal" ] (interval MDP, robust).

The first N seeds (--candlestick-seeds, default 20) also run, on the same PRISM,
compute_bounded_reachability_curve and compute_bounded_safety_curve (max_k=--bounded-k-max),
store K=1..max_k, and plot two candlesticks (reachability vs finite-horizon safety).

Depends on stormpy, policyrandomizedbuildings.generate_random_prism, and
Propertiesv1.compute_bounded_reachability_curve.
"""

from __future__ import annotations

import random
from collections import deque
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

from Propertiesv1 import (
    compute_bounded_reachability_curve,
    compute_bounded_safety_curve,
)



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

	all_cells = {
		(x, y)
		for x in range(1, M + 1)
		for y in range(1, N + 1)
	}

	free_spaces = all_cells - (building_cells | hospital_cells)

	return building_cells, hospital_cells, free_spaces


def generate_random_prism(M: int = 5, N: int = 6, num_obstacles: int = 3, seed=None):
	"""Generate a PRISM file with a randomized city, hospital goal, and obstacles."""
	if seed is not None:
		random.seed(seed)

	buildings, hospital, free_spaces = generate_city_with_hospital(
		M=M, N=N, num_buildings=num_obstacles, seed=seed
	)

	obstacle_positions = sorted(buildings - hospital)
	goal_positions = sorted(hospital)

	def far_enough(x, y, targets, min_dist=10):
		for tx, ty in targets:
			if abs(x - tx) + abs(y - ty) < min_dist:
				return False
		return True

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
const double p = 0.1;
const double e = 0.02;
formula a = x * e;

const int M = {M};
const int N = {N};
const double eps = 0.0001;

// --- DYNAMICALLY GENERATED ELEMENTS ---
{goal_formula}
{crashed_formula}
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
"""

	filename = f"uav_random_seed_{seed}.prism"
	with open(filename, "w", encoding="utf-8") as prism_file:
		prism_file.write(prism_template)

	return filename




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


def _robust_pmax_init(
    model: Any,
    prism_program: Any,
    formula_str: str,
    *,
    produce_scheduler: bool = False,
) -> float:
    properties = stormpy.parse_properties(formula_str, prism_program)
    task = stormpy.CheckTask(properties[0].raw_formula)
    task.set_produce_schedulers(produce_scheduler)
    task.set_uncertainty_resolution_mode(stormpy.UncertaintyResolutionMode.ROBUST)
    result = stormpy.check_interval_mdp(model, task, stormpy.Environment())
    init = _initial_state(model)
    return _to_float(result.at(init))


def robust_reach_probability(
    prism_path: str, produce_scheduler: bool = False
) -> tuple[float, int]:
    """Return (Pmax robust at init, number of states) for the standard city formula."""
    prism_program = stormpy.parse_prism_program(prism_path)
    options = stormpy.BuilderOptions()
    options.set_build_state_valuations()
    if produce_scheduler:
        options.set_build_choice_labels()

    model = stormpy.build_sparse_interval_model_with_options(prism_program, options)
    n_states = int(getattr(model, "nr_states", len(list(model.states))))

    p = _robust_pmax_init(
        model,
        prism_program,
        'Pmax=? [ !"crashed" U "goal" ]',
        produce_scheduler=produce_scheduler,
    )
    return p, n_states


def run_seed(
    seed: int,
    m: int,
    n: int,
    num_obstacles: int,
    workspace: str,
    keep_prism: bool,
    bounded_k_max: int | None = None,
    jamming: dict | None = None,
) -> dict[str, Any]:
    suffix = "_jam" if jamming is not None else ""
    prism_name = f"uav_random_seed_{seed}.prism"
    prism_path = os.path.join(workspace, prism_name)
    with working_directory(workspace):
        generate_random_prism(M=m, N=n, num_obstacles=num_obstacles, seed=seed)
    try:
        try:
            pmax, n_states = robust_reach_probability(
                prism_path, produce_scheduler=False
            )
        except Exception as e:  # noqa: BLE001 - surface any model-check failure
            return {
                "seed": seed,
                "pmax_robust": math.nan,
                "n_states": -1,
                "bounded_curve": None,
                "bounded_error": "",
                "safety_curve": None,
                "safety_error": "",
                "error": str(e),
            }
        bounded: list[float] | None = None
        bounded_error = ""
        safety_curve: list[float] | None = None
        safety_error = ""
        if bounded_k_max is not None and bounded_k_max > 0:
            try:
                curve = compute_bounded_reachability_curve(
                    prism_path, max_k=bounded_k_max
                )
                bounded = curve[1 : bounded_k_max + 1]
            except Exception as e:  # noqa: BLE001
                bounded_error = str(e)
            try:
                scurve = compute_bounded_safety_curve(
                    prism_path, max_k=bounded_k_max
                )
                safety_curve = scurve[1 : bounded_k_max + 1]
            except Exception as e:  # noqa: BLE001
                safety_error = str(e)
        return {
            "seed": seed,
            "pmax_robust": pmax,
            "n_states": n_states,
            "bounded_curve": bounded,
            "bounded_error": bounded_error,
            "safety_curve": safety_curve,
            "safety_error": safety_error,
            "error": "",
        }
    finally:
        if not keep_prism and os.path.isfile(prism_path):
            try:
                os.remove(prism_path)
            except OSError:
                pass


def summarize(values: list[float]) -> dict[str, float]:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "stdev": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "median": float("nan"),
            "q1": float("nan"),
            "q3": float("nan"),
            "iqr": float("nan"),
        }
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


def plot_distribution(
    pmax_values: list[float],
    summary: dict[str, float],
    title: str,
    outfile: str | None,
    *,
    show: bool = True,
) -> None:
    clean = [p for p in pmax_values if not math.isnan(p)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax0 = axes[0]
    ax0.hist(clean, bins="auto", color="steelblue", edgecolor="black", alpha=0.85)
    ax0.set_xlabel(
        r"Robust $P_{\max}(\neg \mathrm{crashed} \,\mathrm{U}\, \mathrm{goal})$"
    )
    ax0.set_ylabel("Count")
    ax0.set_title("Histogram")

    ax1 = axes[1]
    ax1.boxplot(clean, vert=True, labels=[""])
    ax1.set_ylabel(r"$P_{\max}$ robust at init")
    ax1.set_title("Box plot")

    fig.suptitle(title, fontsize=12)

    lines = [
        f"n = {int(summary['n'])}",
        f"mean = {summary['mean']:.6f}",
        f"stdev = {summary['stdev']:.6f}",
        f"min / max = {summary['min']:.6f} / {summary['max']:.6f}",
        f"median = {summary['median']:.6f}",
        f"IQR [{summary['q1']:.6f}, {summary['q3']:.6f}]",
    ]
    fig.text(0.5, 0.02, "  |  ".join(lines), ha="center", fontsize=9, family="monospace")

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])

    if outfile:
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {outfile}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_bounded_candlesticks(
    per_k_values: list[list[float]],
    ks: list[int],
    title: str,
    outfile: str | None,
    *,
    yscale: str = "linear",
    show: bool = True,
    body_color: str = "steelblue",
    subplot_title: str = "Distribution over seeds per K",
    ylabel: str = "Maximum probability",
    footer: str = "Robust VI; IQR; min–max",
) -> None:
    """
    per_k_values[j] = probabilities for K = ks[j] across seeds (one candlestick per K).
    Body: IQR (Q1–Q3); wicks: min–max; white line: median.
    """
    fig, ax = plt.subplots(figsize=(12, 5.5))
    width = 0.35
    for j, k in enumerate(ks):
        vals = [v for v in per_k_values[j] if not math.isnan(v)]
        if not vals:
            continue
        stats = summarize(vals)
        lo, hi = stats["min"], stats["max"]
        q1, q3 = stats["q1"], stats["q3"]
        med = stats["median"]
        x = float(k)
        ax.vlines(x, lo, hi, colors="#333333", linewidth=1.25, zorder=1)
        bottom = min(q1, q3)
        top = max(q1, q3)
        height = top - bottom if top != bottom else max(top, 1e-6) * 1e-4 + 1e-6
        rect = mpatches.Rectangle(
            (x - width / 2, bottom),
            width,
            height,
            facecolor=body_color,
            edgecolor="black",
            linewidth=1,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.hlines(
            med,
            x - width / 2,
            x + width / 2,
            colors="white",
            linewidth=2.0,
            zorder=3,
        )

    ax.set_xlabel("Bound limit (K steps)")
    ax.set_ylabel(ylabel)
    ax.set_title(subplot_title)
    ax.set_yscale(yscale)
    ax.grid(True, linestyle=":", alpha=0.6)
    if ks:
        ax.set_xlim(float(ks[0]) - 0.5, float(ks[-1]) + 0.5)
    ax.set_ylim(-0.02, 1.02)
    fig.suptitle(title, fontsize=11, y=1.02)

    ax.text(
        0.99,
        0.02,
        footer,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="right",
        color="#444444",
    )

    plt.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
        print(f"Saved candlestick figure to {outfile}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-run randomized UAV grid PRISM models and plot P_max variability."
    )
    parser.add_argument("--num-seeds", type=int, default=50, help="How many seeds to run")
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="First seed; seeds used are base_seed, base_seed+1, ...",
    )
    parser.add_argument("--M", type=int, default=12, dest="M")
    parser.add_argument("--N", type=int, default=12, dest="N")
    parser.add_argument("--obstacles", type=int, default=30)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="randomized_city_batch",
        help="Workspace dir for generated PRISM files and CSV output",
    )
    parser.add_argument(
        "--figure",
        type=str,
        default="",
        help="If set, save matplotlib figure to this path (e.g. stats.png)",
    )
    parser.add_argument(
        "--keep-prism",
        action="store_true",
        help="Keep generated uav_env_<seed>.prism files",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not call plt.show() (useful headless with --figure)",
    )
    parser.add_argument(
        "--candlestick-seeds",
        type=int,
        default=20,
        help=(
            "First N seeds: bounded reachability + bounded safety curves (max_k=--bounded-k-max)"
        ),
    )
    parser.add_argument(
        "--bounded-k-max",
        type=int,
        default=20,
        help="max_k for compute_bounded_reachability_curve; candlestick/CSV use K=1..max_k",
    )
    parser.add_argument(
        "--figure-candlestick",
        type=str,
        default="",
        help="Path for reachability candlestick PNG (default: bounded_candlestick.png)",
    )
    parser.add_argument(
        "--figure-candlestick-safety",
        type=str,
        default="",
        help="Path for safety candlestick PNG (default: bounded_safety_candlestick.png)",
    )
    parser.add_argument(
        "--jamming",
        action="store_true",
        help="Enable adversarial jamming overlay (baseline scenario scaled to MxN)",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    workspace = os.path.join(out_dir, "prism_workspace")
    os.makedirs(workspace, exist_ok=True)

    seeds = [args.base_seed + i for i in range(args.num_seeds)]
    candle_n = min(args.candlestick_seeds, args.num_seeds)
    bounded_k = args.bounded_k_max if candle_n > 0 else None

    jamming_scenario = baseline_for_grid(args.M, args.N) if args.jamming else None
    if jamming_scenario is not None:
        print(f"Jamming overlay ENABLED (scaled baseline for {args.M}x{args.N})")

    rows: list[dict[str, Any]] = []
    for i, seed in enumerate(seeds):
        print(f"[{i + 1}/{len(seeds)}] seed={seed} ...", flush=True)
        use_bounded = bounded_k is not None and i < candle_n and bounded_k > 0
        row = run_seed(
            seed=seed,
            m=args.M,
            n=args.N,
            num_obstacles=args.obstacles,
            workspace=workspace,
            keep_prism=args.keep_prism,
            bounded_k_max=args.bounded_k_max if use_bounded else None,
            jamming=jamming_scenario,
        )
        rows.append(row)
        if row["error"]:
            print(f"    FAILED: {row['error']}", flush=True)
        else:
            extra = ""
            if row.get("bounded_curve"):
                extra = f"  (+ bounded K=1..{len(row['bounded_curve'])})"
            print(
                f"    Pmax robust = {row['pmax_robust']:.6f}  (|S|={row['n_states']}){extra}",
                flush=True,
            )
            if row.get("bounded_error"):
                print(f"    reachability curve FAILED: {row['bounded_error']}", flush=True)
            if row.get("safety_error"):
                print(f"    safety curve FAILED: {row['safety_error']}", flush=True)

    csv_path = os.path.join(out_dir, "batch_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "pmax_robust",
                "n_states",
                "error",
                "bounded_error",
                "safety_error",
            ],
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows)

    pmax_values = [float(r["pmax_robust"]) for r in rows]
    summary = summarize(pmax_values)
    print("\n--- Summary (valid runs) ---")
    for k, v in summary.items():
        if k == "n":
            print(f"  {k}: {int(v)}")
        else:
            print(f"  {k}: {v}")

    title = (
        f"Randomized city: M={args.M}, N={args.N}, obstacles={args.obstacles}, "
        f"seeds {args.base_seed}..{args.base_seed + args.num_seeds - 1}"
    )
    plot_distribution(
        pmax_values,
        summary,
        title,
        args.figure or None,
        show=not args.no_show,
    )

    bounded_csv = os.path.join(out_dir, "bounded_by_k.csv")
    safety_csv = os.path.join(out_dir, "safety_by_k.csv")
    bounded_ok = [
        r
        for r in rows
        if r.get("bounded_curve") and not r.get("bounded_error")
    ]
    safety_ok = [
        r
        for r in rows
        if r.get("safety_curve") and not r.get("safety_error")
    ]

    if args.bounded_k_max > 0 and (bounded_ok or safety_ok):
        k_full = args.bounded_k_max
        if bounded_ok:
            k_full = len(bounded_ok[0]["bounded_curve"])
        elif safety_ok:
            k_full = len(safety_ok[0]["safety_curve"])
        ks = list(range(1, k_full + 1))

        if bounded_ok:
            per_k = [[] for _ in range(k_full)]
            for r in bounded_ok:
                for ki, p in enumerate(r["bounded_curve"]):
                    per_k[ki].append(p)
            s0 = bounded_ok[0]["seed"]
            s1 = bounded_ok[-1]["seed"]
            candle_title = (
                f"Bounded safe reachability (!crashed U<=K goal), {len(bounded_ok)} seeds "
                f"(IDs {s0}..{s1}), K=1..{k_full}; "
                f"M={args.M}, N={args.N}, obstacles={args.obstacles}"
            )
            with open(bounded_csv, "w", newline="", encoding="utf-8") as f:
                bw = csv.writer(f)
                bw.writerow(["seed", "k", "pmax_robust_bounded"])
                for r in rows:
                    if not r.get("bounded_curve") or r.get("bounded_error"):
                        continue
                    for k, p in enumerate(r["bounded_curve"], start=1):
                        bw.writerow([r["seed"], k, p])

            fig_candle_path = args.figure_candlestick or os.path.join(
                out_dir, "bounded_candlestick.png"
            )
            plot_bounded_candlesticks(
                per_k,
                ks,
                candle_title,
                fig_candle_path,
                show=not args.no_show,
                subplot_title="Bounded safe reachability (!crashed U<=K goal)",
                footer="Robust bounded reachability VI; IQR; min–max",
            )
            print(f"Wrote {bounded_csv}")
        elif args.candlestick_seeds > 0:
            print(
                "Reachability candlestick skipped: no successful bounded reachability curves."
            )

        if safety_ok:
            per_k_s = [[] for _ in range(k_full)]
            for r in safety_ok:
                for ki, p in enumerate(r["safety_curve"]):
                    per_k_s[ki].append(p)
            s0 = safety_ok[0]["seed"]
            s1 = safety_ok[-1]["seed"]
            safety_title = (
                f"Bounded safety (finite-horizon robust G !crashed), {len(safety_ok)} seeds "
                f"(IDs {s0}..{s1}), K=1..{k_full}; "
                f"M={args.M}, N={args.N}, obstacles={args.obstacles}"
            )
            with open(safety_csv, "w", newline="", encoding="utf-8") as f:
                bw = csv.writer(f)
                bw.writerow(["seed", "k", "p_safety_bounded"])
                for r in rows:
                    if not r.get("safety_curve") or r.get("safety_error"):
                        continue
                    for k, p in enumerate(r["safety_curve"], start=1):
                        bw.writerow([r["seed"], k, p])

            fig_safety_path = args.figure_candlestick_safety or os.path.join(
                out_dir, "bounded_safety_candlestick.png"
            )
            plot_bounded_candlesticks(
                per_k_s,
                ks,
                safety_title,
                fig_safety_path,
                yscale="log",
                show=not args.no_show,
                body_color="#2ca02c",
                subplot_title="Bounded safety (robust G !crashed prefix)",
                ylabel="Safety probability",
                footer="Robust safety VI (same as plot_global_safety); IQR; min–max",
            )
            print(f"Wrote {safety_csv}")
        elif args.candlestick_seeds > 0:
            print(
                "Safety candlestick skipped: no successful bounded safety curves."
            )
    elif args.candlestick_seeds > 0 and args.bounded_k_max > 0:
        print(
            "Candlesticks skipped: no successful bounded curves "
            "(check bounded_error / safety_error in batch_results.csv)."
        )

    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
