"""
Side-by-side demo of the base UAV city MDP vs. the jamming-extended model.

Run:
    python demo_jamming.py                 # random seed
    python demo_jamming.py --seed 4242     # reproducible
    python demo_jamming.py --seed 4242 --M 5 --N 6 --obstacles 3

To reproduce the 30x30 procedural city map (seed=3875, multi-cell hospital
goal in the upper-left, drone start near the center, building-shaped
obstacles — same layout style as uav_random_seed_3875.prism):

    python demo_jamming.py --seed 3875 --M 30 --N 30 --obstacles 30 \
        --generator city --out demo_30.png

The `city` generator auto-strips the hardcoded `module policy` block so the
synthesized scheduler is unconstrained, and Storm auto-falls-back to a
step-bounded robust VI (default K = 4*(M+N)) when end-component elimination
fails on the larger sparse grid.

For the same seed and obstacle layout, generates two PRISM files (base and
base+jamming overlay), synthesizes the optimal robust scheduler for each via
Storm, and renders both grids side by side with obstacles, jam zones, goal,
start, and synthesized policy arrows. Each panel is annotated with the
Pmax (robust) probability of reaching the goal without crashing.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import stormpy

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from jamming_overlay import baseline_for_grid
import policyrandomizedbuildings
import RandomizedBuildingsv2

_GENERATORS = {
    "policy": policyrandomizedbuildings.generate_random_prism,
    "city":   RandomizedBuildingsv2.generate_random_prism,
}


_JAM_COLORS = {
    "camera": ("#cfe2ff", "Camera jam"),
    "gps":    ("#ffe5b4", "GPS jam"),
    "comms":  ("#e2d4f0", "Comms jam"),
}


def _robust_min_expected(action, prev):
    """Worst-case expected value over an interval distribution: assign all
    lower-bound mass, then push remaining slack to the smallest values
    (matches Propertiesv1.compute_bounded_reachability_curve semantics)."""
    bounded = []
    lower_sum = 0.0
    for transition in action.transitions:
        interval = transition.value()
        lo = float(interval.lower())
        up = float(interval.upper())
        bounded.append([prev[transition.column], lo, up])
        lower_sum += lo
    expected = sum(v * lo for v, lo, _ in bounded)
    remaining = max(0.0, 1.0 - lower_sum)
    bounded.sort(key=lambda r: r[0])
    for v, lo, up in bounded:
        if remaining <= 0.0:
            break
        add = min(up - lo, remaining)
        expected += v * add
        remaining -= add
    return expected


def _bounded_robust_synthesize(model, horizon: int):
    """Hand-rolled robust value iteration for `Pmax=? [!crashed U<=K goal]`
    on an interval MDP. Returns (pmax_at_init, scheduler_action_per_state).

    Used because stormpy.check_interval_mdp does not accept bounded
    reachability formulas, but on larger sparse grids unbounded checking
    fails on end-component elimination."""
    labels = model.labeling
    goal = {s.id for s in model.states if "goal" in labels.get_labels_of_state(s.id)}
    crashed = {s.id for s in model.states if "crashed" in labels.get_labels_of_state(s.id)}
    prev = {s.id: (1.0 if s.id in goal else 0.0) for s in model.states}
    chosen: dict[int, int] = {}  # state.id -> action local index

    for _ in range(horizon):
        curr = {}
        for state in model.states:
            sid = state.id
            if sid in goal:
                curr[sid] = 1.0
                continue
            if sid in crashed or len(state.actions) == 0:
                curr[sid] = 0.0
                continue
            best_v, best_i = -1.0, 0
            for i, action in enumerate(state.actions):
                v = _robust_min_expected(action, prev)
                if v > best_v:
                    best_v, best_i = v, i
            curr[sid] = best_v
            chosen[sid] = best_i
        prev = curr

    return prev[model.initial_states[0]], chosen, prev


def synthesize(prism_path: str, horizon: int | None = None):
    """Build the model and synthesize a robust Pmax scheduler.

    If `horizon` is given, performs hand-rolled robust value iteration for
    `Pmax=? [!crashed U<=horizon goal]` — required on larger sparse grids
    where Storm cannot eliminate end components for interval MDPs (and
    cannot check bounded formulas via `check_interval_mdp`).
    Otherwise uses unbounded `Pmax=? [ !crashed U goal ]` via Storm.
    """
    program = stormpy.parse_prism_program(prism_path)

    options = stormpy.BuilderOptions()
    options.set_build_state_valuations()
    options.set_build_choice_labels()
    model = stormpy.build_sparse_interval_model_with_options(program, options)

    if horizon is not None:
        pmax, chosen, state_values = _bounded_robust_synthesize(model, horizon)
        scheduler = None  # bounded path uses chosen-dict directly
    else:
        properties = stormpy.parse_properties(
            'Pmax=? [ !"crashed" U "goal" ]', program)
        task = stormpy.CheckTask(properties[0].raw_formula)
        task.set_produce_schedulers(True)
        task.set_uncertainty_resolution_mode(stormpy.UncertaintyResolutionMode.ROBUST)
        result = stormpy.check_interval_mdp(model, task, stormpy.Environment())
        pmax = result.at(model.initial_states[0])
        scheduler = result.scheduler
        chosen = None
        state_values = {s.id: result.at(s.id) for s in model.states}

    x_var = next(v for v in program.variables if v.name == "x")
    y_var = next(v for v in program.variables if v.name == "y")
    jcrashed_var = next(
        (v for v in program.variables if v.name == "jcrashed"), None
    )
    return {
        "program": program,
        "model": model,
        "scheduler": scheduler,    # None when horizon is set
        "chosen": chosen,           # state.id -> local action index, or None
        "pmax": pmax,
        "state_values": state_values,  # state.id -> Pmax at that state
        "x": x_var,
        "y": y_var,
        "jcrashed": jcrashed_var,
    }


def _action_for_choice(model, state, local_choice_index):
    """Resolve action label for the scheduler's local choice index."""
    cl = model.choice_labeling
    if cl is None:
        return None
    try:
        global_idx = state.actions[local_choice_index].id
    except Exception:
        global_idx = local_choice_index
    for label in cl.get_labels():
        if cl.get_choices(label).get(global_idx):
            return label
    return None


def _extract_layout(synth, M, N):
    """Pull obstacles, goal, start, and per-cell scheduler action from a
    synthesized model. Cells are 1-indexed."""
    model = synth["model"]
    sched = synth["scheduler"]
    chosen = synth["chosen"]
    state_values = synth["state_values"]
    val = model.state_valuations
    labels = model.labeling
    x_var, y_var, jc_var = synth["x"], synth["y"], synth["jcrashed"]

    obstacles, goals, start_pos = set(), set(), None
    cell_action: dict[tuple[int, int], str] = {}
    zero_value_eps = 1e-9

    for state in model.states:
        s_id = state.id
        x = int(val.get_value(s_id, x_var))
        y = int(val.get_value(s_id, y_var))
        if jc_var is not None and val.get_value(s_id, jc_var):
            continue

        st_labels = labels.get_labels_of_state(s_id)
        if "crashed" in st_labels:
            obstacles.add((x, y))
            continue
        if "goal" in st_labels:
            goals.add((x, y))
        if "init" in st_labels:
            start_pos = (x, y)

        if sched is not None:
            choice = sched.get_choice(state)
            if choice.defined:
                local_idx = choice.get_deterministic_choice()
            elif state_values.get(s_id, 0.0) > zero_value_eps:
                # Tie: scheduler considered all actions equivalent.
                # Pick the first one so the cell still renders an arrow.
                local_idx = 0
            else:
                # Pmax = 0 at this cell — genuinely no path to goal.
                # Leave blank to communicate that visually.
                local_idx = None
        else:
            local_idx = chosen.get(s_id) if chosen is not None else None
        if local_idx is not None:
            act = _action_for_choice(model, state, local_idx)
            if act in ("up", "down", "left", "right"):
                cell_action[(x, y)] = act

    return obstacles, goals, start_pos, cell_action


def _draw_panel(ax, M, N, obstacles, goals, start, cell_action,
                jam_scenario, title, pmax):
    """Render one grid panel. `goals` is a set of (x,y) cells (the city
    generator produces multi-cell hospital goals)."""
    # Base cells
    for x in range(1, M + 1):
        for y in range(1, N + 1):
            face = "white"
            if (x, y) in goals:
                face = "#d9ead3"      # green
            elif (x, y) in obstacles:
                face = "#f4cccc"      # red
            ax.add_patch(patches.Rectangle(
                (x - 1, y - 1), 1, 1, linewidth=1,
                edgecolor="black", facecolor=face,
            ))

    # Jam zones (translucent overlay)
    if jam_scenario is not None:
        for key, (color, _) in _JAM_COLORS.items():
            z = jam_scenario.get(key)
            if not z or not z.get("enabled", False):
                continue
            xmin, xmax = z["xmin"], z["xmax"]
            ymin, ymax = z["ymin"], z["ymax"]
            ax.add_patch(patches.Rectangle(
                (xmin - 1, ymin - 1), xmax - xmin + 1, ymax - ymin + 1,
                linewidth=0, facecolor=color, alpha=0.55, zorder=2,
            ))

    # Start marker
    if start is not None:
        ax.plot(start[0] - 0.5, start[1] - 0.5, marker="o",
                markersize=18, color="#212121", zorder=3, alpha=0.45)

    # Policy arrows
    arrow_len = 0.35
    deltas = {"up": (0, arrow_len), "down": (0, -arrow_len),
              "right": (arrow_len, 0), "left": (-arrow_len, 0)}
    for (x, y), act in cell_action.items():
        if (x, y) in goals or (x, y) in obstacles:
            continue
        dx, dy = deltas[act]
        ax.arrow(x - 0.5, y - 0.5, dx, dy,
                 head_width=0.15, head_length=0.15,
                 fc="#1f77b4", ec="#1f77b4",
                 length_includes_head=True, zorder=4)

    ax.set_xlim(0, M)
    ax.set_ylim(0, N)
    ax.set_xticks(range(M + 1))
    ax.set_yticks(range(N + 1))
    ax.set_xticklabels([str(i) for i in range(1, M + 2)])
    ax.set_yticklabels([str(i) for i in range(1, N + 2)])
    ax.set_aspect("equal")
    pmax_str = f"{pmax:.4f}" if pmax >= 1e-4 else f"{pmax:.2e}"
    ax.set_title(f"{title}\nPmax = {pmax_str}")


def _legend_handles(jam_scenario):
    h = [
        patches.Patch(facecolor="#d9ead3", edgecolor="black", label="Goal"),
        patches.Patch(facecolor="#f4cccc", edgecolor="black", label="Obstacle"),
    ]
    if jam_scenario is not None:
        for key, (color, label) in _JAM_COLORS.items():
            z = jam_scenario.get(key)
            if z and z.get("enabled", False):
                h.append(patches.Patch(facecolor=color, alpha=0.55, label=label))
    return h


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--M", type=int, default=5)
    parser.add_argument("--N", type=int, default=6)
    parser.add_argument("--obstacles", type=int, default=3)
    parser.add_argument(
        "--horizon", type=int, default=None,
        help=("Optional step-bound K for `Pmax=? [F<=K goal]`. Required on "
              "larger sparse grids where Storm cannot eliminate end "
              "components for interval MDPs. If omitted, the demo first "
              "tries unbounded reachability and falls back to K=4*(M+N) "
              "on failure."),
    )
    parser.add_argument(
        "--generator", choices=list(_GENERATORS.keys()), default="policy",
        help=("Which base-model generator to use. 'policy' = single-cell "
              "goal, start at (1,1), simple obstacles. 'city' = multi-cell "
              "hospital goal, randomized inner start, building-shaped "
              "obstacles (matches RandomizedBuildingsv2.py)."),
    )
    parser.add_argument("--out", type=str, default=None,
                        help="Optional path to save the figure (PNG).")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip the interactive plt.show() window.")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(1000, 9999)
    print(f"=== UAV jamming demo (seed={seed}, grid={args.M}x{args.N}) ===")

    scenario = baseline_for_grid(args.M, args.N)
    generate_random_prism = _GENERATORS[args.generator]

    # The 'city' generator includes a hardcoded `module policy` that we
    # strip for synthesis; 'policy' has no such block so the kwarg is unused.
    extra = {"include_manual_policy": False} if args.generator == "city" else {}

    base_path = generate_random_prism(
        M=args.M, N=args.N, num_obstacles=args.obstacles, seed=seed, **extra)
    jam_path = generate_random_prism(
        M=args.M, N=args.N, num_obstacles=args.obstacles, seed=seed,
        jamming=scenario, **extra)
    print(f"  base PRISM: {base_path}")
    print(f"  jam  PRISM: {jam_path}")

    def _synth_with_fallback(path, label):
        horizon = args.horizon
        if horizon is None:
            try:
                print(f"Synthesizing {label} policy (unbounded) ...")
                return synthesize(path), None
            except RuntimeError as e:
                if "end components" not in str(e):
                    raise
                horizon = 4 * (args.M + args.N)
                print(f"  unbounded failed (end components); "
                      f"retrying with horizon K={horizon}")
        print(f"Synthesizing {label} policy (F<={horizon}) ...")
        return synthesize(path, horizon=horizon), horizon

    base_synth, base_K = _synth_with_fallback(base_path, "base")
    print(f"  Pmax (base) = {base_synth['pmax']:.6f}  |S|={base_synth['model'].nr_states}")
    jam_synth, jam_K = _synth_with_fallback(jam_path, "jamming")
    print(f"  Pmax (jam)  = {jam_synth['pmax']:.6f}  |S|={jam_synth['model'].nr_states}")
    horizon_label = base_K or jam_K  # for plot title

    base_obs, base_goal, base_start, base_act = _extract_layout(
        base_synth, args.M, args.N)
    jam_obs, jam_goal, jam_start, jam_act = _extract_layout(
        jam_synth, args.M, args.N)

    fig, (ax_base, ax_jam) = plt.subplots(1, 2, figsize=(12, 6))
    _draw_panel(ax_base, args.M, args.N, base_obs, base_goal, base_start,
                base_act, jam_scenario=None,
                title=f"Base (seed={seed})", pmax=base_synth["pmax"])
    _draw_panel(ax_jam, args.M, args.N, jam_obs, jam_goal, jam_start,
                jam_act, jam_scenario=scenario,
                title=f"Base + Jamming (seed={seed})", pmax=jam_synth["pmax"])

    fig.legend(handles=_legend_handles(scenario), loc="lower center",
               ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    horizon_suffix = (
        f", step horizon K={horizon_label}" if horizon_label else ""
    )
    fig.suptitle(
        f"UAV city MDP — synthesized robust policy "
        f"(grid {args.M}x{args.N}, {args.obstacles} obstacles{horizon_suffix})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))

    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved figure: {args.out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
