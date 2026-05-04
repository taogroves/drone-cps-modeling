"""
Jamming overlay for the base UAV city MDP.

The base model in `randomizedbuildings.py` emits a self-contained MDP with
`module env` over `x:[1..M], y:[1..N]` synchronizing on `[up] [down] [left]
[right]`. This module produces a PRISM block that, when concatenated onto
that base file, adds adversarial jamming as a parallel module synchronized
on the same movement actions.

Design choices (deliberately simplified vs. the standalone Riley jamming
model in `prism/models/jamming.prism`):

  * No push_through / slow_down nondeterminism — the drone always takes the
    `slow_down` response in jam zones. This collapses the second MDP choice
    while preserving the time/risk tradeoff at the property level.
  * Jam activation is folded into expected crash probability per cell:
        p_jam_crash(x,y) = sum over active jam types of
                           p_active_type * p_crash_type_slow
    rather than a separate probabilistic activation step.
  * Jamming is keyed on the *source* cell (where the drone is when it
    commits to a move) rather than the destination — avoids needing
    (x',y') lookahead inside the overlay.
  * Time is exposed via a "time" reward structure, not a state variable.

Coordinates are 1-indexed to match `randomizedbuildings.py` (x in [1..M],
y in [1..N]).
"""

from typing import Optional, TypedDict


class Zone(TypedDict, total=False):
    enabled: bool
    xmin: int
    xmax: int
    ymin: int
    ymax: int
    activation_prob: float
    crash_prob_slow: float
    time_extra: int


class JammingScenario(TypedDict, total=False):
    base_crash_prob: float
    normal_move_time: int
    slow_move_time: int
    camera: Zone
    gps: Zone
    comms: Zone


# Sensible default sized for Gabe's 5x6 grid. Can be passed straight into
# generate_random_prism(jamming=BASELINE_5x6).
# Riley's baseline jamming layout, expressed in its native 10x10 0-indexed
# coordinate system. Source of truth: scenarios/jamming_baseline.yaml on the
# `riley` branch. `baseline_for_grid` rescales these onto an (M,N) Gabe grid.
_RILEY_BASELINE_10x10 = {
    "base_crash_prob": 0.001,
    "normal_move_time": 1,
    "slow_move_time": 2,
    # bounds are inclusive 0-indexed [xmin..xmax, ymin..ymax] on a 10x10 grid
    "camera": {
        "enabled": True,
        "bounds_0idx_10x10": (3, 5, 3, 5),
        "activation_prob": 0.80,
        "crash_prob_slow": 0.05,
        "time_extra": 0,
    },
    "gps": {
        "enabled": True,
        "bounds_0idx_10x10": (6, 8, 2, 4),
        "activation_prob": 0.70,
        "crash_prob_slow": 0.03,
        "time_extra": 1,
    },
    "comms": {
        "enabled": True,
        "bounds_0idx_10x10": (4, 7, 6, 8),
        "activation_prob": 0.75,
        "crash_prob_slow": 0.02,
        "time_extra": 1,
    },
}


def _rescale_bound(v: int, src_max: int, dst_max: int) -> int:
    """Map a 0-indexed coord on a [0..src_max] axis onto a 1-indexed
    coord on a [1..dst_max] axis. Rounds to nearest, clamps to range."""
    if src_max <= 0:
        return 1
    scaled = round(v * (dst_max - 1) / src_max) + 1
    return max(1, min(dst_max, scaled))


def baseline_for_grid(M: int, N: int) -> "JammingScenario":
    """Return a jamming scenario sized for an MxN Gabe grid by scaling
    Riley's 10x10 baseline layout. Activation probabilities, crash
    probabilities, and time penalties are preserved verbatim — only zone
    bounds are remapped."""
    out: JammingScenario = {
        "base_crash_prob": _RILEY_BASELINE_10x10["base_crash_prob"],
        "normal_move_time": _RILEY_BASELINE_10x10["normal_move_time"],
        "slow_move_time": _RILEY_BASELINE_10x10["slow_move_time"],
    }
    for key in ("camera", "gps", "comms"):
        z = _RILEY_BASELINE_10x10[key]
        xmin, xmax, ymin, ymax = z["bounds_0idx_10x10"]
        out[key] = {
            "enabled": z["enabled"],
            "xmin": _rescale_bound(xmin, 9, M),
            "xmax": _rescale_bound(xmax, 9, M),
            "ymin": _rescale_bound(ymin, 9, N),
            "ymax": _rescale_bound(ymax, 9, N),
            "activation_prob": z["activation_prob"],
            "crash_prob_slow": z["crash_prob_slow"],
            "time_extra": z["time_extra"],
        }
    return out


BASELINE_5x6: JammingScenario = {
    "base_crash_prob": 0.001,
    "normal_move_time": 1,
    "slow_move_time": 2,
    "camera": {
        "enabled": True,
        "xmin": 2, "xmax": 3, "ymin": 3, "ymax": 4,
        "activation_prob": 0.80,
        "crash_prob_slow": 0.05,
        "time_extra": 0,
    },
    "gps": {
        "enabled": True,
        "xmin": 4, "xmax": 5, "ymin": 2, "ymax": 3,
        "activation_prob": 0.70,
        "crash_prob_slow": 0.03,
        "time_extra": 1,
    },
    "comms": {
        "enabled": True,
        "xmin": 3, "xmax": 4, "ymin": 5, "ymax": 6,
        "activation_prob": 0.75,
        "crash_prob_slow": 0.02,
        "time_extra": 1,
    },
}


DEFAULT_MOVE_ACTIONS = ("up", "left", "right")


def _zone_const_block(prefix: str, zone: Zone) -> str:
    enabled = 1 if zone.get("enabled", False) else 0
    return (
        f"const int  {prefix}_ENABLE = {enabled};\n"
        f"const int  {prefix}_XMIN = {zone.get('xmin', 0)};\n"
        f"const int  {prefix}_XMAX = {zone.get('xmax', 0)};\n"
        f"const int  {prefix}_YMIN = {zone.get('ymin', 0)};\n"
        f"const int  {prefix}_YMAX = {zone.get('ymax', 0)};\n"
        f"const double {prefix}_PACT = {zone.get('activation_prob', 0.0)};\n"
        f"const double {prefix}_PCRASH = {zone.get('crash_prob_slow', 0.0)};\n"
        f"const int  {prefix}_TEXTRA = {zone.get('time_extra', 0)};\n"
    )


def emit_jamming_block(scenario: JammingScenario,
                       move_actions=DEFAULT_MOVE_ACTIONS) -> str:
    """
    Returns a PRISM fragment to be appended to a base UAV model. The
    fragment defines:
      - constants for each jam zone
      - `in_*_zone` formulas
      - `p_jam_crash` and `t_jam_extra` formulas
      - `module jamming` with one command per movement action
      - `jcrashed` label and a "time" reward structure

    The caller is responsible for rewriting the base model's `crashed`
    formula/label to OR in `jcrashed` (see `patch_base_for_jamming`).
    """
    cam = scenario.get("camera", {"enabled": False})
    gps = scenario.get("gps", {"enabled": False})
    cms = scenario.get("comms", {"enabled": False})

    base_crash = scenario.get("base_crash_prob", 0.001)
    t_norm = scenario.get("normal_move_time", 1)
    t_slow = scenario.get("slow_move_time", 2)

    parts = []
    parts.append("// ============================================================")
    parts.append("//  JAMMING OVERLAY (synchronized on [up] [down] [left] [right])")
    parts.append("// ============================================================")
    parts.append("")
    parts.append(f"const double JAM_BASE_CRASH = {base_crash};")
    parts.append(f"const int    T_NORM = {t_norm};")
    parts.append(f"const int    T_SLOW = {t_slow};")
    parts.append("")
    parts.append("// Camera jamming zone")
    parts.append(_zone_const_block("CAM", cam))
    parts.append("// GPS jamming zone")
    parts.append(_zone_const_block("GPS", gps))
    parts.append("// Comms jamming zone")
    parts.append(_zone_const_block("CMS", cms))

    parts.append(
        "formula in_cam_zone = (CAM_ENABLE = 1 "
        "& x >= CAM_XMIN & x <= CAM_XMAX & y >= CAM_YMIN & y <= CAM_YMAX);"
    )
    parts.append(
        "formula in_gps_zone = (GPS_ENABLE = 1 "
        "& x >= GPS_XMIN & x <= GPS_XMAX & y >= GPS_YMIN & y <= GPS_YMAX);"
    )
    parts.append(
        "formula in_comms_zone = (CMS_ENABLE = 1 "
        "& x >= CMS_XMIN & x <= CMS_XMAX & y >= CMS_YMIN & y <= CMS_YMAX);"
    )
    parts.append("formula in_any_jam = in_cam_zone | in_gps_zone | in_comms_zone;")
    parts.append("")
    parts.append("// Expected per-move jamming-induced crash probability,")
    parts.append("// folding zone-activation into the crash mass.")
    parts.append(
        "formula p_jam_crash = JAM_BASE_CRASH"
        " + (in_cam_zone   ? CAM_PACT * CAM_PCRASH : 0.0)"
        " + (in_gps_zone   ? GPS_PACT * GPS_PCRASH : 0.0)"
        " + (in_comms_zone ? CMS_PACT * CMS_PCRASH : 0.0);"
    )
    parts.append(
        "formula t_jam_extra ="
        " (in_any_jam      ? (T_SLOW - T_NORM) : 0)"
        " + (in_gps_zone   ? GPS_TEXTRA : 0)"
        " + (in_comms_zone ? CMS_TEXTRA : 0);"
    )
    parts.append("")
    parts.append("module jamming")
    parts.append("    jcrashed : bool init false;")
    parts.append("")
    for act in move_actions:
        parts.append(
            f"    [{act}] !jcrashed -> "
            "[p_jam_crash, p_jam_crash] : (jcrashed'=true)"
            " + [1 - p_jam_crash, 1 - p_jam_crash] : true;"
        )
    parts.append("endmodule")
    parts.append("")
    parts.append('label "jcrashed" = jcrashed;')
    parts.append('label "in_jam"  = in_any_jam;')
    parts.append("")
    parts.append('rewards "time"')
    for act in move_actions:
        parts.append(f"    [{act}] true : T_NORM + t_jam_extra;")
    parts.append("endrewards")
    parts.append("")
    parts.append('rewards "jam_exposure"')
    for act in move_actions:
        parts.append(f"    [{act}] in_any_jam : 1;")
    parts.append("endrewards")
    parts.append("")
    return "\n".join(parts)


def patch_base_for_jamming(base_prism: str) -> str:
    """
    Rewrite the base PRISM text so that `crashed` ORs in `jcrashed` from
    the jamming overlay. Specifically:
      - rename the existing `formula crashed = ...` to `formula obs_crashed`
      - add `formula crashed = obs_crashed | jcrashed;`
      - rewrite the `label "crashed" = crashed;` line to use the combined formula
        (it already does, since `crashed` now includes jcrashed)

    The base model's `!crashed` movement guards then transparently halt
    the drone on jamming-induced crashes too.
    """
    if "formula crashed =" not in base_prism:
        raise ValueError("base PRISM text is missing `formula crashed = ...`")

    # Replace only the first occurrence (there is exactly one).
    patched = base_prism.replace(
        "formula crashed =",
        "formula obs_crashed =",
        1,
    )
    # Insert combined formula right after the obs_crashed line.
    needle = "formula obs_crashed ="
    line_end = patched.find(";", patched.find(needle))
    if line_end == -1:
        raise ValueError("could not find end of obs_crashed formula")
    insertion = "\nformula crashed = obs_crashed | jcrashed;"
    patched = patched[: line_end + 1] + insertion + patched[line_end + 1 :]
    return patched
