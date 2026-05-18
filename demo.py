"""Running the Table - one-file demo (multi-pocket tabular Q-learning).

A literal Q-table because the whole point is for the agent's memory
to be a thing you can look at and understand.  One row per pocket
variant (pocket + bounce count), 180 columns per row (one per fired
angle).  No torch, no gymnasium, no stable-baselines3 -- the
computational object stays on screen the entire time.

The coach aims the cue, the agent runs the shot.  Teach this shot
trains the variant under the trajectory; the Q-row for that variant
fills in live, streamed to the browser over a WebSocket.  Untrained
pockets, when called, fire the agent's FIRST learned reflex (its
"native angle"), which is confidently wrong everywhere else -- the
classic out-of-distribution failure mode, made visible.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# World constants
# ---------------------------------------------------------------------------

TABLE_WIDTH = 100.0
TABLE_HEIGHT = 200.0

# Cue is locked at (50, 30) on purpose: from (50, 50) the optimal 135 deg
# bank into TR passes exactly through LM, which would make the agent's
# OOD answer for LM accidentally succeed.  (50, 30) breaks that
# coincidence so OOD pockets miss cleanly.
FIXED_CUE: Tuple[float, float] = (50.0, 30.0)
POCKET_RADIUS = 5.0

POCKETS: Dict[str, Tuple[float, float]] = {
    "TL": (0.0, TABLE_HEIGHT),
    "TR": (TABLE_WIDTH, TABLE_HEIGHT),
    "LM": (0.0, TABLE_HEIGHT / 2.0),
    "RM": (TABLE_WIDTH, TABLE_HEIGHT / 2.0),
    "BL": (0.0, 0.0),
    "BR": (TABLE_WIDTH, 0.0),
}

POCKET_LABELS: Dict[str, str] = {
    "TL": "Top Left",
    "TR": "Top Right",
    "LM": "Left Side",
    "RM": "Right Side",
    "BL": "Bottom Left",
    "BR": "Bottom Right",
}

DEFAULT_TARGET_POCKET = "TR"
N_ANGLES = 180

# Action -> angle mapping covers the FULL 360 degrees so the agent can
# fire downward (toward BL/BR) as well as upward.  Resolution = 2 deg.
def angle_for_action(action: int) -> float:
    return float(action) * (360.0 / N_ANGLES)

LOOP_STEPS = [
    "belief",
    "guess",
    "move",
    "reality",
    "surprise",
    "sharper belief",
]


# ---------------------------------------------------------------------------
# Physics: one-rail raycast + reflection
# ---------------------------------------------------------------------------


def _ray_to_rail(
    x: float, y: float, dx: float, dy: float, w: float, h: float
) -> Optional[Tuple[str, float, float, float]]:
    candidates: List[Tuple[str, float]] = []
    if dx > 1e-12:
        candidates.append(("right", (w - x) / dx))
    elif dx < -1e-12:
        candidates.append(("left", (0.0 - x) / dx))
    if dy > 1e-12:
        candidates.append(("top", (h - y) / dy))
    elif dy < -1e-12:
        candidates.append(("bottom", (0.0 - y) / dy))

    candidates = [(rail, t) for rail, t in candidates if t > 1e-9]
    if not candidates:
        return None
    rail, t = min(candidates, key=lambda c: c[1])
    return rail, t, x + dx * t, y + dy * t


def _segment_point_distance(
    sx: float, sy: float, ex: float, ey: float, px: float, py: float
) -> float:
    vx, vy = ex - sx, ey - sy
    wx, wy = px - sx, py - sy
    seg_len_sq = vx * vx + vy * vy
    if seg_len_sq < 1e-12:
        return math.hypot(px - sx, py - sy)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len_sq))
    cx, cy = sx + t * vx, sy + t * vy
    return math.hypot(px - cx, py - cy)


MAX_BOUNCES = 3
SCRATCH_PENALTY = 60.0

_MIDDLE_POCKETS = {"LM", "RM"}
_CORNER_POCKETS = {"TL", "TR", "BL", "BR"}

# A pocket is a NOTCH in the cushion - the rail physically ends and
# reappears on the far side of the pocket. The opening along each
# adjacent rail (the "jaw") has half-width = POCKET_RADIUS in our
# coordinate model, so JAW_HALFWIDTH = POCKET_RADIUS.
#
# Two complementary checks per segment:
#
#   1) Mid-flight disc crossing (with direction gate). A ball whose
#      straight-line trajectory cuts through a pocket disc BEFORE it
#      reaches a rail drops in - as long as it's actually heading
#      INTO the pocket. The direction gate is what stops a ball
#      travelling parallel to a rail at x~99 from being falsely
#      pocketed in RM as it crosses the disc on the way past.
#
#   2) Rail-jaw arrival. A ball that arrives at a "rail-bounce" point
#      which is INSIDE a pocket jaw has nothing to bounce off of -
#      the cushion is notched away - so it drops in regardless of
#      angle. This is what fixes the "1.98 units past BL but bounces
#      anyway" pathology.
JAW_HALFWIDTH = POCKET_RADIUS

POCKET_MOUTH_MIN_ANGLE_DEG = 6.0
_MOUTH_MIN_DIR = math.sin(math.radians(POCKET_MOUTH_MIN_ANGLE_DEG))

# Legacy constants kept for /config compatibility.
MIDDLE_POCKET_MOUTH_MARGIN = 1.5
CORNER_POCKET_MOUTH_MARGIN = 1.5


def _rail_jaw_pocket(rail: str, hx: float, hy: float) -> Optional[str]:
    """If a rail-bounce point falls inside a pocket jaw, return that
    pocket's id. The cushion is notched away at the pocket mouth so a
    ball arriving there drops in - direction doesn't matter, there's
    no cushion to redirect it."""
    j = JAW_HALFWIDTH
    W = TABLE_WIDTH
    H = TABLE_HEIGHT
    mid = H / 2.0

    if rail == "left":
        if hy <= j:
            return "BL"
        if hy >= H - j:
            return "TL"
        if abs(hy - mid) <= j:
            return "LM"
    elif rail == "right":
        if hy <= j:
            return "BR"
        if hy >= H - j:
            return "TR"
        if abs(hy - mid) <= j:
            return "RM"
    elif rail == "bottom":
        if hx <= j:
            return "BL"
        if hx >= W - j:
            return "BR"
    elif rail == "top":
        if hx <= j:
            return "TL"
        if hx >= W - j:
            return "TR"
    return None


def _is_middle_pocket(pid: str) -> bool:
    return pid in _MIDDLE_POCKETS


def _middle_pocket_entry_ok(pid: str, dx: float) -> bool:
    """Ball must be heading TOWARD the rail with a non-grazing angle.
    A trajectory travelling along the rail (dx near zero for RM/LM) is a
    cushion graze, not a pocket entry, and is rejected."""
    if pid == "RM":
        return dx > _MOUTH_MIN_DIR  # heading +x toward the right rail
    if pid == "LM":
        return dx < -_MOUTH_MIN_DIR  # heading -x toward the left rail
    return True


def _corner_pocket_entry_ok(pid: str, dx: float, dy: float) -> bool:
    """Ball must be heading INTO the corner with a non-grazing angle off
    BOTH adjacent rails. Replaces the old position-based check which
    rejected legitimate corner shots whose disc-entry point grazed a
    rail."""
    if pid == "TL":
        return dx < -_MOUTH_MIN_DIR and dy > _MOUTH_MIN_DIR
    if pid == "TR":
        return dx > _MOUTH_MIN_DIR and dy > _MOUTH_MIN_DIR
    if pid == "BL":
        return dx < -_MOUTH_MIN_DIR and dy < -_MOUTH_MIN_DIR
    if pid == "BR":
        return dx > _MOUTH_MIN_DIR and dy < -_MOUTH_MIN_DIR
    return True


def _pocket_entry_ok(pid: str, dx: float, dy: float) -> bool:
    if pid in _MIDDLE_POCKETS:
        return _middle_pocket_entry_ok(pid, dx)
    if pid in _CORNER_POCKETS:
        return _corner_pocket_entry_ok(pid, dx, dy)
    return True


def _first_pocket_hit_on_segment(
    sx: float,
    sy: float,
    ex: float,
    ey: float,
    pockets: Dict[str, Tuple[float, float]],
) -> Optional[Tuple[str, float, float, float, float]]:
    """Return (pocket_id, t, hit_x, hit_y, distance) for the FIRST pocket the
    segment passes within POCKET_RADIUS of, or None.

    t is the parametric position along the segment (0..1) where the ball
    first enters the pocket disc - that's where the ball actually drops in.
    """
    vx, vy = ex - sx, ey - sy
    seg_len_sq = vx * vx + vy * vy
    if seg_len_sq < 1e-12:
        return None

    # Direction of travel, normalized. The mouth gate is direction-based:
    # a ball with dx, dy must be moving INTO the pocket (with a non-grazing
    # angle off the adjacent rails) to drop.
    seg_len = math.sqrt(seg_len_sq)
    dirx = vx / seg_len
    diry = vy / seg_len

    best = None
    best_t = float("inf")
    r2 = POCKET_RADIUS * POCKET_RADIUS

    for pid, (px, py) in pockets.items():
        # Reject pockets the ball is travelling AWAY from before doing any
        # disc math. This is what stops a rail-bounce that occurs INSIDE a
        # corner pocket disc from being falsely treated as a drop on the
        # outgoing segment.
        if not _pocket_entry_ok(pid, dirx, diry):
            continue
        wx, wy = px - sx, py - sy
        # Closest point on (infinite) line through (sx,sy)->(ex,ey)
        t_closest = (wx * vx + wy * vy) / seg_len_sq
        cx = sx + t_closest * vx
        cy = sy + t_closest * vy
        d2 = (px - cx) ** 2 + (py - cy) ** 2
        if d2 > r2:
            continue
        # Solve for the t at which the segment first crosses the pocket disc.
        # |segment(t) - p|^2 = r^2  ->  quadratic in t.
        a = seg_len_sq
        b = -2.0 * (wx * vx + wy * vy)
        c = wx * wx + wy * wy - r2
        disc = b * b - 4.0 * a * c
        if disc < 0:
            continue
        sqd = math.sqrt(disc)
        t_enter = (-b - sqd) / (2.0 * a)
        # If the segment starts already inside the pocket disc, t_enter < 0;
        # the ball is dropping in right now. Clamp to 0.
        if t_enter < 0:
            t_enter = 0.0
        if t_enter > 1.0:
            continue
        hx = sx + t_enter * vx
        hy = sy + t_enter * vy
        if t_enter < best_t:
            best_t = t_enter
            best = (pid, t_enter, hx, hy, math.sqrt(max(0.0, d2)))

    return best


def evaluate_angle(
    cue_x: float,
    cue_y: float,
    angle_deg: float,
    pocket_xy: Tuple[float, float],
    pockets: Dict[str, Tuple[float, float]] = POCKETS,
    max_bounces: int = MAX_BOUNCES,
) -> dict:
    """Simulate an N-rail shot from (cue_x, cue_y) at angle_deg.

    Walks the ball segment-by-segment. On each segment we check ALL pockets:
    the first one the ball's disc-path enters drops the ball in (target ->
    made; any other -> scratch). If no pocket is hit on a segment, the ball
    bounces off the rail and continues. After `max_bounces` rails without a
    pocket, the ball stops at the final rail point and we return the
    distance-to-target as a graceful-failure reward signal.
    """

    tx, ty = pocket_xy
    angle_rad = math.radians(angle_deg)
    dx, dy = math.cos(angle_rad), math.sin(angle_rad)

    target_id = None
    for pid, (px, py) in pockets.items():
        if abs(px - tx) < 1e-6 and abs(py - ty) < 1e-6:
            target_id = pid
            break

    x, y = cue_x, cue_y
    path: List[List[float]] = [[x, y]]
    rails: List[str] = []
    min_dist = math.hypot(tx - cue_x, ty - cue_y)

    for bounce_idx in range(max_bounces + 1):
        hit = _ray_to_rail(x, y, dx, dy, TABLE_WIDTH, TABLE_HEIGHT)
        if hit is None:
            # Ball traveling parallel to all rails: just stop here.
            break
        rail, _, hx, hy = hit

        # Whether or not we pocketed, the segment's closest-approach to the
        # target still updates the dense reward signal.
        seg_min = _segment_point_distance(x, y, hx, hy, tx, ty)
        if seg_min < min_dist:
            min_dist = seg_min

        # 1) Mid-flight disc crossing (direction-gated).
        pocket_hit = _first_pocket_hit_on_segment(x, y, hx, hy, pockets)
        drop_pid = None
        if pocket_hit is not None:
            drop_pid = pocket_hit[0]

        # 2) Rail-jaw arrival. If the rail-bounce point is INSIDE a
        #    pocket jaw, there's no cushion there - the ball drops in.
        #    Direction doesn't matter because there's nothing to redirect it.
        if drop_pid is None:
            drop_pid = _rail_jaw_pocket(rail, hx, hy)

        if drop_pid is not None:
            pcx, pcy = pockets[drop_pid]
            path.append([pcx, pcy])
            scratched = drop_pid != target_id
            made = not scratched
            if made:
                reward = 100.0
            else:
                reward = -SCRATCH_PENALTY
            return {
                "made": made,
                "scratched": scratched,
                "scratched_into": None if made else drop_pid,
                "reward": float(reward),
                "min_distance": float(min_dist),
                "first_rail": rails[0] if rails else "none",
                "rails": rails,
                "bounces": len(rails),
                "path": path,
                "final": [pcx, pcy],
            }

        # No pocket on this segment: bounce off the rail and keep going.
        path.append([hx, hy])
        rails.append(rail)
        if bounce_idx == max_bounces:
            break
        if rail in ("left", "right"):
            dx = -dx
        else:
            dy = -dy
        x, y = hx, hy

    fx, fy = path[-1]
    return {
        "made": False,
        "scratched": False,
        "scratched_into": None,
        "reward": -float(min_dist),
        "min_distance": float(min_dist),
        "first_rail": rails[0] if rails else "none",
        "rails": rails,
        "bounces": len(rails),
        "path": path,
        "final": [fx, fy],
    }


# ---------------------------------------------------------------------------
# Multi-pocket tabular Q-learning.
#
# State is constant per row (cue is fixed), so Q for one pocket is just a
# 1-D array of length N_ANGLES.  We hold one such array per pocket, plus
# a list of which pockets have actually been trained (in the order they
# were trained -- the first-trained pocket's argmax is the agent's
# "native angle" that it fires at every unknown pocket).
# ---------------------------------------------------------------------------


def variant_id(pocket_id: str, target_bounces: int = 0) -> str:
    """Q-table key. target_bounces==0 means 'any bounce count' (the default
    variant); 1/2/3 mean 'must land via exactly N rail bounces'."""
    if not target_bounces:
        return pocket_id
    return f"{pocket_id}:b{int(target_bounces)}"


def split_variant_id(vid: str) -> Tuple[str, int]:
    if ":b" in vid:
        p, b = vid.split(":b", 1)
        try:
            return p, int(b)
        except ValueError:
            return vid, 0
    return vid, 0


# Order in which we try a pocket's learned variants when asked to fire at it.
# We prefer 1-bank (cleanest), then 2-bank, then 3-bank, then "any". This
# means the agent's reflex for "make TR" is the shortest learned shot.
VARIANT_PREFERENCE: Tuple[int, ...] = (1, 2, 3, 0)


@dataclass
class MultiPocketTrainer:
    n_angles: int = N_ANGLES
    q: Dict[str, np.ndarray] = field(default_factory=dict)
    visits: Dict[str, np.ndarray] = field(default_factory=dict)
    trained_pockets: List[str] = field(default_factory=list)  # variant ids in order trained
    # Exact analog angle (in degrees) to FIRE on inference for a given
    # variant, when one has been pinned down by a successful strict
    # training pass. The Q-table still uses bucket-grained learning,
    # but tricky multi-bank windows can be narrower than the 2deg
    # bucket grid - so on inference we bypass the bucket-center
    # rounding and play back the user's exact coached angle (or, if
    # that exact angle didn't survive python physics, the nearest
    # slid bucket center that did). Keyed by variant id.
    exact_angles: Dict[str, float] = field(default_factory=dict)

    def ensure(self, pocket_id: str) -> None:
        if pocket_id not in self.q:
            self.q[pocket_id] = np.zeros(self.n_angles, dtype=np.float64)
            self.visits[pocket_id] = np.zeros(self.n_angles, dtype=np.int64)

    def pick(
        self, pocket_id: str, rng: np.random.Generator, eps: float
    ) -> int:
        self.ensure(pocket_id)
        if rng.random() < eps:
            return int(rng.integers(0, self.n_angles))
        return int(np.argmax(self.q[pocket_id]))

    def update(
        self, pocket_id: str, action: int, reward: float, lr: float
    ) -> None:
        self.ensure(pocket_id)
        self.q[pocket_id][action] += lr * (reward - self.q[pocket_id][action])
        self.visits[pocket_id][action] += 1
        if pocket_id not in self.trained_pockets:
            self.trained_pockets.append(pocket_id)

    def best_action(self, pocket_id: str) -> Optional[int]:
        if pocket_id in self.q and self.visits[pocket_id].sum() > 0:
            # Only consider buckets we actually trained on. An unvisited
            # bucket sits at Q=0, which is "better" than a visited bucket
            # that converged to a small negative (close-miss) reward -
            # without this mask, argmax would pick a random untouched
            # angle instead of the best one the coach actually demoed.
            visited_mask = self.visits[pocket_id] > 0
            masked = np.where(visited_mask, self.q[pocket_id], -np.inf)
            return int(np.argmax(masked))
        return None

    def best_angle_deg(self, pocket_id: str) -> Optional[float]:
        # Prefer a pinned exact analog angle if one was stored by the
        # strict trainer. This is the angle that ACTUALLY pockets the
        # variant in our physics, which can be off-grid for tight
        # multi-bank shots. Falls back to the bucket-center argmax
        # for variants that pre-date the exact-angle mechanism.
        exact = self.exact_angles.get(pocket_id)
        if exact is not None:
            return float(exact)
        a = self.best_action(pocket_id)
        return None if a is None else angle_for_action(a)

    def native_reflex_angle(self) -> Optional[float]:
        """The agent's 'first language' angle - argmax of the first-trained pocket."""
        if not self.trained_pockets:
            return None
        return self.best_angle_deg(self.trained_pockets[0])

    def is_trained(self, pocket_id: str) -> bool:
        return (
            pocket_id in self.q
            and self.visits[pocket_id].sum() > 0
        )

    def snapshot(self) -> dict:
        return {
            "n_angles": self.n_angles,
            "trained_pockets": list(self.trained_pockets),
            "q": {p: self.q[p].tolist() for p in self.q},
            "visits": {p: self.visits[p].tolist() for p in self.visits},
            "exact_angles": dict(self.exact_angles),
        }

    @classmethod
    def from_snapshot(cls, payload: dict) -> "MultiPocketTrainer":
        t = cls(n_angles=int(payload.get("n_angles", N_ANGLES)))
        t.trained_pockets = list(payload.get("trained_pockets", []))
        for p, q in (payload.get("q") or {}).items():
            arr = np.asarray(q, dtype=np.float64)
            if arr.size == t.n_angles:
                t.q[p] = arr
        for p, v in (payload.get("visits") or {}).items():
            arr = np.asarray(v, dtype=np.int64)
            if arr.size == t.n_angles:
                t.visits[p] = arr
        for p, ang in (payload.get("exact_angles") or {}).items():
            try:
                t.exact_angles[p] = float(ang)
            except (TypeError, ValueError):
                pass
        # repair any q/visit mismatches
        for p in list(t.q.keys()):
            if p not in t.visits:
                t.visits[p] = np.zeros(t.n_angles, dtype=np.int64)
        # drop trained_pockets entries that don't have a Q-row
        t.trained_pockets = [
            p for p in t.trained_pockets if p in t.q and t.visits[p].sum() > 0
        ]
        # Drop exact_angles for variants that are no longer trained.
        t.exact_angles = {
            p: a for p, a in t.exact_angles.items() if p in t.trained_pockets
        }
        return t


# ---------------------------------------------------------------------------
# Model state + decision logic
# ---------------------------------------------------------------------------


@dataclass
class SessionState:
    trainer: MultiPocketTrainer
    training_lock: threading.Lock
    training_status: dict
    training_buffer: List[dict]
    last_accessed: float

SESSIONS: Dict[str, SessionState] = {}

def get_session(session_id: str) -> SessionState:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = SessionState(
            trainer=MultiPocketTrainer(n_angles=N_ANGLES),
            training_lock=threading.Lock(),
            training_status={
                "status": "idle",
                "target_pocket": None,
                "episodes_done": 0,
                "downsampled_frames": 0,
                "total_timesteps": 0,
                "first_make_episode": None,
                "converged_at_episode": None,
                "final_angle": None,
                "started_at": None,
                "finished_at": None,
                "error": None,
                "strict_match_learned": None,
            },
            training_buffer=[],
            last_accessed=time.time()
        )
    else:
        SESSIONS[session_id].last_accessed = time.time()
    return SESSIONS[session_id]


def _agent_angle_for(
    session_id: str, pocket_id: str, target_bounces: Optional[int] = None
) -> Tuple[Optional[float], str, int]:
    """Return (angle, source, bounces_used) where source is one of:
       'agent_trained' | 'agent_native_reflex' | 'no_model'.

    If ``target_bounces`` is provided we look up that EXACT variant
    (this is the post-cinematic auto-fire path - the coach just
    taught a specific shot and we want to fly that exact shot back,
    not a different bank count that happens to be ahead of it in
    VARIANT_PREFERENCE).

    Otherwise we try learned variants in VARIANT_PREFERENCE order:
    1-bank first (cleanest), then 2, 3, then 'any'. If none are
    trained we fall back to the agent's native reflex (argmax of the
    first row ever trained, applied confidently to terrain it has
    never seen).
    """
    trainer = get_session(session_id).trainer
    if not trainer.trained_pockets:
        return None, "no_model", 0

    if target_bounces is not None:
        vid = variant_id(pocket_id, int(target_bounces))
        if trainer.is_trained(vid):
            return trainer.best_angle_deg(vid), "agent_trained", int(target_bounces)
        # Fall through to the preference list so a request for an
        # untrained variant still gets the best available answer
        # instead of an oracle fallback.

    for b in VARIANT_PREFERENCE:
        vid = variant_id(pocket_id, b)
        if trainer.is_trained(vid):
            return trainer.best_angle_deg(vid), "agent_trained", b

    native = trainer.native_reflex_angle()
    if native is None:
        return None, "no_model", 0
    return native, "agent_native_reflex", 0


def _oracle_angle(pocket_xy: Tuple[float, float]) -> float:
    best_angle = 0.0
    best_dist = float("inf")
    best_reward = -float("inf")
    for angle in range(360):
        trial = evaluate_angle(FIXED_CUE[0], FIXED_CUE[1], float(angle), pocket_xy)
        dist = float(trial["min_distance"])
        reward = float(trial["reward"])
        if dist < best_dist or (math.isclose(dist, best_dist) and reward > best_reward):
            best_dist = dist
            best_reward = reward
            best_angle = float(angle)
    return best_angle


def _decide_angle(
    session_id: str, pocket_id: str, mode: str, target_bounces: Optional[int] = None
) -> Tuple[float, str]:
    pocket_xy = POCKETS[pocket_id]
    if mode == "agent":
        angle, source, _bounces = _agent_angle_for(session_id, pocket_id, target_bounces)
        if angle is not None:
            return float(angle), source
        return _oracle_angle(pocket_xy), "oracle_fallback"
    return _oracle_angle(pocket_xy), "oracle"


# ---------------------------------------------------------------------------
# Narrator (predict-mode 6-step loop)
# ---------------------------------------------------------------------------


def _narrator_lines(
    pocket_id: str,
    angle_deg: float,
    sim: dict,
    decision: str,
    is_trained_pocket: bool,
    native_pocket: Optional[str],
) -> Dict[str, str]:
    """Six-beat narrator copy mapped to the live shot outcome."""

    pocket_name = POCKET_LABELS[pocket_id]
    miss = sim["min_distance"]
    made = sim["made"]
    native_name = POCKET_LABELS[native_pocket] if native_pocket else None

    if decision == "agent_trained":
        belief = (
            f"I have rehearsed this exact shot. "
            f"My Q-row for {pocket_name} has a single tall spoke."
        )
        guess = (
            f"My reflex says {angle_deg:.0f} degrees. The action stops "
            "feeling computed and starts feeling retrieved."
        )
    elif decision == "agent_native_reflex":
        belief = (
            f"I have never been taught {pocket_name}. My only learned reflex "
            f"is for {native_name}. Every prediction engine is bounded by its "
            "training data."
        )
        guess = (
            f"I will fire that same {angle_deg:.0f} degrees. I am completely "
            "confident -- because confidence does not require correctness."
        )
    elif decision == "oracle":
        belief = "Pure geometry. No memory, just math."
        guess = f"The optimal one-rail angle is {angle_deg:.0f} degrees."
    else:  # oracle_fallback
        belief = "No trained agent yet. Falling back to geometric search."
        guess = (
            f"The optimal one-rail angle is {angle_deg:.0f} degrees. "
            "Train the agent on this pocket to compare against its reflex."
        )

    move = "The story your brain is telling itself about what is about to happen next."
    bounces = int(sim.get("bounces", 0))
    bounce_label = (
        "no rail"
        if bounces == 0
        else "one rail"
        if bounces == 1
        else f"{bounces} rails"
    )
    scratched = bool(sim.get("scratched"))
    scratched_into = sim.get("scratched_into")
    scratched_into_name = (
        POCKET_LABELS.get(scratched_into) if scratched_into else None
    )

    if made:
        if bounces >= 2:
            reality = (
                f"Pocketed {pocket_name} off {bounce_label}. A multi-bank shot "
                "the agent built itself through exploration."
            )
        elif bounces == 1:
            reality = f"Pocketed {pocket_name} off one rail. Reality matched the prediction."
        else:
            reality = f"Pocketed {pocket_name} clean. Reality matched the prediction."
        surprise = (
            "Reward +100. Dopamine isn't the chemistry of pleasure -- "
            "it's the chemistry of surprise that exceeds expectation."
        )
        sharper = (
            f"Q[{pocket_name}] just deepened. The reflex is now a little "
            "more permanent. Same shot tomorrow, less doubt."
        )
    else:
        if scratched and scratched_into_name:
            reality = (
                f"Scratched into {scratched_into_name}. Right confidence, "
                "wrong pocket."
            )
        else:
            reality = f"Missed {pocket_name} by {miss:.1f} units after {bounce_label}."
        if decision == "agent_native_reflex":
            surprise = (
                "When the input falls outside the training distribution, the "
                "engine does not throw an error. It confidently generates a "
                "plausible answer from its existing priors."
            )
            sharper = (
                "Hold your model firmly enough to act on it. Loosely enough "
                "to update it. Running the simulations forward now..."
            )
        else:
            surprise = (
                f"Reward {sim['reward']:+.1f}. Reward prediction error -- "
                "the gap between expected and actual."
            )
            sharper = (
                "Fold this miss back in. The Q-table only updates from the "
                "action that was actually taken."
            )

    return {
        "belief": belief,
        "guess": guess,
        "move": move,
        "reality": reality,
        "surprise": surprise,
        "sharper belief": sharper,
    }


# ---------------------------------------------------------------------------
# Live training: background thread + shared buffer + WebSocket fan-out
# ---------------------------------------------------------------------------


def _training_thread(
    session_id: str,
    target_pocket: str,
    target_bounces: int,
    timesteps: int,
    seed: int,
    learning_rate: float,
    eps_start: float,
    eps_end: float,
    eps_decay_frac: float,
    pace_ms: float,
    reset_q: bool,
    anchor_angle_deg: Optional[float] = None,
) -> None:
    """Background entry point.  Owns TRAINING_LOCK while running.

    Trains a single Q-row, keyed by (pocket, target_bounces). When
    target_bounces > 0, only shots that pocket the ball using exactly
    that many rail bounces are considered successful; making the
    ball into the right pocket with the wrong bounce count gets a
    penalty so the agent doesn't reinforce the wrong variant.

    When ``anchor_angle_deg`` is provided, training is "demonstration
    anchored": the coach has shown a specific angle, and the agent's
    job is to imprint THAT shot, not just find any angle that pockets
    the ball.  Two things change:

      1. The Q-row is pre-seeded with a small positive value at the
         anchor bucket so the agent's first guess matches the demo.
      2. Exploration samples angles from a Gaussian centered on the
         anchor (stdev decays with eps - wide while exploring, narrow
         while exploiting), instead of uniform over all 180 angles.

    In strict coach mode (anchor provided), the trained action is the
    EXACT coached bucket. No widening, no relaxed fallback, no alternate
    route substitution. If that exact coached shot is physically
    impossible from the fixed cue position, training reports that it
    could not imprint the requested trajectory.
    """

    session = get_session(session_id)
    trainer = session.trainer
    TRAINING_LOCK = session.training_lock
    TRAINING_STATUS = session.training_status
    TRAINING_BUFFER = session.training_buffer

    vid = variant_id(target_pocket, target_bounces)

    # Snap the demonstrated angle to a discrete action bucket.
    anchor_action: Optional[int] = None
    if anchor_angle_deg is not None:
        bucket_size = 360.0 / N_ANGLES
        anchor_action = int(round((anchor_angle_deg % 360.0) / bucket_size)) % N_ANGLES

    # Tricky multi-bank shots: the user's analog aim might pocket the
    # ball at e.g. 76.4 deg via 3 banks, but the nearest 2 deg bucket
    # center (76 deg) reflects slightly differently and arrives at the
    # wrong pocket or the wrong bank count. We resolve this in two
    # independent steps:
    #
    #   1) ANCHOR BUCKET (used for Q-spike + strict_learned check).
    #      Must pocket the variant at its bucket center, otherwise
    #      the strict trainer can never satisfy `action == anchor`.
    #      If the user's bucket center misses, slide ±5 buckets to
    #      the nearest viable bucket center.
    #
    #   2) EXACT INFERENCE ANGLE (what auto-fire actually shoots).
    #      Prefer the user's EXACT analog angle if python physics
    #      agrees it pockets the variant. Otherwise fall back to the
    #      (possibly slid) anchor bucket center. This lets the agent
    #      fire the precise coached aim even when the bucket grid
    #      itself can't represent the tight valid window.
    #
    # Decoupling these means tricky 3-bank shots can have an anchor
    # bucket that's 1-2 buckets away from the user's bucket (so the
    # Q-row training loop actually finds wins) while still firing the
    # user's exact analog angle on inference (so the auto-fire shot
    # matches the cinematic locked-in path 1:1).
    exact_inference_angle: Optional[float] = None
    if anchor_action is not None and anchor_angle_deg is not None:
        target_xy_for_snap = POCKETS[target_pocket]

        def _angle_pockets_variant(ang: float) -> bool:
            sim = evaluate_angle(
                FIXED_CUE[0], FIXED_CUE[1], float(ang), target_xy_for_snap
            )
            if not bool(sim.get("made")):
                return False
            if int(target_bounces) == 0:
                return True
            return int(sim.get("bounces", 0)) == int(target_bounces)

        user_exact_pockets = _angle_pockets_variant(float(anchor_angle_deg))
        if user_exact_pockets:
            exact_inference_angle = float(anchor_angle_deg)

        # Anchor-bucket viability check. Slide if the user's bucket
        # center can't make the shot at the requested bank count.
        if not _angle_pockets_variant(float(angle_for_action(anchor_action))):
            slid: Optional[int] = None
            for off in range(1, 6):
                for sign in (-1, 1):
                    cand = (anchor_action + sign * off) % N_ANGLES
                    if _angle_pockets_variant(float(angle_for_action(cand))):
                        slid = cand
                        break
                if slid is not None:
                    break
            if slid is not None:
                anchor_action = slid
                # If the user's exact aim ALSO failed (rare, physics
                # divergence), use the slid bucket center for both
                # learning AND inference so the visuals match.
                if exact_inference_angle is None:
                    anchor_angle_deg = float(angle_for_action(slid))
                    exact_inference_angle = float(anchor_angle_deg)
            # else: no viable bucket nearby. anchor_action keeps the
            # user's original bucket and strict_learned will fail
            # honestly. The Q-row gets wiped at the end of training.

    try:
        TRAINING_BUFFER.clear()
        TRAINING_STATUS.update(
            {
                "status": "training",
                "target_pocket": target_pocket,
                "target_bounces": int(target_bounces),
                "variant_id": vid,
                "episodes_done": 0,
                "downsampled_frames": 0,
                "total_timesteps": int(timesteps),
                "first_make_episode": None,
                "converged_at_episode": None,
                "final_angle": None,
                "started_at": time.time(),
                "finished_at": None,
                "error": None,
                "anchor_angle_deg": (
                    float(anchor_angle_deg) if anchor_angle_deg is not None else None
                ),
                "anchor_action": anchor_action,
                "strict_match_learned": None,
            }
        )

        target_xy = POCKETS[target_pocket]
        if reset_q:
            trainer.q[vid] = np.zeros(N_ANGLES, dtype=np.float64)
            trainer.visits[vid] = np.zeros(N_ANGLES, dtype=np.int64)
            trainer.exact_angles.pop(vid, None)
            if vid in trainer.trained_pockets:
                trainer.trained_pockets.remove(vid)
        trainer.ensure(vid)

        # Pre-seed the demonstration so the agent's "first guess" and
        # the fallback argmax (when nothing in the window is viable)
        # are both the coach's exact angle.
        if anchor_action is not None:
            trainer.q[vid][anchor_action] = 5.0

        # STRICT coach mode with VISIBLE exploration: we let the agent
        # try a ±20deg window around the coached angle so the cinematic
        # actually shows try/fail/adjust playing out. The angle-distance
        # reward penalty (below) guarantees the coached bucket wins the
        # argmax when it's viable, and we only count strict_learned if
        # the coached bucket itself made the shot - so the agent never
        # "learns" a sibling angle instead of yours.
        if anchor_action is not None:
            anchor_tiers = [
                (10, 1.0, False),  # +/-10 buckets = +/-20deg around demo
            ]
        else:
            anchor_tiers = [(None, 1.0, False)]

        def _window_for(tier_idx: int) -> Optional[np.ndarray]:
            half = anchor_tiers[tier_idx][0]
            if half is None or anchor_action is None:
                return None
            return np.array(
                [(anchor_action + off) % N_ANGLES for off in range(-half, half + 1)],
                dtype=np.int64,
            )

        def _tier_relaxed(tier_idx: int) -> bool:
            return bool(anchor_tiers[tier_idx][2])

        rng = np.random.default_rng(seed)
        wins: List[int] = []
        first_make: Optional[int] = None
        converged: Optional[int] = None
        decay_episodes = max(1, int(timesteps * eps_decay_frac))
        downsample = max(1, timesteps // 400)
        pace_seconds = max(0.0, pace_ms / 1000.0)

        tier_idx = 0
        allowed_actions = _window_for(tier_idx)
        tier_started_ep = 1
        tier_budget = max(50, int(timesteps * anchor_tiers[tier_idx][1]))
        tier_made = False  # has THIS tier landed at least one strict make yet?
        relaxed_mode = _tier_relaxed(tier_idx)

        def _tier_radius_deg(idx: int) -> Optional[float]:
            half = anchor_tiers[idx][0]
            if half is None:
                return None
            return float(half) * (360.0 / N_ANGLES)

        TRAINING_STATUS["search_tier"] = tier_idx
        TRAINING_STATUS["search_radius_deg"] = _tier_radius_deg(tier_idx)
        TRAINING_STATUS["relaxed_bounces"] = relaxed_mode
        TRAINING_STATUS["actual_bounces_learned"] = None

        for ep in range(1, timesteps + 1):
            # Tier widening for non-strict modes. In strict anchored mode
            # there is only one tier (exact coached bucket), so this block
            # never executes.
            if (
                anchor_action is not None
                and not tier_made
                and tier_idx + 1 < len(anchor_tiers)
                and (ep - tier_started_ep) >= tier_budget
            ):
                tier_idx += 1
                allowed_actions = _window_for(tier_idx)
                tier_started_ep = ep
                tier_budget = max(50, int(timesteps * anchor_tiers[tier_idx][1]))
                relaxed_mode = _tier_relaxed(tier_idx)
                TRAINING_STATUS["search_tier"] = tier_idx
                TRAINING_STATUS["search_radius_deg"] = _tier_radius_deg(tier_idx)
                TRAINING_STATUS["relaxed_bounces"] = relaxed_mode
                if relaxed_mode:
                    # Kept for compatibility with non-strict flows.
                    trainer.q[vid] = np.zeros(N_ANGLES, dtype=np.float64)
                    trainer.visits[vid] = np.zeros(N_ANGLES, dtype=np.int64)
                    if anchor_action is not None:
                        trainer.q[vid][anchor_action] = 1.0

            eps = max(
                eps_end,
                eps_start - (eps_start - eps_end) * (ep / decay_episodes),
            )
            trainer.ensure(vid)
            if allowed_actions is not None:
                # Restricted to the current tier's window. Exploration
                # samples uniformly inside the window; exploitation
                # takes argmax *within the window* so a zero bucket
                # outside the window can never beat negative learning
                # inside it.
                if rng.random() < eps:
                    action = int(allowed_actions[rng.integers(0, allowed_actions.size)])
                else:
                    sub_q = trainer.q[vid][allowed_actions]
                    sub_visits = trainer.visits[vid][allowed_actions]
                    if np.any(sub_visits > 0):
                        masked_sub_q = np.where(sub_visits > 0, sub_q, -np.inf)
                        action = int(allowed_actions[int(np.argmax(masked_sub_q))])
                    else:
                        action = int(allowed_actions[int(np.argmax(sub_q))])
            else:
                if rng.random() < eps:
                    action = int(rng.integers(0, N_ANGLES))
                else:
                    visits = trainer.visits[vid]
                    if np.any(visits > 0):
                        masked_q = np.where(visits > 0, trainer.q[vid], -np.inf)
                        action = int(np.argmax(masked_q))
                    else:
                        action = int(np.argmax(trainer.q[vid]))
            angle_deg = angle_for_action(action)
            sim = evaluate_angle(
                FIXED_CUE[0], FIXED_CUE[1], angle_deg, target_xy
            )
            base_reward = float(sim["reward"])

            # Variant-aware reward shaping. Success means:
            # made target pocket AND used requested bounce count.
            actual_bounces = int(sim.get("bounces", 0))
            shot_made = bool(sim["made"])
            if target_bounces == 0:
                variant_match = True
            else:
                variant_match = actual_bounces == target_bounces
            variant_made = shot_made and variant_match

            if variant_made:
                # Reward shape MUST honor the coach's exact angle when
                # multiple angles in the search window pocket correctly.
                # Without this term every "good" angle in the window
                # accumulates the same Q (=100) and argmax tie-breaks on
                # whichever was visited first - so the agent "learns the
                # right pocket" but NOT the angle the coach demoed.
                #
                # Penalty: 0.5 reward per degree of bucket-distance from
                # the demo, capped so a make is always strongly favoured
                # over a miss. With the cap at 60 the minimum make-reward
                # (40) still dominates any miss reward (which is
                # -distance, typically -10 to -100). At 0 deg distance
                # the full +100 applies; the demo's bucket is always the
                # global Q-row argmax when it's viable.
                if anchor_action is not None:
                    bucket_dist = abs(action - anchor_action)
                    bucket_dist = min(
                        bucket_dist, N_ANGLES - bucket_dist
                    )  # circular
                    deg_dist = bucket_dist * (360.0 / N_ANGLES)
                    angle_penalty = min(60.0, deg_dist * 0.5)
                    reward = 100.0 - angle_penalty
                else:
                    reward = 100.0
                # Strict mode keeps this equal to the requested bounces.
                TRAINING_STATUS["actual_bounces_learned"] = actual_bounces
            elif shot_made and not variant_match:
                reward = -25.0
            else:
                reward = base_reward

            trainer.update(vid, action, reward, learning_rate)

            wins.append(1 if variant_made else 0)
            if variant_made:
                # Mark strict success for this training run.
                tier_made = True
            # `first_make` is reserved for the coached bucket specifically.
            # That way a sibling angle in the exploration window cannot
            # masquerade as "the agent learned your shot" - strict success
            # requires the demo bucket itself to pocket the ball.
            anchor_made_now = variant_made and (
                anchor_action is None or action == anchor_action
            )
            if anchor_made_now and first_make is None:
                first_make = ep
                TRAINING_STATUS["first_make_episode"] = first_make
            if converged is None and len(wins) >= 50:
                if sum(wins[-50:]) / 50.0 >= 0.95:
                    converged = ep
                    TRAINING_STATUS["converged_at_episode"] = converged

            TRAINING_STATUS["episodes_done"] = ep

            if ep % downsample == 0 or ep == timesteps:
                window = min(50, len(wins))
                win_rate = sum(wins[-window:]) / window
                best_angle = trainer.best_angle_deg(vid)
                frame = {
                    "episode_idx": ep,
                    "target_pocket": target_pocket,
                    "target_bounces": int(target_bounces),
                    "variant_id": vid,
                    "angle_deg": float(angle_deg),
                    "made": bool(variant_made),
                    "raw_made": bool(shot_made),
                    "bounces": actual_bounces,
                    "reward": reward,
                    "min_distance": float(sim["min_distance"]),
                    "exploration_rate": float(eps),
                    "win_rate": float(win_rate),
                    "path": [list(p) for p in sim["path"]],
                    "best_angle_deg": float(best_angle) if best_angle is not None else None,
                    "q": trainer.q[vid].tolist(),
                    "search_tier": tier_idx,
                    "search_radius_deg": _tier_radius_deg(tier_idx),
                    "relaxed_bounces": bool(relaxed_mode),
                    "actual_bounces_learned": TRAINING_STATUS.get(
                        "actual_bounces_learned"
                    ),
                    "anchor_angle_deg": (
                        float(anchor_angle_deg) if anchor_angle_deg is not None else None
                    ),
                }
                TRAINING_BUFFER.append(frame)
                TRAINING_STATUS["downsampled_frames"] = len(TRAINING_BUFFER)

                if pace_seconds > 0:
                    time.sleep(pace_seconds)

        # Strict-learned determination.
        #
        # The training loop runs on the 2-deg bucket grid; for tight
        # multi-bank shots the valid analog window can be narrower
        # than that grid, meaning NO bucket center pockets the shot
        # even though the user's analog aim does. In that case the
        # in-loop `first_make` flag never fires - but we already
        # pre-verified the user's exact analog at training start, so
        # we know the shot is physically real. Trust that pre-check.
        #
        # Strict success requires:
        #   (a) a pinned inference angle (user's exact, or a slid
        #       viable bucket center) - this is the ANGLE we'll fire.
        #   AND
        #   (b) EITHER the bucket grid found at least one in-loop win
        #       at the anchor bucket (normal case), OR we know the
        #       inference angle is the user's own analog aim and that
        #       pre-validation step passed (tight-window case).
        #
        # When (b) is satisfied only by pre-validation we still need
        # to populate the Q-row so `is_trained()` returns True - we
        # do that by forcing a single visit and a Q-spike at the
        # anchor bucket. The visualization shows the spike; inference
        # bypasses the bucket and fires the precise analog angle.
        bucket_learned = (anchor_action is None) or (first_make is not None)
        pre_validated = exact_inference_angle is not None
        strict_learned = bucket_learned or pre_validated if anchor_action is not None else True
        TRAINING_STATUS["strict_match_learned"] = bool(strict_learned)

        if not strict_learned and anchor_action is not None:
            # Strict coached shot could not be imprinted. Remove the row so
            # we never pretend an alternate trajectory is "learned".
            trainer.q[vid] = np.zeros(N_ANGLES, dtype=np.float64)
            trainer.visits[vid] = np.zeros(N_ANGLES, dtype=np.int64)
            if vid in trainer.trained_pockets:
                trainer.trained_pockets.remove(vid)
            trainer.exact_angles.pop(vid, None)
            TRAINING_STATUS["final_angle"] = None
        elif anchor_action is not None:
            # Force the anchor bucket's Q to dominate so the Q-row
            # visualization shows a clean spike on the trained bucket.
            trainer.q[vid][anchor_action] = max(
                float(trainer.q[vid][anchor_action]), 100.0
            )
            # If the loop never visited (tight-window case), give the
            # anchor one synthetic visit so is_trained() returns True
            # and best_action() picks it for the visualization.
            if trainer.visits[vid].sum() == 0:
                trainer.visits[vid][anchor_action] = 1
            if vid not in trainer.trained_pockets:
                trainer.trained_pockets.append(vid)
            # Inference always uses the exact angle when we have one.
            if exact_inference_angle is not None:
                trainer.exact_angles[vid] = float(exact_inference_angle)
                TRAINING_STATUS["final_angle"] = float(exact_inference_angle)
            else:
                # bucket_learned must be True here (otherwise strict
                # would have been False). Fall back to bucket center.
                trainer.exact_angles.pop(vid, None)
                TRAINING_STATUS["final_angle"] = float(angle_for_action(anchor_action))
        else:
            trainer.exact_angles.pop(vid, None)
            TRAINING_STATUS["final_angle"] = trainer.best_angle_deg(vid)

        TRAINING_STATUS["status"] = "done"
        TRAINING_STATUS["finished_at"] = time.time()

    except Exception as exc:  # pragma: no cover
        TRAINING_STATUS["status"] = "error"
        TRAINING_STATUS["error"] = repr(exc)
        TRAINING_STATUS["finished_at"] = time.time()
    finally:
        if TRAINING_LOCK.locked():
            try:
                TRAINING_LOCK.release()
            except RuntimeError:
                pass


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


app = FastAPI(title="Running the Table")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class PredictRequest(BaseModel):
    session_id: str
    pocket_id: str
    mode: str = "agent"
    # When the coach just taught a specific variant (e.g. TR via
    # 2-bank), the auto-fire wants to play THAT exact variant back -
    # not whatever variant happens to be first in VARIANT_PREFERENCE.
    # None means "let the agent pick its favorite variant for this
    # pocket" (1-bank if trained, else 2-bank, etc).
    target_bounces: Optional[int] = None


class TrainStartRequest(BaseModel):
    session_id: str
    target_pocket: str = DEFAULT_TARGET_POCKET
    target_bounces: int = 0  # 0 = any bounce count, 1/2/3 = exact rail count
    timesteps: int = 2000
    seed: int = 42
    learning_rate: float = 0.3
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.5
    pace_ms: float = 4.0
    reset_q: bool = True
    # The coach's demonstrated angle. When set, training pre-seeds the
    # Q-row at the matching bucket and biases exploration to a Gaussian
    # centered on this angle (stdev decays with eps). Without this, the
    # trainer would converge to whichever angle physics happens to
    # reward first - which "selects the hole" but ignores the specific
    # trajectory the coach actually pointed at. Anchoring makes the
    # cinematic land exactly on the demonstrated shot.
    anchor_angle_deg: Optional[float] = None


def _model_summary(session_id: str) -> dict:
    session = get_session(session_id)
    t = session.trainer
    if t is None:
        return {
            "loaded": False,
            "trained_pockets": [],
            "trained_variants": {},
            "trained_variant_angles": {},
            "native_pocket": None,
            "native_variant": None,
            "trained_angles": {},
            "q_rows": {},
            "visits": {},
            "n_angles": N_ANGLES,
        }

    trained_pockets: List[str] = []
    trained_variants: Dict[str, List[int]] = {}
    trained_variant_angles: Dict[str, Dict[int, float]] = {}
    trained_angles: Dict[str, float] = {}  # pocket -> preferred-variant angle
    # Full Q-rows and visit counts, keyed by variant id ("TR:b1"). The
    # dashboard renders these as the agent's CURRENT BELIEF histogram.
    q_rows: Dict[str, List[float]] = {}
    visits: Dict[str, List[int]] = {}

    for vid in t.trained_pockets:
        pid, b = split_variant_id(vid)
        if pid not in trained_pockets:
            trained_pockets.append(pid)
        trained_variants.setdefault(pid, []).append(b)
        trained_variant_angles.setdefault(pid, {})[b] = t.best_angle_deg(vid)

    # Snapshot every trained Q-row plus its visit counts for the dashboard.
    # Cheap: 180 floats per row, max 18 variants = ~26KB worst case.
    for vid, row in t.q.items():
        q_rows[vid] = [float(v) for v in row.tolist()]
        if vid in t.visits:
            visits[vid] = [int(v) for v in t.visits[vid].tolist()]

    # Per-pocket "best" angle = preferred variant's angle (1-bank > 2 > 3 > any)
    for pid in trained_pockets:
        for b in VARIANT_PREFERENCE:
            vid = variant_id(pid, b)
            if t.is_trained(vid):
                trained_angles[pid] = t.best_angle_deg(vid)
                break

    native_variant_id = t.trained_pockets[0] if t.trained_pockets else None
    native_pocket, _ = (
        split_variant_id(native_variant_id) if native_variant_id else (None, 0)
    )
    return {
        "loaded": bool(t.trained_pockets),
        "trained_pockets": trained_pockets,
        "trained_variants": trained_variants,
        "trained_variant_angles": trained_variant_angles,
        "native_pocket": native_pocket,
        "native_variant": native_variant_id,
        "trained_angles": trained_angles,
        "q_rows": q_rows,
        "visits": visits,
        "n_angles": N_ANGLES,
    }


def _asset_version() -> str:
    """Cache-buster based on the freshest static file mtime so dev
    edits to styles.css / app.js never end up stale behind a 304."""
    try:
        css = Path("static/styles.css").stat().st_mtime
        js = Path("static/app.js").stat().st_mtime
        return str(int(max(css, js)))
    except Exception:
        return str(int(time.time()))


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"title": "Running the Table", "asset_v": _asset_version()},
    )


@app.get("/config")
def config(session_id: str) -> dict:
    session = get_session(session_id)
    return {
        "table": {"width": TABLE_WIDTH, "height": TABLE_HEIGHT},
        "fixed_cue": {"x": FIXED_CUE[0], "y": FIXED_CUE[1]},
        "pocket_radius": POCKET_RADIUS,
        "middle_pocket_mouth_margin": MIDDLE_POCKET_MOUTH_MARGIN,
        "middle_pocket_ids": sorted(_MIDDLE_POCKETS),
        "corner_pocket_mouth_margin": CORNER_POCKET_MOUTH_MARGIN,
        "corner_pocket_ids": sorted(_CORNER_POCKETS),
        "pockets": {
            k: {"x": v[0], "y": v[1], "label": POCKET_LABELS[k]}
            for k, v in POCKETS.items()
        },
        "model": _model_summary(session_id),
        "training": dict(session.training_status),
        "loop_steps": LOOP_STEPS,
        "n_angles": N_ANGLES,
        "angle_per_action_deg": 360.0 / N_ANGLES,
    }


@app.post("/preview")
def preview(req: PredictRequest) -> dict:
    """Cheap version of /predict for hover-trajectory previews.

    Same physics + same decision, but no narration / confidence math /
    anything the hover overlay doesn't need. Safe to fire on every mouse
    move because the trainer isn't touched.
    """
    session_id = req.session_id
    pocket_id = req.pocket_id.upper()
    if pocket_id not in POCKETS:
        raise HTTPException(status_code=400, detail="Unknown pocket_id")
    mode = (req.mode or "agent").lower()
    if mode not in ("agent", "oracle"):
        raise HTTPException(status_code=400, detail="mode must be 'agent' or 'oracle'")

    target_bounces = req.target_bounces
    if target_bounces is not None and int(target_bounces) not in (0, 1, 2, 3):
        raise HTTPException(
            status_code=400, detail="target_bounces must be 0, 1, 2, or 3"
        )
    angle_deg, decision = _decide_angle(session_id, pocket_id, mode, target_bounces)
    sim = evaluate_angle(FIXED_CUE[0], FIXED_CUE[1], angle_deg, POCKETS[pocket_id])
    trainer = get_session(session_id).trainer
    return {
        "pocket_id": pocket_id,
        "decision_source": decision,
        "is_trained_pocket": bool(trainer and trainer.is_trained(pocket_id)),
        "angle_deg": round(float(angle_deg), 2),
        "path": sim["path"],
        "bounces": int(sim.get("bounces", 0)),
        "made": bool(sim["made"]),
        "scratched": bool(sim.get("scratched")),
        "scratched_into": sim.get("scratched_into"),
        "min_distance": float(sim["min_distance"]),
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    session_id = req.session_id
    pocket_id = req.pocket_id.upper()
    if pocket_id not in POCKETS:
        raise HTTPException(status_code=400, detail="Unknown pocket_id")
    mode = (req.mode or "agent").lower()
    if mode not in ("agent", "oracle"):
        raise HTTPException(status_code=400, detail="mode must be 'agent' or 'oracle'")

    target_bounces = req.target_bounces
    if target_bounces is not None and int(target_bounces) not in (0, 1, 2, 3):
        raise HTTPException(
            status_code=400, detail="target_bounces must be 0, 1, 2, or 3"
        )
    angle_deg, decision = _decide_angle(session_id, pocket_id, mode, target_bounces)
    sim = evaluate_angle(FIXED_CUE[0], FIXED_CUE[1], angle_deg, POCKETS[pocket_id])

    confidence = max(0.0, min(1.0, 1.0 - (sim["min_distance"] / 30.0)))
    trainer = get_session(session_id).trainer
    # "Trained pocket" means ANY variant for this pocket is trained.
    is_trained = bool(
        trainer
        and any(
            split_variant_id(v)[0] == pocket_id for v in trainer.trained_pockets
        )
    )
    # Native pocket = base pocket of the first variant ever trained.
    native_pocket = None
    if trainer and trainer.trained_pockets:
        native_pocket = split_variant_id(trainer.trained_pockets[0])[0]
    # The deduped list of trained base pockets (for the frontend badges).
    trained_pockets_base: List[str] = []
    if trainer:
        for v in trainer.trained_pockets:
            p, _ = split_variant_id(v)
            if p not in trained_pockets_base:
                trained_pockets_base.append(p)

    return {
        "pocket_id": pocket_id,
        "pocket_label": POCKET_LABELS[pocket_id],
        "mode_requested": mode,
        "decision_source": decision,
        "is_trained_pocket": is_trained,
        "native_pocket": native_pocket,
        "trained_pockets": trained_pockets_base,
        "cue": {"x": FIXED_CUE[0], "y": FIXED_CUE[1]},
        "angle_deg": round(float(angle_deg), 2),
        "predicted": sim,
        "confidence": round(confidence, 3),
        "narration": _narrator_lines(
            pocket_id, angle_deg, sim, decision, is_trained, native_pocket
        ),
        "loop_steps": LOOP_STEPS,
    }


class MemoryResetRequest(BaseModel):
    session_id: str

@app.post("/memory/reset")
def memory_reset(req: MemoryResetRequest) -> dict:
    """Wipe the agent's Q-table entirely. Lets a user demo from a blank
    slate without restarting the server."""
    session_id = req.session_id
    session = get_session(session_id)
    if session.training_status.get("status") == "running":
        raise HTTPException(
            status_code=409,
            detail="Training is in progress. Wait for it to finish before wiping memory.",
        )
    if session_id in SESSIONS:
        del SESSIONS[session_id]
    return {"status": "ok", "model": _model_summary(session_id)}


@app.post("/train/start")
def train_start(req: TrainStartRequest) -> dict:
    session_id = req.session_id
    session = get_session(session_id)
    pocket = req.target_pocket.upper()
    if pocket not in POCKETS:
        raise HTTPException(status_code=400, detail="Unknown target_pocket")
    bounces = int(req.target_bounces)
    if bounces not in (0, 1, 2, 3):
        raise HTTPException(
            status_code=400, detail="target_bounces must be 0, 1, 2, or 3"
        )

    if not session.training_lock.acquire(blocking=False):
        return {"status": "already_running", **session.training_status}

    timesteps = max(200, min(50_000, int(req.timesteps)))
    anchor = req.anchor_angle_deg
    if anchor is not None:
        # Snap into [0, 360) before passing down.
        anchor = float(anchor) % 360.0
    thread = threading.Thread(
        target=_training_thread,
        args=(
            session_id,
            pocket,
            bounces,
            timesteps,
            int(req.seed),
            float(req.learning_rate),
            float(req.eps_start),
            float(req.eps_end),
            float(req.eps_decay_frac),
            float(req.pace_ms),
            bool(req.reset_q),
            anchor,
        ),
        daemon=True,
        name=f"rtt-train-{session_id}-{variant_id(pocket, bounces)}",
    )
    thread.start()
    return {
        "status": "started",
        "target_pocket": pocket,
        "target_bounces": bounces,
        "variant_id": variant_id(pocket, bounces),
        "timesteps": timesteps,
        "seed": int(req.seed),
        "reset_q": bool(req.reset_q),
    }


@app.get("/train/status")
def train_status(session_id: str) -> dict:
    session = get_session(session_id)
    return dict(session.training_status)


@app.websocket("/ws/train/{session_id}")
async def ws_train(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    session = get_session(session_id)
    sent = 0
    try:
        await websocket.send_json(
            {"type": "status", "payload": dict(session.training_status)}
        )
        while sent < len(session.training_buffer):
            await websocket.send_json(
                {"type": "episode", "payload": session.training_buffer[sent]}
            )
            sent += 1

        last_status = session.training_status["status"]
        idle_ticks = 0
        max_idle_ticks = 600

        while True:
            await asyncio.sleep(0.05)
            while sent < len(session.training_buffer):
                await websocket.send_json(
                    {"type": "episode", "payload": session.training_buffer[sent]}
                )
                sent += 1
                idle_ticks = 0

            if session.training_status["status"] != last_status:
                last_status = session.training_status["status"]
                await websocket.send_json(
                    {"type": "status", "payload": dict(session.training_status)}
                )
                idle_ticks = 0

            if last_status in ("done", "error"):
                while sent < len(session.training_buffer):
                    await websocket.send_json(
                        {"type": "episode", "payload": session.training_buffer[sent]}
                    )
                    sent += 1
                summary = dict(session.training_status)
                summary["model"] = _model_summary(session_id)
                await websocket.send_json({"type": "complete", "payload": summary})
                break

            if last_status == "idle":
                idle_ticks += 1
                if idle_ticks > max_idle_ticks:
                    break
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover
        try:
            await websocket.send_json(
                {"type": "error", "payload": {"message": repr(exc)}}
            )
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "5000"))
    host = os.getenv("HOST", "127.0.0.1")
    uvicorn.run("demo:app", host=host, port=port, reload=False)
