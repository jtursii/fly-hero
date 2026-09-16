"""Thin gymnasium-flavored wrapper around sim.VecRhythmEnv (PLAN Phase 2
task 10's repo layout keeps sim.py and env.py separate; the simulator
mechanics -- RuleEngine driving, retina/render calls -- live in sim.py since
render_debug_video needs the same pieces without going through an env
reset/step cycle)."""

from __future__ import annotations

from flyhero.game.sim import VecRhythmEnv

__all__ = ["VecRhythmEnv"]
