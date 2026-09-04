"""Single-command demo: trains, starts the API, and replays the test slice.

    python run_demo.py                 # http://127.0.0.1:8000
    python run_demo.py --speed 400     # faster stream
    python run_demo.py --no-agent      # skip investigations (no API calls)

Everything runs in ONE process: FastAPI serves the SPA and the API, a daemon
thread replays the test stream through the real pipeline, and investigations
run on their own short-lived threads so a slow agent never stalls the stream.
No Docker, no Node build, no external services.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.envfile import load_env  # noqa: E402

load_env()  # picks up ANTHROPIC_API_KEY from .env if present

import uvicorn  # noqa: E402

from src.serve import api as api_mod  # noqa: E402
from src.serve import replay  # noqa: E402
from src.serve.state import STATE  # noqa: E402


def should_loop(loop: bool, no_loop: bool, defer_prepare: bool) -> bool:
    """Does the replay repeat? Derived from whether we are HOSTED.

    --defer-prepare already means "running on a PaaS host" (see its help text).
    On such a host an uptime pinger keeps the free instance warm, so it never
    cold-boots - and without a loop every visitor for the next month lands on a
    FINISHED replay. The cold start was the only thing making the demo look
    live; removing it without this makes the demo strictly worse.

    Derived rather than configured because the first attempt WAS configured:
    --loop went into render.yaml, Render redeployed the code but not the
    blueprint's start command, and the live service kept serving one finished
    pass. Caught by polling /api/status - it sat at 14160/14160 for 92 seconds
    when a restart was due at 42. A setting that only takes effect if a
    dashboard is also updated is a setting that will eventually be wrong.
    """
    return (loop or defer_prepare) and not no_loop


def main():
    ap = argparse.ArgumentParser(description="Fraud Spike Investigator demo")
    ap.add_argument("--speed", type=float, default=250.0,
                    help="replay rate in transactions/sec (default 250 ~ 60s for the full slice)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-agent", action="store_true",
                    help="disable LLM investigations (dashboard still runs)")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--defer-prepare", action="store_true",
                    help="bind the port immediately and train in the background. "
                         "Required on PaaS hosts (Render/Railway/Fly), which kill a "
                         "service that has not bound its port within a short window - "
                         "and on a shared-CPU free instance, training first can exceed it.")
    ap.add_argument("--loop", action="store_true",
                    help="replay on repeat instead of stopping when the slice ends. "
                         "ON BY DEFAULT whenever --defer-prepare is set, i.e. on a "
                         "hosted deployment; this flag only exists to force it on "
                         "locally. Trains once, streams many.")
    ap.add_argument("--no-loop", action="store_true",
                    help="force a single pass even on a hosted deployment")
    ap.add_argument("--loop-pause", type=float, default=20.0,
                    help="seconds to hold the finished board before restarting, so "
                         "the final totals are readable (default 20)")
    args = ap.parse_args()

    args.loop = should_loop(args.loop, args.no_loop, args.defer_prepare)

    STATE.speed = args.speed
    api_mod.set_agent_enabled(not args.no_agent)
    url = f"http://{args.host}:{args.port}"

    print("=" * 68)
    print("  Fraud Spike Investigator - demo")
    print("=" * 68)
    print("  Training the model and scoring the test slice (~20s)...")

    def _replay_forever(scored, ctx):
        """Replay, show the finished board for a beat, reset, replay again.

        Only the REPLAY repeats - prepare() is not called again, so the model
        is trained exactly once. Each pass streams the same slice through the
        real fusion -> policy path, so 'not a canned animation' still holds:
        the decisions are recomputed every time, not replayed from a recording.
        """
        while True:
            replay.run(scored, ctx, investigate_enabled=not args.no_agent)
            # Hold on the finished board. Pausing MID-replay already freezes
            # everything correctly - the transaction loop in replay.run()
            # blocks on STATE.paused and never returns, so this line is never
            # reached. But once a pass finishes, the old code slept a fixed
            # loop_pause and reset REGARDLESS of pause - so pausing to narrate
            # over a just-finished board still got reset out from under you
            # partway through. Poll pause instead of sleeping through it once.
            held = 0.0
            while held < args.loop_pause or STATE.paused:
                time.sleep(0.5)
                held += 0.5
            replay.INVESTIGATED.clear()   # else investigations never re-fire
            STATE.reset()
            STATE.log_event("system", "replay restarting")

    def _prepare_and_replay():
        scored, ctx = replay.prepare()
        api_mod.set_context(ctx)
        STATE.total = len(scored)
        print(f"  Ready. Replaying {len(scored):,} transactions at {args.speed:.0f}/sec")
        if args.loop:
            _replay_forever(scored, ctx)
        else:
            replay.run(scored, ctx, investigate_enabled=not args.no_agent)

    if args.defer_prepare:
        # Port first, training second. The dashboard polls /api/status and
        # renders an empty grid for a few seconds, which is a better failure
        # mode than the host killing the service before it ever binds.
        threading.Thread(target=_prepare_and_replay, daemon=True).start()
    else:
        # Local default: prepare synchronously so the dashboard is never served
        # against empty state - a judge loading the page mid-training would see
        # a blank grid.
        scored, ctx = replay.prepare()
        api_mod.set_context(ctx)
        STATE.total = len(scored)
        # Both branches go through the same runner. An earlier version wired
        # --loop into the deferred path only, so it silently did nothing
        # locally - the flag parsed, printed no error, and just never looped.
        _run = (lambda: _replay_forever(scored, ctx)) if args.loop else \
               (lambda: replay.run(scored, ctx, investigate_enabled=not args.no_agent))
        threading.Thread(target=_run, daemon=True).start()
        print(f"  Ready. Dashboard: {url}")
        print(f"  Replaying {len(scored):,} transactions at {args.speed:.0f}/sec")
    print("  Watch for: attack merchants spike -> investigation fires ->")
    print("             flash-sale merchant m11 spikes in VOLUME and is NOT flagged.")
    if args.no_agent:
        print("  (investigations disabled via --no-agent)")
    print("=" * 68)

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    uvicorn.run(api_mod.app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
