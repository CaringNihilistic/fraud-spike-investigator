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
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.envfile import load_env  # noqa: E402

load_env()  # picks up ANTHROPIC_API_KEY from .env if present

import uvicorn  # noqa: E402

from src.serve import api as api_mod  # noqa: E402
from src.serve import replay  # noqa: E402
from src.serve.state import STATE  # noqa: E402


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
    args = ap.parse_args()

    STATE.speed = args.speed
    api_mod.set_agent_enabled(not args.no_agent)
    url = f"http://{args.host}:{args.port}"

    print("=" * 68)
    print("  Fraud Spike Investigator - demo")
    print("=" * 68)
    print("  Training the model and scoring the test slice (~20s)...")

    def _prepare_and_replay():
        scored, ctx = replay.prepare()
        api_mod.set_context(ctx)
        STATE.total = len(scored)
        print(f"  Ready. Replaying {len(scored):,} transactions at {args.speed:.0f}/sec")
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
        threading.Thread(
            target=lambda: replay.run(scored, ctx, investigate_enabled=not args.no_agent),
            daemon=True).start()
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
