#!/usr/bin/env python
"""Overnight experiment driver for Zymera_env.

Runs a manifest of experiment jobs sequentially, each isolated (one failure does
not stop the rest), with a live STATUS.html dashboard + optional ntfy push.

  python -m experiments.run_overnight --dry-run     # tiny CPU smoke of every job
  python -m experiments.run_overnight --gpu          # full runs
  python -m experiments.run_overnight --only belief  # one job
  python -m experiments.run_overnight --gpu --resume # skip already-done jobs

Leave it overnight (on the server):
  tmux new -d -s zymera-gpu '.venv/bin/python -m experiments.run_overnight --gpu'

Phone pings: export ZYMERA_NTFY=<your-ntfy-topic>   (then watch ntfy.sh/<topic>)
"""
from __future__ import annotations
import argparse, datetime, html, json, os, sys, time, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                       # so swarm_explore / zymera import without install
RUNS = os.path.join(ROOT, "experiments", "runs")
os.makedirs(RUNS, exist_ok=True)


class RunCtx:
    """Handed to each job: scoped run dir, logging, and metric recording."""
    def __init__(self, name, mode):
        self.name, self.mode = name, mode
        self.dir = os.path.join(RUNS, name); os.makedirs(self.dir, exist_ok=True)
        self.t0 = time.time(); self.metrics = {}
        self._log = open(os.path.join(self.dir, "log.txt"), "a")

    def log(self, *a):
        m = " ".join(str(x) for x in a)
        print(f"  [{self.name}] {m}", flush=True); self._log.write(m + "\n"); self._log.flush()

    def metric(self, **kw):
        self.metrics.update(kw)
        json.dump(self.metrics, open(os.path.join(self.dir, "metrics.json"), "w"), indent=2, default=float)

    def close(self):
        self._log.close()


def _notify(msg):
    topic = os.environ.get("ZYMERA_NTFY")
    if not topic:
        return
    try:
        import urllib.request
        urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{topic}", data=msg.encode()), timeout=5)
    except Exception:
        pass


def _dashboard(state):
    icon = {"queued": "•", "running": "⏳", "done": "✓", "failed": "✗", "skipped": "—"}
    col = {"done": "#1f7a44", "failed": "#c0392b", "running": "#1f6feb", "skipped": "#5b6570", "queued": "#5b6570"}
    rows = ""
    for j in state["jobs"]:
        st = j["status"]
        m = " · ".join(f"{k} {v}" for k, v in (j.get("metrics") or {}).items())
        rows += (f"<tr><td>{icon.get(st,'?')} <b style='color:{col[st]}'>{st}</b></td>"
                 f"<td><b>{html.escape(j['name'])}</b></td><td>{j.get('elapsed','')}</td>"
                 f"<td style='font-size:12px;color:#445'>{html.escape(m)}</td></tr>")
    doc = (f"<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=20>"
           f"<title>Zymera runs</title><style>body{{font-family:-apple-system,Segoe UI,sans-serif;"
           f"max-width:840px;margin:26px auto;padding:0 20px;color:#1f2328}}table{{border-collapse:collapse;width:100%}}"
           f"td,th{{border:1px solid #e3e6ea;padding:8px 11px;text-align:left}}th{{background:#f1f3f5}}"
           f".m{{color:#5b6570;font-size:13px}}</style>"
           f"<h2>Zymera — overnight runs</h2><p class=m>mode <b>{state['mode']}</b> · device "
           f"{html.escape(state['device'])} · updated {state['updated']}</p>"
           f"<table><tr><th>status</th><th>experiment</th><th>elapsed</th><th>metrics</th></tr>{rows}</table>")
    open(os.path.join(ROOT, "experiments", "STATUS.html"), "w").write(doc)
    json.dump(state, open(os.path.join(ROOT, "experiments", "status.json"), "w"), indent=2, default=str)


def main():
    from experiments.jobs import MANIFEST
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="full-scale runs")
    ap.add_argument("--dry-run", action="store_true", help="tiny smoke of every job (default if no --gpu)")
    ap.add_argument("--only", nargs="*", help="run only these job names")
    ap.add_argument("--resume", action="store_true", help="skip jobs already marked DONE")
    a = ap.parse_args()
    mode = "gpu" if (a.gpu and not a.dry_run) else "smoke"
    jobs = [j for j in MANIFEST if not a.only or j["name"] in a.only]

    try:
        import jax; dev = str(jax.devices()[0])
    except Exception as e:
        dev = f"(no jax: {e})"
    print(f"=== Zymera overnight · mode={mode} · device={dev} · {len(jobs)} job(s) ===", flush=True)

    state = {"mode": mode, "device": dev, "updated": "",
             "jobs": [{"name": j["name"], "status": "queued", "elapsed": "", "metrics": {}} for j in jobs]}

    def touch():
        state["updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"); _dashboard(state)
    touch()

    for i, j in enumerate(jobs):
        js = state["jobs"][i]
        done = os.path.join(RUNS, j["name"], "DONE")
        if a.resume and os.path.exists(done):
            js["status"] = "skipped"; touch(); _notify(f"skip {j['name']} (already done)"); continue
        ctx = RunCtx(j["name"], mode)
        js["status"] = "running"; touch(); _notify(f"▶ {j['name']} started ({mode})")
        try:
            j["fn"](ctx, **j.get(mode, j.get("smoke", {})))
            js["status"] = "done"; js["metrics"] = ctx.metrics
            open(done, "w").write(datetime.datetime.now().isoformat())
            _notify(f"✓ {j['name']} done · " + ", ".join(f"{k}={v}" for k, v in ctx.metrics.items()))
        except Exception:
            tb = traceback.format_exc(); ctx.log("FAILED:\n" + tb)
            js["status"] = "failed"; js["metrics"] = {"error": (tb.strip().splitlines() or ["?"])[-1][:140]}
            _notify(f"✗ {j['name']} FAILED: {js['metrics']['error']}")
        finally:
            js["elapsed"] = f"{time.time()-ctx.t0:.0f}s"; ctx.close(); touch()

    print("=== all jobs finished ===", flush=True)
    _notify("🏁 overnight finished — " + ", ".join(f"{j['name']}:{js['status']}" for j, js in zip(jobs, state["jobs"])))


if __name__ == "__main__":
    main()
