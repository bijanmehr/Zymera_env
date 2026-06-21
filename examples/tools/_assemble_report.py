"""Assemble the parallel-drafted section fragments into a single self-contained
research-report HTML. Usage: python examples/_assemble_report.py <workflow_output.json> <out.html>"""
import json
import sys

ORDER = ["journey", "tension", "cbf", "grace", "phases", "roles", "energy_nets"]
TITLES = {
    "journey": "1 · Experimental journey & results",
    "tension": "2 · The coverage–connectivity tension",
    "cbf": "3 · CBF analysis: connectivity vs collision",
    "grace": "4 · Hard comms graph vs a grace value",
    "phases": "5 · Phase transitions in the mission",
    "roles": "6 · Multi-role policies & role switching",
    "energy_nets": "7 · Energy budget & network capacity",
}

src, out = sys.argv[1], sys.argv[2]
data = json.load(open(src))["result"]
by_key = {d["key"]: d["html"] for d in data}

toc = "\n".join(f'<li><a href="#{k}">{TITLES[k]}</a></li>' for k in ORDER if k in by_key)
body = "\n".join(f'<section id="{k}">\n{by_key[k]}\n</section>' for k in ORDER if k in by_key)

HTML = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Swarm Coverage MARL — Research Report</title>
<style>
 :root {{ --ink:#1a1a1a; --mut:#666; --line:#e1e4e8; --accent:#2c4f8a; --warn:#8a2c2c; }}
 body {{ font-family:-apple-system,system-ui,sans-serif; max-width:880px; margin:36px auto;
        color:var(--ink); padding:0 20px; line-height:1.5; }}
 h1 {{ font-size:27px; margin-bottom:2px; }}
 .sub {{ color:var(--mut); font-size:14px; margin-top:0; }}
 h2 {{ font-size:20px; margin-top:34px; padding-bottom:5px; border-bottom:2px solid var(--accent); }}
 h3 {{ font-size:15px; color:#333; margin-top:20px; }}
 section {{ scroll-margin-top:16px; }}
 table {{ border-collapse:collapse; width:100%; margin:12px 0; font-size:13px; }}
 th,td {{ border:1px solid var(--line); padding:5px 9px; text-align:left; vertical-align:top; }}
 th {{ background:#f4f6fa; }}
 code {{ background:#f4f4f4; padding:1px 4px; border-radius:3px; font-size:12.5px; }}
 .toc {{ background:#f9fafb; border:1px solid var(--line); border-radius:8px; padding:10px 14px 10px 30px; }}
 .toc li {{ margin:3px 0; }} a {{ color:var(--accent); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
 .meta {{ font-size:13px; color:var(--mut); }}
 b {{ color:#111; }}
</style></head>
<body>
<h1>Stretching the swarm: coverage vs connectivity in cooperative MARL</h1>
<p class="sub">JAX/PPO study on a 16×16, 4-agent range-limited comm-coverage task · research report (living document)</p>
<p class="meta">Companion to <code>docs/research-journal.md</code>. Numbers are from runs under
<code>experiments/marl/</code>. Sections 4–7 are design/theory for the open problem (connectivity) and
future missions. CBF+attention and CBF-collision-only experiments are pending.</p>
<div class="toc"><b>Contents</b><ul>
{toc}
</ul></div>
{body}
<hr style="margin-top:40px;border:none;border-top:1px solid var(--line)">
<p class="meta">Generated from parallel section drafts + experiment logs. Update by re-running the report workflow
and <code>_assemble_report.py</code>.</p>
</body></html>
"""
open(out, "w").write(HTML)
print(f"wrote {out}  ({len(HTML)} bytes, {len([k for k in ORDER if k in by_key])} sections)")
