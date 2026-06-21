"""Assemble the v2 report: updated experimental sections (from the workflow) +
the still-valid theory sections carried over from the old report.
  python examples/_assemble_report2.py <workflow_output.json> <old_report.html> <out.html>"""
import json
import re
import sys

NEW_ORDER = [
    ("overview", "1 · Overview & the corrected task"),
    ("journey", "2 · Experimental journey & comparisons"),
    ("insights", "3 · Load-bearing insights"),
    ("architecture", "4 · Architecture: frontier + two heads"),
    ("reward_curriculum", "5 · Reward design & the curriculum"),
    ("size_invariance", "6 · Size-invariance"),
]
THEORY = [
    ("grace", "7 · Hard comms vs a grace value (future)"),
    ("phases", "8 · Mission phase transitions (future)"),
    ("roles", "9 · Multi-role policies (future)"),
    ("energy_nets", "10 · Energy budget & network capacity (future)"),
]

src, old, out = sys.argv[1], sys.argv[2], sys.argv[3]
new = {d["key"]: d["html"] for d in json.load(open(src))["result"]}
old_html = open(old).read() if old != "-" else ""
theory = {}
for k, _ in THEORY:
    m = re.search(rf'<section id="{k}">(.*?)</section>', old_html, re.S)
    if m:
        theory[k] = m.group(1)

parts, toc = [], []
for k, title in NEW_ORDER:
    if k in new:
        toc.append(f'<li><a href="#{k}">{title}</a></li>')
        parts.append(f'<section id="{k}">\n{new[k]}\n</section>')
for k, title in THEORY:
    if k in theory:
        toc.append(f'<li><a href="#{k}">{title}</a></li>')
        parts.append(f'<section id="{k}">\n{theory[k]}\n</section>')

HTML = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Swarm Coverage MARL — Research Report v2</title>
<style>
 :root {{ --ink:#1a1a1a; --mut:#666; --line:#e1e4e8; --accent:#2c4f8a; }}
 body {{ font-family:-apple-system,system-ui,sans-serif; max-width:880px; margin:36px auto; color:var(--ink); padding:0 20px; line-height:1.5; }}
 h1 {{ font-size:27px; margin-bottom:2px; }} .sub {{ color:var(--mut); font-size:14px; margin-top:0; }}
 h2 {{ font-size:20px; margin-top:34px; padding-bottom:5px; border-bottom:2px solid var(--accent); }}
 h3 {{ font-size:15px; color:#333; margin-top:20px; }}
 table {{ border-collapse:collapse; width:100%; margin:12px 0; font-size:13px; }}
 th,td {{ border:1px solid var(--line); padding:5px 9px; text-align:left; vertical-align:top; }} th {{ background:#f4f6fa; }}
 code {{ background:#f4f4f4; padding:1px 4px; border-radius:3px; font-size:12.5px; }}
 .toc {{ background:#f9fafb; border:1px solid var(--line); border-radius:8px; padding:10px 14px 10px 30px; }}
 .toc li {{ margin:3px 0; }} a {{ color:var(--accent); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
 .meta {{ font-size:13px; color:var(--mut); }} b {{ color:#111; }}
</style></head><body>
<h1>Stretching the swarm: size-invariant coverage, formation & hard collision</h1>
<p class="sub">JAX/PPO cooperative MARL · 1×1 visit-coverage · dev 10×10 → real 16×16 · research report (living document)</p>
<p class="meta">Companion to <code>docs/research-journal.md</code>. Sections 1–6 are the corrected visit-coverage
study; 7–10 are forward-looking theory. 16×16 validation in progress.</p>
<div class="toc"><b>Contents</b><ul>
{chr(10).join(toc)}
</ul></div>
{chr(10).join(parts)}
<hr style="margin-top:40px;border:none;border-top:1px solid var(--line)">
<p class="meta">Generated from parallel section drafts + experiment logs.</p>
</body></html>
"""
open(out, "w").write(HTML)
print(f"wrote {out}  ({len(HTML)} bytes, {len(parts)} sections: {len([k for k,_ in NEW_ORDER if k in new])} new + {len(theory)} theory)")
