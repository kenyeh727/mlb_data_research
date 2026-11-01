"""Whiff Scout report generator: Scout -> Data -> Analyst -> HTML report.

Usage (from this whiff-scout subfolder):
    python -m src.report "Drew Rasmussen"
    python -m src.report "Jacob Misiorowski"

Writes reports/<Pitcher_Name>.html. Each role is a function that appends
human-readable step logs, which are embedded in the report so the agent's
work is visible, not just its output.
"""
import argparse
import html
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from src.data_tool import load_pitches  # noqa: E402
from src.inference import load_model, run_inference  # noqa: E402
from src.meta_client import generate_prose, get_api_key  # noqa: E402

REPORTS_DIR = os.path.join(HERE, "reports")
SMALL_SAMPLE_N = 50


# ---------------- roles ----------------

def scout(pitcher_name: str, log: list) -> str:
    """Scout (coordinator): decide who to scout and what to fetch."""
    log.append(f"[Scout] Request: scouting report for {pitcher_name}.")
    log.append("[Scout] Plan: resolve to Statcast ID, load cached 2026 "
               "pitches, run the whiff model, write the report.")
    return pitcher_name


def data_step(pitcher_name: str, log: list) -> tuple:
    """Data (fetcher): resolve the pitcher and load cached pitches."""
    pid, pitches, window = load_pitches(pitcher_name)
    log.append(f"[Data] Resolved {pitcher_name} to Statcast ID {pid}.")
    log.append(f"[Data] Loaded {window['n_pitches']:,} pitches from "
               f"{window['source']}.")
    log.append(f"[Data] Sample window: {window['start']} to {window['end']}; "
               f"game types: {window['game_types']}.")
    return pid, pitches, window


def analyst(pitcher_name: str, pitches, window: dict, log: list) -> dict:
    """Analyst: run the model and write the prose (live or template)."""
    model = load_model()
    log.append("[Analyst] Loaded whiff GBM (strict labels, 75 trees).")
    result = run_inference(model, pitches)
    log.append(f"[Analyst] Scored {result['n_swings']:,} swings: overall "
               f"actual {result['overall_actual']:.1%} vs predicted "
               f"{result['overall_predicted']:.1%}.")
    result["prose"], result["prose_error"] = write_prose(
        pitcher_name, result, window, log)
    return result


# ---------------- prose ----------------

def _pct(x: float) -> str:
    return f"{x:.1%}"


def template_insights(pitcher_name: str, result: dict) -> list:
    """Rule-based insights from the numbers; no invented claims."""
    reliable = [r for r in result["by_pitch"] if r["n"] >= SMALL_SAMPLE_N]
    pool = reliable or result["by_pitch"]
    out_pitch = max(pool, key=lambda r: r["actual"])
    weak_pitch = min(pool, key=lambda r: r["actual"])
    gap = max(pool, key=lambda r: abs(r["actual"] - r["predicted"]))
    direction = ("overestimates" if gap["predicted"] > gap["actual"]
                 else "underestimates")
    lines = [
        f"Out pitch: {out_pitch['name']} ({out_pitch['code']}) at "
        f"{_pct(out_pitch['actual'])} actual whiff over "
        f"{out_pitch['n']} swings.",
        f"Lowest-whiff offering: {weak_pitch['name']} "
        f"({weak_pitch['code']}) at {_pct(weak_pitch['actual'])} actual "
        f"whiff over {weak_pitch['n']} swings.",
        f"Largest model gap: {gap['name']} ({gap['code']}) -- model "
        f"{direction} whiff ({_pct(gap['predicted'])} predicted vs "
        f"{_pct(gap['actual'])} actual).",
    ]
    if not reliable:
        lines.append("All pitch types are small-sample in this window; "
                     "treat every rate as uncertain.")
    stands = result["by_stand"]
    if "L" in stands and "R" in stands:
        diff = stands["L"]["actual"] - stands["R"]["actual"]
        side = "left-handed" if diff > 0 else "right-handed"
        lines.append(
            f"Matchup lean: whiff rate is higher vs {side} batters "
            f"(LHB {_pct(stands['L']['actual'])} vs RHB "
            f"{_pct(stands['R']['actual'])} actual).")
    return lines


def write_prose(pitcher_name: str, result: dict, window: dict,
                log: list) -> tuple:
    """Try live Muse Spark prose; fall back to template prose."""
    table = "\n".join(
        f"- {r['name']} ({r['code']}): n={r['n']}, actual={_pct(r['actual'])}, "
        f"predicted={_pct(r['predicted'])}" for r in result["by_pitch"])
    prompt = (
        f"Write a short scouting report section for pitcher {pitcher_name} "
        f"from ONLY these numbers (overall actual whiff "
        f"{_pct(result['overall_actual'])}, predicted "
        f"{_pct(result['overall_predicted'])} over "
        f"{result['n_swings']} swings; sample window "
        f"{window['start']} to {window['end']}):\n{table}\n"
        "Give 3-4 bullet insights (out pitch, weakest pitch, biggest "
        "model gap) and one short matchup-advice paragraph. Do not invent "
        "any numbers or pitches not listed."
    )
    if get_api_key() is None:
        log.append("[Analyst] No META_API_KEY in .env -- using template "
                   "prose. Live Muse Spark prose needs the key.")
        return template_insights(pitcher_name, result), \
            "live Muse Spark prose needs META_API_KEY in .env"
    text, err = generate_prose(prompt)
    if err:
        log.append(f"[Analyst] Muse Spark call failed ({err}) -- using "
                   "template prose.")
        return template_insights(pitcher_name, result), err
    log.append("[Analyst] Prose written live by Muse Spark.")
    return [text], None


# ---------------- HTML ----------------

def render_html(pitcher_name: str, window: dict, result: dict,
                log: list) -> str:
    def row(r):
        flag = "" if r["n"] >= SMALL_SAMPLE_N else \
            " <span class='flag'>(small sample)</span>"
        return (f"<tr><td>{html.escape(r['name'])} ({r['code']})</td>"
                f"<td>{r['n']}</td><td>{_pct(r['actual'])}</td>"
                f"<td>{_pct(r['predicted'])}</td><td>{flag}</td></tr>")

    mix_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v:.1f}%</td></tr>"
        for k, v in sorted(result["pitch_mix_pct"].items(),
                           key=lambda kv: -kv[1]))
    pitch_rows = "".join(row(r) for r in result["by_pitch"])
    stand_rows = "".join(
        f"<tr><td>{'Left-handed' if s == 'L' else 'Right-handed' if s == 'R' else s}</td>"
        f"<td>{v['n']}</td><td>{_pct(v['actual'])}</td>"
        f"<td>{_pct(v['predicted'])}</td></tr>"
        for s, v in sorted(result["by_stand"].items()))
    insights = "".join(f"<li>{html.escape(i)}</li>"
                       for i in result["prose"])
    steps = "".join(f"<li>{html.escape(s)}</li>" for s in log)
    types_note = ", ".join(f"{k}: {v}"
                           for k, v in window["game_types"].items())
    prose_note = ("" if result["prose_error"] is None else
                  f"<p class='note'>Prose note: {html.escape(result['prose_error'])}; "
                  "numbers below are unaffected.</p>")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Whiff Scout: {html.escape(pitcher_name)}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:760px;margin:2em auto;
padding:0 1em;color:#222}} table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccc;padding:6px 10px;text-align:left}}
th{{background:#f0f0f0}} .flag{{color:#b00;font-size:.85em}}
.note{{background:#fff8e1;border:1px solid #e6c200;padding:10px}}
.small{{color:#666;font-size:.9em}}
</style></head><body>
<h1>Whiff Scout: {html.escape(pitcher_name)}</h1>
<p class="small">Sample window: {window['start']} to {window['end']} "
({window['n_pitches']:,} pitches, {result['n_swings']:,} swings).
Game types in sample: {html.escape(types_note)}. This window mixes
regular season, spring training, and postseason games; it is NOT a
pure regular-season sample.</p>
{prose_note}
<h2>Pitch mix (share of all pitches)</h2>
<table><tr><th>Pitch</th><th>Usage</th></tr>{mix_rows}</table>
<h2>Whiff rate by pitch type</h2>
<table><tr><th>Pitch</th><th>Swings (n)</th><th>Actual whiff</th>
<th>Model predicted</th><th></th></tr>{pitch_rows}
<tr><td><b>Overall</b></td><td><b>{result['n_swings']}</b></td>
<td><b>{_pct(result['overall_actual'])}</b></td>
<td><b>{_pct(result['overall_predicted'])}</b></td><td></td></tr></table>
<h2>Whiff vs batter handedness</h2>
<table><tr><th>Batters</th><th>Swings (n)</th><th>Actual whiff</th>
<th>Model predicted</th></tr>{stand_rows}</table>
<h2>Insights &amp; matchup advice</h2><ul>{insights}</ul>
<h2>How this report was made</h2><ul>{steps}</ul>
<p class="small">Model: GBM whiff predictor, strict labels (foul tips excluded),
75 trees, validation AUC 0.777 (95% CI 0.766-0.789), trained on ken's own
MLB Statcast project. Data: Statcast via pybaseball (local cache).
Small-sample rates (n &lt; 50) carry wide uncertainty and are flagged.</p>
</body></html>"""


def build_report(pitcher_name: str) -> tuple:
    """Run Scout -> Data -> Analyst; return (html_path, result, log)."""
    log: list = []
    scout(pitcher_name, log)
    _pid, pitches, window = data_step(pitcher_name, log)
    result = analyst(pitcher_name, pitches, window, log)
    out = render_html(pitcher_name, window, result, log)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fname = "_".join(pitcher_name.split()) + ".html"
    path = os.path.join(REPORTS_DIR, fname)
    with open(path, "w") as f:
        f.write(out)
    return path, result, log


def main() -> None:
    ap = argparse.ArgumentParser(description="Whiff Scout report CLI")
    ap.add_argument("pitcher", help="Pitcher full name, e.g. 'Drew Rasmussen'")
    args = ap.parse_args()
    path, result, log = build_report(args.pitcher)
    for line in log:
        print(line)
    print(f"Report written to {path}")
    print(f"Overall: actual {_pct(result['overall_actual'])} vs predicted "
          f"{_pct(result['overall_predicted'])} "
          f"({result['n_swings']} swings)")


if __name__ == "__main__":
    main()
