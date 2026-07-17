#!/usr/bin/env python3
"""Build transcripts/official_abstracts.json from the conference schedule CSV.

The authoritative schedule (titles, speakers, affiliations and abstracts) lives
in assets/schedule-abstracts.csv — the same file the public schedule page is
generated from (https://sbi-galev.github.io/2026/schedule.html, which links the
CSV as its editable source). This script maps each CSV row that carries an
abstract onto its saved talk folder and writes the sidecar

    transcripts/official_abstracts.json  =  {folder_name: {speaker, affiliation,
                                             title, time, abstract}}

which transcript_server injects into each talk's summary at load/generation time
(so it shows on the per-talk and per-day pages and feeds the white paper).

Matching is by conference day + normalised speaker name (falling back to the
folder slug), which is exact enough to reproduce every previously-present entry.
Rows without an abstract (intros, breaks, discussions, closing remarks) are
skipped — those talks legitimately have no schedule abstract. Any existing entry
that the CSV cannot reproduce is preserved, so a rebuild never drops an abstract.

Usage:
    python3 build_official_abstracts.py [--csv PATH] [--dry-run]

Re-run whenever the schedule CSV changes, then re-export/publish the site.
"""
import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

import transcript_server as ts

CSV_DEFAULT = Path(__file__).parent / "assets" / "schedule-abstracts.csv"
OUT = ts.SAVE_DIR / "official_abstracts.json"

# CSV 'day' label -> ISO date used in the folder-name prefix.
DAY_TO_DATE = {
    "tuesday (june 23rd)": "2026-06-23",
    "wednesday (june 24th)": "2026-06-24",
    "thursday (june 25th)": "2026-06-25",
    "friday (june 26th)": "2026-06-26",
}


def _norm(s: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _slug(folder: str) -> str:
    return re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_", "", folder)


def load_csv_rows(csv_path: Path):
    """Return [(date, {normalised names}, row)] for CSV rows that have an abstract."""
    out = []
    for r in csv.DictReader(csv_path.open()):
        if not (r.get("abstract") or "").strip():
            continue
        date = DAY_TO_DATE.get((r.get("day") or "").strip().lower())
        if not date:
            continue
        names = {_norm(r.get("name")), _norm(r.get("speakers"))} - {""}
        out.append((date, names, r))
    return out


def build(csv_path: Path):
    rows = load_csv_rows(csv_path)
    existing = {}
    if OUT.exists():
        try:
            existing = json.loads(OUT.read_text())
        except Exception:
            existing = {}

    result, matched_folders, unmatched = {}, [], []
    for talk_dir in sorted(p for p in ts.SAVE_DIR.iterdir() if p.is_dir()):
        folder = talk_dir.name
        if not re.match(r"^\d{4}-\d{2}-\d{2}_", folder):
            continue
        date = folder[:10]
        speaker = ""
        sj = talk_dir / "summary.json"
        if sj.exists():
            try:
                speaker = json.loads(sj.read_text()).get("speaker") or ""
            except Exception:
                pass
        cands = {_norm(speaker), _norm(_slug(folder).replace("-", " "))} - {""}
        hit = next((r for (d, names, r) in rows if d == date and names & cands), None)
        if hit:
            result[folder] = {
                "speaker": hit.get("name") or hit.get("speakers") or speaker,
                "affiliation": (hit.get("affiliation") or "").strip(),
                "title": (hit.get("title") or "").strip(),
                "time": (hit.get("time") or "").strip(),
                "abstract": hit["abstract"].strip(),
            }
            matched_folders.append(folder)
        else:
            unmatched.append((folder, speaker))

    # Never drop an abstract that already worked but the CSV can't reproduce.
    preserved = [k for k in existing if k not in result and existing[k].get("abstract")]
    for k in preserved:
        result[k] = existing[k]

    return result, matched_folders, unmatched, preserved


def main():
    ap = argparse.ArgumentParser(description="Build official_abstracts.json from the schedule CSV.")
    ap.add_argument("--csv", type=Path, default=CSV_DEFAULT, help="schedule-abstracts.csv path")
    ap.add_argument("--dry-run", action="store_true", help="report matches without writing")
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    result, matched, unmatched, preserved = build(args.csv)
    by_day = {}
    for f in result:
        by_day[f[:10]] = by_day.get(f[:10], 0) + 1
    print(f"matched {len(matched)} talks to a schedule abstract; "
          f"{len(preserved)} preserved from the existing file.")
    for day in sorted(by_day):
        print(f"  {day}: {by_day[day]} abstract(s)")
    if unmatched:
        print(f"\nno schedule abstract (skipped — intros/breaks/discussions/closing): {len(unmatched)}")
        for folder, sp in unmatched:
            print(f"  {folder}  (speaker={sp!r})")

    if args.dry_run:
        print("\n--dry-run: not writing.")
        return
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {OUT}  ({len(result)} entries)")


if __name__ == "__main__":
    main()
