#!/usr/bin/env python3
"""Generate a conference summary white paper (LaTeX) for SBI4GALEV.

The whole write-up is done by the *local* model ("alan", the same LLM that powers
the talk summaries) so the meeting's content never leaves the local network — no
external AI tool ever sees the transcripts or abstracts.

It assembles the full meeting context that has already been distilled on-box:
  * every talk's summary.json (title, speaker, affiliation, official abstract,
    key points, topics),
  * the per-day editorial overviews (day_summaries/<date>.json),
  * the conference-wide hot-topics synthesis (topic_summaries/topics.json),
and (optionally) the *previous year's* abstract booklet, then asks alan to write
a <=3 page white paper covering: what was discussed, the hot subjects, how the
meeting compares with last year, and an outlook for the future.

It is built to be automated: re-run it whenever new talks land (optionally with
--refresh to rebuild the day/topic syntheses first) and it regenerates the paper
from the current state.

By default it first spends a separate "thinking" pass: alan drafts an editorial
plan (a complete topic inventory weighted by how many talks discussed each topic,
with proportional space recommendations) before writing the paper, so coverage is
comprehensive and depth tracks how much each topic was mentioned. Use --no-think
to skip it, or --plan-tokens / --max-tokens / --timeout to give it more room/time.

Usage:
    python3 whitepaper.py [--prev-booklet PATH] [--out FILE.tex] [--pdf]
                          [--refresh] [--pages N] [--budget CHARS]
                          [--dump-context FILE] [--model NAME]
                          [--no-think] [--max-tokens N] [--plan-tokens N]
                          [--timeout SECS] [--dump-plan FILE]

    --prev-booklet PATH   last year's abstract booklet (.pdf or .txt) for the
                          year-on-year comparison. If omitted, looks for
                          $SBI_PREV_BOOKLET, ./prev_year_booklet.{txt,pdf}, then
                          any *abstract*booklet*.pdf dropped in the repo root
                          (newest first). If still missing, that section is
                          written cautiously and flagged as not grounded.
    --pdf                 also compile the .tex to PDF (needs latexmk/pdflatex).
    --refresh             rebuild day overviews + topic synthesis first, so the
                          paper reflects the very latest talks.
    --pages N             target page budget (default 3; soft, via the prompt).
    --budget CHARS        max characters of per-talk abstract context (default
                          90000); titles/speakers for every talk are always
                          included, full abstracts until the budget is hit.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import config
import export_static as es
import transcript_server as ts

CONF_NAME = f"{config.CONF_FULL} ({config.CONF_SHORT}) {config.CONF_YEAR}"
CONF_SHORT = f"{config.CONF_SHORT} {config.CONF_YEAR}"
ASSISTANT = config.ASSISTANT_NAME
ROOT = Path(__file__).parent


# ── context assembly ─────────────────────────────────────────────────────────

def _talk_digest(t: dict) -> dict:
    s = t.get("summary") or {}
    return {
        "date": ts._talk_day(t),
        "when": t["meta"].get("saved_at", ""),
        "speaker": s.get("speaker") or "",
        "affiliation": s.get("affiliation") or "",
        "title": s.get("title") or t.get("title") or t["folder"],
        # Prefer the authoritative schedule abstract; fall back to alan's.
        "abstract": (s.get("official_abstract") or s.get("abstract") or "").strip(),
        "key_points": [p for p in s.get("key_points", []) if p][:5],
        "topics": [x for x in s.get("topics", []) if x][:6],
    }


def _load_occurrences(allowed_folders: set[str]) -> tuple[list[dict], list[dict]]:
    """Return (topics, tools) occurrence tallies with transcript/slide mention
    counts, most-mentioned first. These are the on-box concept/tool extractions
    (transcripts/concepts.json + tools.json), so the summary table alan writes is
    grounded in real counts rather than invented.

    Counts are recomputed from each item's per-talk breakdown, restricted to
    `allowed_folders` (the permission allowlist) — so an unapproved talk never
    contributes to the table. Items with no mentions across approved talks drop
    out entirely."""
    def _load(path: Path, key: str) -> list[dict]:
        try:
            items = json.loads(path.read_text()).get(key) or []
        except Exception:
            return []
        out = []
        for it in items:
            name = it.get("name") or ""
            if not name:
                continue
            tm = sm = n = 0
            for tk in it.get("talks", []):
                if tk.get("folder") in allowed_folders:
                    tm += int(tk.get("t", 0) or 0)
                    sm += int(tk.get("s", 0) or 0)
                    n += 1
            if tm or sm:
                out.append({"name": name, "transcript": tm, "slide": sm, "talks": n})
        out.sort(key=lambda x: (x["transcript"] + x["slide"]), reverse=True)
        return out

    return (_load(ts.SAVE_DIR / "concepts.json", "concepts"),
            _load(ts.SAVE_DIR / "tools.json", "tools"))


def gather_context(budget: int) -> str:
    """Build the meeting-wide context blob fed to alan.

    Always lists every talk (date / speaker / title); includes full abstracts and
    key points until `budget` characters of abstract text are spent, then drops to
    title-only for the remainder so coverage is complete and bounded."""
    talks = [t for t in ts._load_talks() if t.get("summary")]
    talks.sort(key=lambda t: t.get("saved_at") or "")
    digests = [_talk_digest(t) for t in talks]

    days = ts._group_by_day(ts._load_talks())
    parts = [f"CONFERENCE: {CONF_NAME}"]
    if days:
        parts.append(f"DATES: {days[0][0]} to {days[-1][0]}  ·  "
                     f"{len(digests)} talks across {len(days)} days")

    # Per-day editorial overviews (already synthesised by alan from full talks).
    parts.append("\n=== PER-DAY OVERVIEWS ===")
    for date, day_label, day_talks in days:
        ds = ts._load_day_summary(date) or {}
        ov = (ds.get("overview") or "").strip()
        themes = ", ".join(ds.get("themes", []))
        parts.append(f"\n[{day_label} — {date} — {len(day_talks)} talks]")
        if ov:
            parts.append(ov)
        if themes:
            parts.append(f"Themes: {themes}")

    # Conference-wide hot-topics synthesis.
    topics = ts._load_topics_summary() or {}
    if topics.get("topics"):
        parts.append("\n=== HOT TOPICS (conference-wide synthesis, most-discussed first) ===")
        ranked = sorted(topics["topics"], key=lambda x: len(x.get("talks", [])), reverse=True)
        for tp in ranked:
            parts.append(f"\n- {tp['name']} ({len(tp.get('talks', []))} talks): "
                         f"{tp.get('description', '')}")

    # On-box concept/tool tallies with per-source mention counts, for the
    # topics-and-tools occurrence table alan is asked to produce. Restricted to
    # the approved talks so unapproved speakers never contribute a count.
    allowed_folders = {t["folder"] for t in ts._load_talks()}
    topic_occ, tool_occ = _load_occurrences(allowed_folders)
    if topic_occ or tool_occ:
        parts.append("\n=== OCCURRENCE COUNTS (for the topics-and-tools table; "
                     "transcript vs slide mentions, most-mentioned first) ===")
        if topic_occ:
            parts.append("\nTOPICS/CONCEPTS  (name | transcript mentions | slide mentions | talks)")
            for o in topic_occ:
                parts.append(f"  {o['name']} | {o['transcript']} | {o['slide']} | {o['talks']}")
        if tool_occ:
            parts.append("\nTOOLS/SOFTWARE  (name | transcript mentions | slide mentions | talks)")
            for o in tool_occ:
                parts.append(f"  {o['name']} | {o['transcript']} | {o['slide']} | {o['talks']}")

    # Per-talk detail, budgeted.
    parts.append("\n=== TALKS (chronological) ===")
    spent = 0
    for d in digests:
        who = d["speaker"] + (f" ({d['affiliation']})" if d["affiliation"] else "")
        line = f"\n[{d['date']}] {who} — \"{d['title']}\""
        if d["abstract"] and spent < budget:
            line += f"\n  Abstract: {d['abstract']}"
            spent += len(d["abstract"])
            if d["key_points"]:
                line += "\n  Key points: " + "; ".join(d["key_points"])
            if d["topics"]:
                line += "\n  Topics: " + ", ".join(d["topics"])
        parts.append(line)
    if spent >= budget:
        parts.append("\n(Note: some later talks listed by title only to stay within the "
                     "context budget; their themes are captured in the overviews above.)")
    return "\n".join(parts)


def load_prev_booklet(path: str | None, budget: int = 60000) -> str:
    """Return text of last year's abstract booklet, or '' if unavailable."""
    candidates = []
    if path:
        candidates.append(Path(path))
    if os.environ.get("SBI_PREV_BOOKLET"):
        candidates.append(Path(os.environ["SBI_PREV_BOOKLET"]))
    candidates += [ROOT / "prev_year_booklet.txt", ROOT / "prev_year_booklet.pdf"]
    # Auto-detect a dropped-in booklet by its usual filename, newest first.
    for pat in ("*abstract*booklet*.pdf", "*booklet*.pdf", "*abstract*booklet*.txt"):
        candidates += sorted(ROOT.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)

    for p in candidates:
        if not p or not p.exists():
            continue
        try:
            if p.suffix.lower() == ".pdf":
                text = _pdf_to_text(p)
            else:
                text = p.read_text(errors="ignore")
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if text:
                print(f"  previous-year booklet: {p}  ({len(text)} chars)")
                return text[:budget]
        except Exception as e:
            print(f"  ! could not read {p}: {e}", file=sys.stderr)
    print("  previous-year booklet: none found — comparison will be flagged as indicative.")
    return ""


def _pdf_to_text(p: Path) -> str:
    if subprocess.run(["which", "pdftotext"], capture_output=True).returncode == 0:
        out = subprocess.run(["pdftotext", "-layout", str(p), "-"],
                             capture_output=True, text=True, timeout=120)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout
    from pdfminer.high_level import extract_text  # type: ignore
    return extract_text(str(p))


# ── prompt + generation ──────────────────────────────────────────────────────

SYSTEM = (
    f"You are {ASSISTANT}, the rapporteur for a scientific conference, writing the official "
    "summary white paper. You are given distilled context: per-day overviews, a "
    "conference-wide hot-topics synthesis, and per-talk abstracts/key points — and "
    "possibly last year's abstract booklet. Write a faithful, well-structured, "
    "publication-quality white paper. Ground every claim in the provided context; "
    "do NOT invent talks, names, results or numbers. Be specific: name representative "
    "speakers/talks when illustrating a theme. Write in measured scientific prose.\n\n"
    "Output ONLY a complete, self-contained LaTeX document — no markdown, no code "
    "fences, no commentary before or after. Requirements for guaranteed compilation:\n"
    "  * \\documentclass[11pt]{article}\n"
    "  * Use ONLY these packages: geometry, parskip, enumitem, hyperref, titlesec, "
    "booktabs, longtable.\n"
    "  * \\usepackage[margin=1in]{geometry}; keep it within the page budget.\n"
    f"  * Provide \\title/\\author{{{ASSISTANT} (automated rapporteur)}}/\\date and \\maketitle.\n"
    "  * Escape LaTeX special characters in any prose: % & _ # $ -> \\% \\& \\_ \\# \\$.\n"
    "  * Use \\section / \\subsection and itemize/enumerate; no figures, no \\cite, no "
    "bibliography, no custom macros, no \\input.\n"
    "  * Tables are allowed and required (see below): use tabular or longtable with "
    "booktabs rules (\\toprule/\\midrule/\\bottomrule); escape special characters in cells.\n"
    "Required sections, in order:\n"
    "  1. A short abstract (\\begin{abstract}) framing the meeting.\n"
    "  2. Overview of topics discussed.\n"
    "  3. Hot subjects (what dominated, what was emerging).\n"
    "  4. A table of the main topics and tools with their occurrence counts (see "
    "TABLE below).\n"
    "  5. Comparison with the previous year.\n"
    "  6. Outlook for the future.\n"
    "SPEAKER COVERAGE (important): every speaker listed under '=== TALKS ===' in the "
    "context has been approved for inclusion in this write-up. Mention EVERY ONE of them "
    "by name at least once in the prose — weave them into the topic they spoke on. Do not "
    "silently drop any speaker; if a talk fits only a minor theme, still name its speaker "
    "in a grouped sentence. Use ONLY the speakers present in the context — never invent "
    "names or attribute talks to people not listed.\n"
    "TABLE (required): include one table titled to the effect of \"Main topics and tools "
    "and their occurrence counts\". Build it STRICTLY from the '=== OCCURRENCE COUNTS ===' "
    "block in the context, which gives, per topic/concept and per tool, the number of "
    "mentions in the transcripts and in the slides. Give it columns: Topic/Tool, "
    "Transcript mentions, Slide mentions (a Talks column is optional). List the main "
    "entries most-mentioned first; you may group the long tail of low-count entries but do "
    "not fabricate any counts — copy the numbers verbatim from the context. If the "
    "OCCURRENCE COUNTS block is absent, omit the table rather than invent one.\n"
    "COVERAGE POLICY (important): aim to mention EVERY topic/theme discussed at the "
    "conference — do not silently drop any. Allocate space in PROPORTION to how much "
    "each topic was discussed: the '=== HOT TOPICS ===' list in the context is ordered "
    "most-discussed-first and states the number of talks per topic — use those counts "
    "as the weighting. Dominant, recurring topics earn dedicated paragraphs naming "
    "several representative speakers; mid-weight topics get a few sentences; niche or "
    "single-talk topics get at least a clause or a grouped mention. Be proportional in "
    "DEPTH, comprehensive in BREADTH.\n"
    "Keep the whole paper to AT MOST {pages} pages: stay within budget by compressing "
    "minor topics into grouped sentences, never by omitting whole themes. This is a "
    "weighted synthesis, not minutes."
)

PREV_PRESENT = (
    "Last year's abstract booklet text is provided below under === PREVIOUS YEAR "
    "BOOKLET ===. Base the comparison on concrete shifts in topics, methods and "
    "emphasis between it and this year's programme."
)
PREV_ABSENT = (
    "No previous-year abstract booklet was provided. Still write the comparison "
    "section, but base it only on the general trajectory of the field and clearly "
    "state, in the text, that it is indicative and not grounded in last year's "
    "booklet."
)


# ── planning ("thinking") pass ───────────────────────────────────────────────
# alan (gemma) is not a native reasoning model, so we buy it extra "thinking time"
# by spending a separate call to reason over the context and draft an editorial
# plan BEFORE writing any LaTeX. The plan inventories every topic, weights it by
# how often it was discussed, and decides proportional space — which is then fed
# into the write pass so coverage is complete and depth tracks prominence.
PLAN_SYSTEM = (
    f"You are {ASSISTANT}, planning a conference summary white paper BEFORE writing it. "
    "Think carefully and produce a structured editorial PLAN — NOT the paper, and NOT "
    "LaTeX. From the provided context (per-day overviews, the conference-wide "
    "hot-topics synthesis which states how many talks engaged each topic, and per-talk "
    "abstracts/topic tags):\n"
    "  1. Build a COMPLETE inventory of every distinct topic/theme discussed. Give each "
    "an approximate weight = number of talks that engaged it (use the hot-topics counts "
    "and per-talk topic tags). Rank most- to least-discussed.\n"
    "  2. For each topic, recommend how much space it should get in the paper, "
    "PROPORTIONAL to its weight: a dedicated paragraph for the heavyweight topics, a "
    "shared/grouped sentence for niche ones — but every topic gets at least a mention so "
    "nothing is dropped. List 1-3 representative speakers per topic.\n"
    "  3. Identify the 2-4 genuinely dominant 'hot subjects', the emerging ones, and "
    "(if a previous-year booklet is provided) the clearest year-on-year shifts to "
    "foreground in the comparison.\n"
    "  4. List EVERY speaker in the '=== TALKS ===' block (they are all approved for "
    "inclusion) and note, for each, the topic under which the paper should name them, so "
    "the write pass can mention every one of them.\n"
    "  5. Note that the paper must contain a table of the main topics and tools with their "
    "transcript- and slide-mention counts, taken verbatim from the '=== OCCURRENCE COUNTS "
    "==='  block; identify which entries are the main ones and which form the low-count "
    "tail that can be grouped.\n"
    "Output a concise plain-text outline. Ground everything strictly in the context; do "
    "not invent topics, names, results or counts."
)


def write_plan(context: str, prev_text: str, max_tokens: int, timeout: int) -> str:
    user = [f"Plan the {CONF_SHORT} summary white paper.",
            ("A previous-year booklet is also provided below; note the shifts worth "
             "foregrounding." if prev_text else ""),
            "\n=== THIS YEAR'S MEETING CONTEXT ===\n" + context]
    if prev_text:
        user.append("\n=== PREVIOUS YEAR BOOKLET ===\n" + prev_text)
    msgs = [{"role": "system", "content": PLAN_SYSTEM},
            {"role": "user", "content": "\n".join(p for p in user if p)}]
    return ts._llm_chat(msgs, max_tokens=max_tokens, temperature=0.2, timeout=timeout)


def build_messages(context: str, prev_text: str, pages: int, plan: str = ""):
    sys_prompt = SYSTEM.replace("{pages}", str(pages))
    user = [f"Write the {CONF_SHORT} summary white paper (max {pages} pages).",
            PREV_PRESENT if prev_text else PREV_ABSENT]
    if plan:
        user.append("\n=== YOUR EDITORIAL PLAN (follow it: cover every topic listed and "
                    "allocate space in proportion to each topic's weight) ===\n" + plan)
    user.append("\n=== THIS YEAR'S MEETING CONTEXT ===\n" + context)
    if prev_text:
        user.append("\n=== PREVIOUS YEAR BOOKLET ===\n" + prev_text)
    return [{"role": "system", "content": sys_prompt},
            {"role": "user", "content": "\n".join(user)}]


def clean_latex(raw: str) -> str:
    txt = raw.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-zA-Z]*\s*", "", txt)
        txt = re.sub(r"\s*```$", "", txt).strip()
    i = txt.find("\\documentclass")
    if i > 0:
        txt = txt[i:]
    if "\\end{document}" in txt:
        txt = txt[: txt.rindex("\\end{document}") + len("\\end{document}")]
    header = (f"% {CONF_SHORT} summary white paper — generated by whitepaper.py\n"
              f"% Written locally by {ts.LLM_MODEL} ({ASSISTANT}); context never sent to external tools.\n"
              f"% Generated {datetime.now().isoformat(timespec='seconds')}\n")
    return header + txt + "\n"


def compile_pdf(tex_path: Path) -> bool:
    workdir = tex_path.parent
    have = lambda c: subprocess.run(["which", c], capture_output=True).returncode == 0
    try:
        if have("latexmk"):
            r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode",
                                "-halt-on-error", tex_path.name],
                               cwd=workdir, capture_output=True, text=True, timeout=300)
        elif have("pdflatex"):
            for _ in range(2):
                r = subprocess.run(["pdflatex", "-interaction=nonstopmode",
                                    "-halt-on-error", tex_path.name],
                                   cwd=workdir, capture_output=True, text=True, timeout=300)
        else:
            print("  ! no latexmk/pdflatex on PATH — skipping PDF compile", file=sys.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("  ! LaTeX compile timed out", file=sys.stderr)
        return False
    pdf = tex_path.with_suffix(".pdf")
    if pdf.exists():
        return True
    tail = "\n".join((r.stdout or "").splitlines()[-25:])
    print(f"  ! LaTeX compile failed; see {tex_path.with_suffix('.log')}\n{tail}",
          file=sys.stderr)
    return False


# ── cli ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=f"Generate the {CONF_SHORT} summary white paper via {ASSISTANT}.")
    ap.add_argument("--prev-booklet", help="last year's abstract booklet (.pdf or .txt)")
    ap.add_argument("--out", default="whitepaper.tex", help="output .tex path")
    ap.add_argument("--pdf", action="store_true", help="also compile to PDF")
    ap.add_argument("--refresh", action="store_true",
                    help="rebuild day overviews + topic synthesis first")
    ap.add_argument("--pages", type=int, default=3, help="target page budget (default 3)")
    ap.add_argument("--budget", type=int, default=90000,
                    help="max chars of per-talk abstract context (default 90000)")
    ap.add_argument("--dump-context", help="also write the assembled context to this file")
    ap.add_argument("--model", help="override ALAN_LLM_MODEL for this run")
    ap.add_argument("--no-think", dest="think", action="store_false",
                    help="skip the editorial planning pass (less 'thinking time')")
    ap.add_argument("--max-tokens", type=int, default=9000,
                    help="output token budget for the write pass (default 9000)")
    ap.add_argument("--timeout", type=int, default=900,
                    help="per-LLM-call timeout in seconds (default 900)")
    ap.add_argument("--plan-tokens", type=int, default=2500,
                    help="output token budget for the planning pass (default 2500)")
    ap.add_argument("--dump-plan", help="also write the editorial plan to this file")
    args = ap.parse_args()

    if args.model:
        ts.LLM_MODEL = args.model

    # Fail-closed permission gate: only speakers who gave explicit permission
    # (listed in public_talks.txt) are ever seen. This monkeypatches
    # ts._load_talks, so EVERY downstream consumer — context, day/topic caches
    # and the occurrence table — is restricted to approved talks. A missing or
    # empty allowlist aborts rather than leaking an unapproved speaker.
    es.apply_allowlist(verbose=True)
    talks = ts._load_talks()
    print(f"Permission allowlist: {len(talks)} approved talk(s) will be included.")

    if args.refresh:
        # Force day/topic syntheses to rebuild over the APPROVED set only (the
        # subprocess backfills would rebuild over every talk on disk, so we drop
        # the caches and let ensure_caches regenerate them from the filtered talks).
        print("Refreshing day overviews + topic synthesis (approved talks only) ...")
        for p in ts.DAYS_DIR.glob("*.json"):
            p.unlink()
        if ts.TOPICS_PATH.exists():
            ts.TOPICS_PATH.unlink()
    # Rebuild any stale day/topic caches over the approved set. The talk-set
    # signature changes when the allowlist excludes talks, so caches previously
    # built over all talks regenerate here rather than leak unapproved content.
    es.ensure_caches(talks, use_llm=True, strict=False, verbose=True)

    print(f"Assembling meeting context (via {ts.LLM_MODEL} at {ts.LLM_URL}) ...")
    context = gather_context(args.budget)
    prev_text = load_prev_booklet(args.prev_booklet)
    if args.dump_context:
        Path(args.dump_context).write_text(context)
        print(f"  context dumped to {args.dump_context}  ({len(context)} chars)")

    plan = ""
    if args.think:
        print("Thinking: drafting an editorial plan (topic inventory + proportional "
              "space) ...", flush=True)
        tp = time.time()
        plan = write_plan(context, prev_text, args.plan_tokens, args.timeout)
        print(f"  ✓ {time.time() - tp:5.1f}s  plan ready  ({len(plan)} chars)")
        if args.dump_plan:
            Path(args.dump_plan).write_text(plan)
            print(f"  plan dumped to {args.dump_plan}")

    print(f"Writing white paper (<= {args.pages} pages) ...", flush=True)
    t0 = time.time()
    raw = ts._llm_chat(build_messages(context, prev_text, args.pages, plan),
                       max_tokens=args.max_tokens, temperature=0.35, timeout=args.timeout)
    tex = clean_latex(raw)
    out = Path(args.out)
    out.write_text(tex)
    print(f"  ✓ {time.time() - t0:5.1f}s  wrote {out}  ({len(tex)} chars)")

    if args.pdf:
        print("Compiling PDF ...")
        if compile_pdf(out):
            print(f"  ✓ {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
