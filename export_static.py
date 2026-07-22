#!/usr/bin/env python3
"""Export the SBI4GALEV archive as a static site for GitHub Pages.

Renders the same pages the live server serves at /summaries, /summaries/<date>
and /topics into a self-contained `site/` directory of plain HTML + the slide
thumbnails they reference, with all server-only behaviour (absolute links, the
live SSE viewer, audience voting, the topics generation timer) stripped out.

The live transcript viewer (/, /admin, /stream) and voting (/vote) are dynamic
and are intentionally NOT exported.

This reuses transcript_server's render functions and summary generators, so it
must run on the machine that has the saved talks (transcripts/) and, for fresh
day overviews / key topics, the local LLM. Nothing in transcript_server.py is
modified — the static-specific transforms are post-processing here.

Output layout (flat, relative links → works under any base path, incl.
https://<owner>.github.io/<repo>/):
    site/
      index.html            (landing page: conference + tool intro, links below)
      summaries.html        (= /summaries, the day-by-day archive)
      topics.html           (= /topics)
      day-YYYY-MM-DD.html   (= /summaries/<date>, one per day)
      talks/<folder>/slides/slide_*.jpg     (every slide — drives the lightbox)
      .nojekyll

Usage:
    python3 export_static.py [--out DIR] [--clean] [--no-llm]
                            [--all-slides] [--include-transcripts] [--strict] [-v]

Deploy: the built site/ is git-ignored and lives on its own `gh-pages` branch,
which GitHub Pages serves directly. The publish_site.sh helper builds and pushes
it in one step:
    ./publish_site.sh        # = export_static.py --clean, then push site/ -> gh-pages
One-time: repo Settings -> Pages -> Source: Deploy from a branch -> gh-pages /
(root). publish_site.sh force-pushes a single fresh commit, so the branch only
ever holds the latest site; GitHub rebuilds Pages on each push (no workflow).
"""
import argparse
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote, unquote

import config
import transcript_server as ts


def _warn(msg):
    print(f"  ! {msg}", file=sys.stderr, flush=True)


def _load_index(name):
    """Load a cached index JSON (concepts.json / tools.json) from the talk store,
    or None if absent/unreadable."""
    p = ts.SAVE_DIR / name
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ── Permissions allowlist (fail-closed) ──────────────────────────────────────

ALLOWLIST_FILE = Path(__file__).parent / "public_talks.txt"


def load_allowlist(path: Path = ALLOWLIST_FILE):
    """Read the approved-talk folder names from public_talks.txt.

    Returns a set of folder names, or None if the file is absent (caller decides
    what that means). Blank lines and # comments are ignored; an inline '#'
    trailing a folder name is stripped too."""
    if not path.exists():
        return None
    names = set()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(line)
    return names


def apply_allowlist(verbose):
    """Wrap ts._load_talks so EVERY consumer (render functions, cache
    generation) only ever sees talks approved for public export. Fail-closed:
    a missing allowlist exports nothing."""
    allowed = load_allowlist()
    if allowed is None:
        raise SystemExit(
            f"No allowlist at {ALLOWLIST_FILE.name} — refusing to export (fail-closed). "
            f"Create it with one approved folder name per line.")
    if not allowed:
        raise SystemExit(
            f"{ALLOWLIST_FILE.name} lists no approved talks — nothing to export (fail-closed).")

    _orig_load = ts._load_talks

    def _ok(folder):
        # Match an allowlist entry against the full folder name OR its readable
        # slug (the part after the YYYY-MM-DD_HH-MM-SS_ timestamp prefix).
        slug = re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_", "", folder)
        return folder in allowed or slug in allowed

    def _filtered():
        talks = _orig_load()
        kept = [t for t in talks if _ok(t["folder"])]
        if verbose:
            print(f"  allowlist: keeping {len(kept)}/{len(talks)} talks; "
                  f"excluding {len(talks) - len(kept)}", flush=True)
        matched = set()
        for t in talks:
            slug = re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_", "", t["folder"])
            matched |= ({t["folder"], slug} & allowed)
        missing = allowed - matched
        if missing:
            _warn(f"allowlist names {len(missing)} entr(y/ies) with no talk on disk: "
                  f"{', '.join(sorted(missing))}")
        return kept

    ts._load_talks = _filtered


# ── Cache freshness (run before rendering, so pages render fully not "pending") ─

def ensure_caches(talks, use_llm, strict, verbose):
    """Make sure the per-day overviews and key-topics synthesis are present and
    fresh for the current talks, generating any that are stale. Per-talk
    summaries are only checked (generate those with backfill_summaries.py)."""
    missing = [t["folder"] for t in talks if not t["summary"]]
    if missing:
        _warn(f"{len(missing)} talk(s) have no summary.json — run backfill_summaries.py "
              f"first (e.g. {missing[0]})")
        if strict:
            raise SystemExit("strict: refusing to export with unsummarised talks")

    # Per-day overviews (LLM-only: no deterministic fallback).
    for date, label, dtalks in ts._group_by_day(talks):
        sig = ts._day_signature(dtalks)
        cached = ts._load_day_summary(date)
        if cached and cached.get("signature") == sig and cached.get("overview"):
            if verbose:
                print(f"  · day {date} overview up to date", flush=True)
            continue
        if not use_llm:
            _warn(f"day {date} overview is stale/missing and --no-llm — page will show "
                  f"a placeholder")
            if strict:
                raise SystemExit(f"strict: day {date} overview unavailable")
            continue
        print(f"  → generating day {date} overview …", flush=True)
        try:
            data = ts._generate_day_summary(date, label, dtalks)
            data["signature"] = sig
            ts._atomic_write_bytes(ts.DAYS_DIR / f"{date}.json",
                                   json.dumps(data, indent=2).encode())
        except Exception as e:
            _warn(f"day {date} overview generation failed: {e}")
            if strict:
                raise

    # Key topics (LLM with a built-in keyword fallback).
    sig = ts._topics_signature(talks)
    cached = ts._load_topics_summary()
    if cached and cached.get("signature") == sig and "topics" in cached:
        if verbose:
            print("  · key topics up to date", flush=True)
    else:
        print("  → generating key topics …", flush=True)
        try:
            data = ts._generate_topics_summary(talks, use_llm)
            data["signature"] = sig
            ts._atomic_write_bytes(ts.TOPICS_PATH, json.dumps(data, indent=2).encode())
            print(f"    {len(data['topics'])} topics, {len(data['edges'])} links "
                  f"({data['source']})", flush=True)
        except Exception as e:
            _warn(f"key topics generation failed: {e}")
            if strict:
                raise


# ── HTML post-processing: server URLs → static, voting → read-only ─────────────

def staticize(s: str) -> str:
    """Rewrite the server-rendered HTML into a standalone static page."""
    # 1. Dated day links (with optional #anchor) — BEFORE the bare /summaries rule.
    s = re.sub(r'href="/summaries/(\d{4}-\d{2}-\d{2})(#[^"]*)?"',
               lambda m: f'href="day-{m.group(1)}.html{m.group(2) or ""}"', s)
    # 2. Summaries index (anchor fallback first, then bare).
    s = s.replace('href="/summaries#', 'href="summaries.html#')
    s = s.replace('href="/summaries"', 'href="summaries.html"')
    # 3. Topics page.
    s = s.replace('href="/topics"', 'href="topics.html"')
    # 4. Slide assets → relative: src=, href= AND the lightbox data-slides JSON
    #    (["/talks/…", …]). Matching the leading quote covers all three.
    s = s.replace('"/talks/', '"talks/')
    # 5. Drop the live-viewer nav link (not part of the static archive).
    s = re.sub(r'<a href="/">[^<]*</a>', '', s)
    # 6. Voting → read-only: strip the vote buttons and remove the voting JS.
    s = re.sub(r'<span class="qvote">.*?</span>', '', s, flags=re.S)
    s = s.replace(ts.SUMMARIES_JS, "")
    return s


# ── Landing page (static archive only — the live server has no front door) ─────

LANDING_CSS = """
    .hero { padding:8px 0 4px; }
    .hero h2 { font-size:1.9em; font-weight:700; color:#fff; line-height:1.2;
        letter-spacing:0.3px; margin-bottom:0.15em; }
    .hero .hero-kicker { color:#9ec9b0; font-size:1.2em; font-weight:600;
        letter-spacing:0.4px; margin-bottom:0.6em; }
    .hero .hero-meta { color:#6a8; font-size:0.8em; letter-spacing:1px;
        text-transform:uppercase; margin-bottom:1.1em; }
    .hero .hero-lead { color:#e6e6e6; line-height:1.7; font-size:1.12em;
        margin-bottom:0.9em; }
    .hero .hero-lead b { color:#fff; }
    .hero .hero-desc { color:#bdbdbd; line-height:1.7; font-size:0.98em; }
    .hero .conf-link { margin-top:1.1em; }
    .hero .conf-link a { color:#9ec9b0; font-size:1.0em; text-decoration:none;
        border:1px solid #3a5a48; border-radius:8px; padding:8px 16px; display:inline-block; }
    .hero .conf-link a:hover { background:#101410; color:#fff; }
    .about { margin:2.4em 0 0.6em; padding-top:1.8em; border-top:1px solid #1a1a1a; }
    .about h3 { font-size:0.78em; font-weight:600; color:#777; letter-spacing:1.5px;
        text-transform:uppercase; margin-bottom:0.7em; }
    .about p { color:#bbb; line-height:1.75; }
    .repo-link { margin-top:1em; font-size:0.92em; }
    .repo-link a { color:#9ec9b0; text-decoration:none; }
    .repo-link a:hover { color:#fff; text-decoration:underline; }
    .landing-cards { display:grid; grid-template-columns:1fr 1fr; gap:16px;
        margin:2.4em 0 1.2em; }
    @media (max-width:560px) { .landing-cards { grid-template-columns:1fr; } }
    .lcard { display:block; padding:22px 22px 24px; border:1px solid #1f1f1f;
        border-radius:10px; background:#0e0e0e; text-decoration:none;
        transition:border-color 0.15s, background 0.15s; }
    .lcard:hover { border-color:#3a5a48; background:#101410; }
    .lcard .lcard-h { display:block; color:#fff; font-size:1.12em; font-weight:600;
        margin-bottom:0.4em; }
    .lcard:hover .lcard-h { color:#9ec9b0; }
    .lcard .lcard-d { display:block; color:#999; font-size:0.88em; line-height:1.6; }
    .byline { color:#555; font-size:0.78em; margin-top:1.8em; }
"""


def render_landing(talks, grouped, have_ct=False, have_wp=False) -> str:
    """Build the static archive's front door: conference title + description,
    a note on the ambient-AI tool that produced it, and cards linking to the
    day summaries, key-topics (and concepts/tools) pages. Static-only (the live
    server opens straight onto /summaries), so it's assembled here."""
    esc = html.escape
    n_talks, n_days = len(talks), len(grouped)
    if grouped:
        first, last = grouped[0][0], grouped[-1][0]
        when = (ts._format_day_date(first) if first == last
                else f"{ts._format_day_date(first)} – {ts._format_day_date(last)}")
        meta = f"{when} · {n_talks} talk{'s' if n_talks != 1 else ''} over {n_days} day{'s' if n_days != 1 else ''}"
    else:
        meta = config.CONF_YEAR

    conf_url = config.SITE_CONFERENCE_URL
    conf_link = (f'<p class="conf-link"><a href="{esc(conf_url)}">Main conference website ↗</a></p>'
                 if conf_url else "")

    ct_cards = ("""
      <a class="lcard" href="methods.html">
        <span class="lcard-h">Methods →</span>
        <span class="lcard-d">Methods and terms (NPE, NLE, evidence…) with mention counts and the talks that covered them.</span>
      </a>
      <a class="lcard" href="tools.html">
        <span class="lcard-h">Tools &amp; software →</span>
        <span class="lcard-d">Codes, packages and simulation suites used across the meeting, with links and the talks that used them.</span>
      </a>""" if have_ct else "")

    wp_card = ("""
      <a class="lcard" href="whitepaper.html">
        <span class="lcard-h">White paper →</span>
        <span class="lcard-d">The AI-generated summary white paper — read it in the browser or download the typeset PDF.</span>
      </a>""" if have_wp else "")

    body = f"""<style>{LANDING_CSS}</style>
    <section class="hero">
      <h2>{esc(config.CONF_FULL)}</h2>
      <div class="hero-kicker">Ambient AI summary</div>
      <div class="hero-meta">{esc(meta)}</div>
      <p class="hero-lead">The official <b>ambient-AI summary</b> of the meeting —
        talk summaries, slides and key topics captured live in the room and written
        up automatically. For the full programme, schedule and details, head to the
        main conference website.</p>
      <p class="hero-desc">{esc(config.SITE_DESCRIPTION)}</p>
      {conf_link}
    </section>
    <section class="about">
      <h3>About this archive</h3>
      <p>{esc(config.SITE_TOOL_BLURB)}</p>
      <p class="repo-link">The toolkit is open source:
        <a href="{esc(config.SITE_REPO_URL)}">{esc(config.SITE_REPO_URL.replace("https://", ""))}</a></p>
    </section>
    <nav class="landing-cards">
      <a class="lcard" href="summaries.html">
        <span class="lcard-h">Day summaries →</span>
        <span class="lcard-d">An editorial overview of each day, talk by talk, with slides and key points.</span>
      </a>
      <a class="lcard" href="topics.html">
        <span class="lcard-h">Key topics →</span>
        <span class="lcard-d">The themes that ran across the meeting, linked together as a topic map.</span>
      </a>{ct_cards}{wp_card}
    </nav>
    <p class="byline">Summaries written automatically by a local, on-device AI assistant.</p>"""

    nav = '<a href="summaries.html">day summaries</a><a href="topics.html">key topics</a>'
    if have_ct:
        nav += '<a href="methods.html">methods</a><a href="tools.html">tools</a>'
    if have_wp:
        nav += '<a href="whitepaper.html">white paper</a>'
    if conf_url:
        nav += f'<a href="{esc(conf_url)}">conference ↗</a>'
    heading = f"{esc(config.CONF_SHORT)} {esc(config.CONF_YEAR)}"
    return ts._page_shell(f"{config.CONF_SHORT} {config.CONF_YEAR} — Ambient AI summary",
                          heading, nav, body)


def add_nav(htmlstr: str, frag: str) -> str:
    """Prepend extra nav links (home, and concepts/tools when present) to a
    server-rendered page's nav bar."""
    return htmlstr.replace('<div class="navlinks">', f'<div class="navlinks">{frag}', 1)


CONCEPTS_CSS = """
    .lede { color:#9a9a9a; line-height:1.6; margin:0 0 1.4em; max-width:66ch; }
    .ci { padding:20px 0; border-top:1px solid #1a1a1a; }
    .ci h2 { font-size:1.15em; color:#fff; font-weight:600; display:flex;
        align-items:baseline; gap:10px; flex-wrap:wrap; }
    .ci h2 .abbr { color:#9ec9b0; font-size:0.7em; letter-spacing:1px; }
    .ci h2 a { color:#9ec9b0; text-decoration:none; }
    .ci h2 a:hover { color:#fff; }
    .ci-meta { color:#777; font-size:0.78em; letter-spacing:0.5px; margin:0.35em 0 0.55em; }
    .ci-meta b { color:#9ec9b0; font-weight:600; }
    .ci-def { color:#cccccc; line-height:1.65; margin-bottom:0.6em; max-width:72ch; }
    .ci-talks { font-size:0.88em; color:#777; line-height:1.9; }
    .ci-talks a { color:#cdd; text-decoration:none; }
    .ci-talks a:hover { color:#fff; }
"""


def _concept_talks_html(talks):
    esc = html.escape
    return " · ".join(
        f'<a href="/summaries/{quote(t["date"])}#{esc(t["folder"])}">{esc(t["title"])}</a>'
        for t in talks)


def render_concepts(data, nav):
    esc = html.escape
    rows = []
    for c in data["concepts"]:
        abbr = (f'<span class="abbr">{esc(c["abbr"])}</span>'
                if c.get("abbr") and c["abbr"] != c["name"] else "")
        rows.append(
            f'<section class="ci"><h2>{esc(c["name"])} {abbr}</h2>'
            f'<div class="ci-meta"><b>{c["transcript_mentions"]}</b> transcript · '
            f'<b>{c["slide_mentions"]}</b> slide mentions · {len(c["talks"])} talk(s)</div>'
            f'<p class="ci-def">{esc(c.get("definition",""))}</p>'
            f'<div class="ci-talks">{_concept_talks_html(c["talks"])}</div></section>')
    lede = (f'The methods and techniques used across {esc(config.CONF_SHORT)}, with how '
            'often each was mentioned in talk transcripts and slides, and the talks that '
            'covered it. Extracted and counted on-box.')
    body = f'<style>{CONCEPTS_CSS}</style><p class="lede">{lede}</p>{"".join(rows)}'
    return ts._page_shell(f"{config.CONF_SHORT} — Methods", "Methods", nav, body)


def render_tools(data, nav):
    esc = html.escape
    rows = []
    for t in data["tools"]:
        url = t.get("url", "")
        name = (f'<a href="{esc(url)}">{esc(t["name"])} ↗</a>' if url else esc(t["name"]))
        rows.append(
            f'<section class="ci"><h2>{name}</h2>'
            f'<div class="ci-meta"><b>{t["transcript_mentions"]}</b> transcript · '
            f'<b>{t["slide_mentions"]}</b> slide mentions · {len(t["talks"])} talk(s)</div>'
            f'<p class="ci-def">{esc(t.get("description",""))}</p>'
            f'<div class="ci-talks">{_concept_talks_html(t["talks"])}</div></section>')
    lede = (f'Software, codes, frameworks and simulation suites used across '
            f'{esc(config.CONF_SHORT)}. Links point to each project\'s page where known '
            '(looked up on-box). Counts are mentions across talk transcripts and slides.')
    body = f'<style>{CONCEPTS_CSS}</style><p class="lede">{lede}</p>{"".join(rows)}'
    return ts._page_shell(f"{config.CONF_SHORT} — Tools", "Tools &amp; software", nav, body)


# ── White paper (LaTeX summary → in-browser HTML + PDF download) ───────────────

WHITEPAPER_TEX = Path(__file__).parent / "whitepaper.tex"
WHITEPAPER_PDF = Path(__file__).parent / "whitepaper.pdf"

_MONTHS = ("", "January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December")


def _pandoc_body(tex_path: Path) -> str | None:
    """Convert the white-paper LaTeX to an HTML fragment via pandoc (standalone,
    so the title block + abstract survive), returning the inner <body> content
    (title, abstract, sections, the occurrence table). Sections are shifted to
    <h2>/<h3> so the paper title is the page's only <h1>. Returns None if pandoc
    is unavailable or the conversion fails — the caller then skips the page."""
    try:
        r = subprocess.run(
            ["pandoc", "-s", "--shift-heading-level-by=1", "-f", "latex", "-t", "html",
             str(tex_path)],
            capture_output=True, text=True, timeout=60)
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as e:
        _warn(f"white paper: pandoc unavailable ({e}) — skipping the in-browser page")
        return None
    if r.returncode != 0 or not r.stdout.strip():
        _warn(f"white paper: pandoc failed — skipping the in-browser page\n{r.stderr[-400:]}")
        return None
    m = re.search(r"<body>(.*)</body>", r.stdout, re.S)
    return m.group(1).strip() if m else None


def _whitepaper_generated(tex_path: Path) -> str:
    """Human-readable generation date parsed from whitepaper.py's '% Generated
    YYYY-MM-DD…' header comment, or '' if not present."""
    try:
        head = tex_path.read_text()[:400]
    except Exception:
        return ""
    m = re.search(r"% Generated (\d{4})-(\d{2})-(\d{2})", head)
    if not m:
        return ""
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return f"{d} {_MONTHS[mo]} {y}" if 1 <= mo <= 12 else f"{y}-{m.group(2)}-{m.group(3)}"


WHITEPAPER_CSS = """
    .wp-intro { color:#9a9a9a; line-height:1.7; margin:0 0 1.3em; max-width:72ch; }
    .wp-actions { display:flex; flex-wrap:wrap; align-items:center; gap:16px; margin:0 0 0.4em; }
    .wp-dl { display:inline-flex; align-items:center; gap:9px; color:#9ec9b0; font-size:0.98em;
        text-decoration:none; border:1px solid #3a5a48; border-radius:8px; padding:10px 18px; }
    .wp-dl:hover { background:#101410; color:#fff; }
    .wp-dl .arr { font-size:1.05em; line-height:1; }
    .wp-meta { color:#5c5c5c; font-size:0.78em; letter-spacing:0.5px; }
    .paper { margin-top:2.2em; padding-top:1.9em; border-top:1px solid #1a1a1a; }
    .paper #title-block-header { margin-bottom:1.7em; }
    .paper #title-block-header .title { font-size:1.85em; font-weight:700; color:#fff;
        line-height:1.25; letter-spacing:0.2px; margin:0 0 0.3em; padding:0; border:none; }
    .paper .author { color:#9a9; font-size:0.95em; margin:0; }
    .paper .date { color:#666; font-size:0.75em; letter-spacing:1.5px; text-transform:uppercase;
        margin:0.35em 0 0; }
    .paper .abstract { margin:1.5em 0 0; padding:1.15em 1.35em; background:#0e0e0e;
        border:1px solid #1f1f1f; border-radius:9px; }
    .paper .abstract-title { color:#9ec9b0; font-size:0.72em; font-weight:600; letter-spacing:1.5px;
        text-transform:uppercase; margin-bottom:0.55em; }
    .paper .abstract p { color:#cccccc; line-height:1.7; margin:0; }
    .paper h2 { font-size:1.22em; font-weight:700; color:#fff; letter-spacing:0.2px;
        margin:1.9em 0 0.55em; padding-top:1.3em; border-top:1px solid #161616; }
    .paper h3 { font-size:1.05em; font-weight:600; color:#fff; margin:1.4em 0 0.4em; }
    .paper p { color:#dcdcdc; line-height:1.78; margin:0 0 1em; }
    .paper strong { color:#fff; font-weight:600; }
    .paper ul, .paper ol { margin:0 0 1em 1.3em; color:#cfcfcf; line-height:1.75; }
    .paper li { margin-bottom:0.3em; }
    .paper .table-wrap { overflow-x:auto; margin:1.3em 0 1.7em; }
    .paper table { width:100%; border-collapse:collapse; font-size:0.9em; }
    .paper caption { color:#888; font-size:0.86em; text-align:left; letter-spacing:0.3px;
        margin-bottom:0.65em; }
    .paper th { color:#9ec9b0; font-weight:600; padding:8px 13px; border-bottom:1px solid #2c2c2c;
        white-space:nowrap; }
    .paper td { color:#cfcfcf; padding:7px 13px; border-bottom:1px solid #151515; }
    .paper tbody tr:hover td { background:#0e0e0e; }
"""


def render_whitepaper(paper_html: str, pdf_href: str, nav: str, generated: str = "") -> str:
    """Wrap the pandoc-converted white paper in the site shell: a short intro, a
    PDF download button, then the full paper text rendered inline and scrollable."""
    esc = html.escape
    meta = " · ".join(
        [b for b in (f"Generated {esc(generated)}" if generated else "",
                     f"written on-device by {esc(config.ASSISTANT_NAME)}") if b])
    meta_html = f'<div class="wp-meta">{meta}</div>' if meta else ""
    # pandoc emits a bare <table>; wrap it so wide tables scroll on narrow screens.
    paper_html = paper_html.replace("<table>", '<div class="table-wrap"><table>') \
                           .replace("</table>", "</table></div>")
    intro = (f"The AI-generated summary white paper for {esc(config.CONF_SHORT)} "
             f"{esc(config.CONF_YEAR)} — a concise synthesis of the meeting written "
             "automatically by the local, on-device assistant from the talk archive. "
             "Read the full text below, or download the typeset PDF.")
    body = (f'<style>{WHITEPAPER_CSS}</style>'
            f'<p class="wp-intro">{intro}</p>'
            f'<div class="wp-actions">'
            f'<a class="wp-dl" href="{esc(pdf_href)}"><span class="arr">&#8681;</span> '
            f'Download PDF</a>{meta_html}</div>'
            f'<article class="paper">{paper_html}</article>')
    return ts._page_shell(f"{config.CONF_SHORT} — White paper", "White paper", nav, body)


# ── Assets ─────────────────────────────────────────────────────────────────────

def copy_assets(refs, talks, out: Path, all_slides, include_transcripts, verbose):
    copied = 0
    if all_slides:
        for t in talks:
            sd = ts.SAVE_DIR / t["folder"] / "slides"
            for f in sorted(sd.glob("slide_*.jpg")) if sd.is_dir() else []:
                dst = out / "talks" / t["folder"] / "slides" / f.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, dst)
                copied += 1
    else:
        for rel in refs:                       # rel = "talks/<folder>/slides/<file>"
            relp = unquote(rel)
            src = ts.SAVE_DIR / relp[len("talks/"):]
            if not src.is_file():
                _warn(f"referenced asset missing on disk: {relp}")
                continue
            dst = out / relp
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            copied += 1

    if include_transcripts:
        _warn("--include-transcripts: PUBLISHING FULL VERBATIM TRANSCRIPTS with no "
              "access control")
        for t in talks:
            for name in ts.PRIVATE_FILES:
                src = ts.SAVE_DIR / t["folder"] / name
                if src.is_file():
                    dst = out / "talks" / t["folder"] / name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
                    copied += 1
    if verbose:
        print(f"  copied {copied} asset file(s)", flush=True)
    return copied


# ── Verification ───────────────────────────────────────────────────────────────

_FORBIDDEN = ['="/summaries', '="/topics"', '"/talks/', 'href="/"',
              'function vote(', '/vote', '/topics.json', '/stream',
              'EventSource', 'class="qvote"', 'class="generating"', 'id="gen-timer"']


def verify(out: Path, pages: dict, refs, include_transcripts) -> list:
    problems = []
    for name, htmlstr in pages.items():
        for bad in _FORBIDDEN:
            if bad in htmlstr:
                problems.append(f"{name}: leftover {bad!r}")
    if 'class="tcard"' not in pages.get("topics.html", ""):
        problems.append("topics.html: no topic cards (generation may have failed)")
    for rel in refs:
        if not (out / unquote(rel)).is_file():
            problems.append(f"missing copied asset: {rel}")
    if not include_transcripts:
        for p in out.rglob("*"):
            if p.name in ts.PRIVATE_FILES:
                problems.append(f"private file leaked into export: {p}")
    return problems


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Export the archive as a static site.")
    ap.add_argument("--out", default="site", help="output directory (default: site)")
    ap.add_argument("--clean", action="store_true", help="remove the output dir first")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip the LLM (topics use the keyword fallback; stale day "
                         "overviews render as placeholders)")
    ap.add_argument("--all-slides", action="store_true",
                    help="(deprecated; now always on) every slide is copied for the lightbox")
    ap.add_argument("--include-transcripts", action="store_true",
                    help="also publish full transcripts (PUBLIC, no access control)")
    ap.add_argument("--strict", action="store_true",
                    help="treat missing summaries / unavailable overviews as errors")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    apply_allowlist(args.verbose)   # fail-closed: only public_talks.txt talks
    talks = ts._load_talks()
    if not talks:
        print("No approved talks found (check public_talks.txt) — nothing to export.")
        return

    use_llm = (not args.no_llm) and ts.SUMMARIES_ENABLED
    if not use_llm:
        # Stop the render functions from spawning background generation threads.
        ts.SUMMARIES_ENABLED = False

    print(f"Exporting {len(talks)} talks "
          f"({'LLM ' + ts.LLM_MODEL if use_llm else 'no LLM'}) → {out}/", flush=True)
    ensure_caches(talks, use_llm, args.strict, args.verbose)

    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # Concepts/Tools indexes (built on-box by concepts_tools.py); pages appear
    # only when both are present and non-empty.
    concepts = _load_index("concepts.json")
    tools = _load_index("tools.json")
    have_ct = bool(concepts and concepts.get("concepts")) and bool(tools and tools.get("tools"))

    # White paper: convert whitepaper.tex → in-browser HTML (needs pandoc) and
    # publish whitepaper.pdf alongside it. The page appears only when both the
    # .tex and .pdf exist and the pandoc conversion succeeds.
    wp_body = None
    if WHITEPAPER_TEX.is_file() and WHITEPAPER_PDF.is_file():
        wp_body = _pandoc_body(WHITEPAPER_TEX)
    elif WHITEPAPER_TEX.is_file() or WHITEPAPER_PDF.is_file():
        _warn("white paper: need both whitepaper.tex and whitepaper.pdf — skipping the page")
    have_wp = wp_body is not None

    nav_frag = '<a href="index.html">home</a>'
    if have_ct:
        nav_frag += '<a href="methods.html">methods</a><a href="tools.html">tools</a>'
    if have_wp:
        nav_frag += '<a href="whitepaper.html">white paper</a>'

    # Render every page through the static post-processor. index.html is the new
    # landing page; the day-by-day archive (formerly index.html) is summaries.html.
    grouped = ts._group_by_day(talks)
    pages = {
        "index.html": staticize(render_landing(talks, grouped, have_ct, have_wp)),
        "summaries.html": add_nav(staticize(ts.render_summaries_page()), nav_frag),
        "topics.html": add_nav(staticize(ts.render_topics_page()), nav_frag),
    }
    for date, _label, _dtalks in grouped:
        pages[f"day-{date}.html"] = add_nav(staticize(ts.render_day_page(date)), nav_frag)
    # Secondary pages share a common inner nav (back to the day/topic archive);
    # add_nav then prepends home/methods/tools/white-paper.
    inner_nav = ('<a href="summaries.html">day summaries</a>'
                 '<a href="topics.html">key topics</a>')
    if have_ct:
        pages["methods.html"] = add_nav(staticize(render_concepts(concepts, inner_nav)), nav_frag)
        pages["tools.html"] = add_nav(staticize(render_tools(tools, inner_nav)), nav_frag)
    if have_wp:
        wp_page = render_whitepaper(wp_body, "whitepaper.pdf", inner_nav,
                                    _whitepaper_generated(WHITEPAPER_TEX))
        pages["whitepaper.html"] = add_nav(staticize(wp_page), nav_frag)
    for name, body in pages.items():
        (out / name).write_text(body)
        if args.verbose:
            print(f"  wrote {name} ({len(body)} bytes)", flush=True)

    # Copy EVERY slide of every published talk — the lightbox pages through them
    # all (the card thumbnail is just slide 1). Culled slides live in slides/.culled/
    # and are never globbed, so they're excluded automatically. `refs` (the slide-1
    # thumbnails parsed from the HTML) is still used by verify() as a sanity subset.
    combined = "\n".join(pages.values())
    refs = sorted(set(re.findall(r'(?:src|href)="(talks/[^"]+)"', combined)))
    copy_assets(refs, talks, out, all_slides=True,
                include_transcripts=args.include_transcripts, verbose=args.verbose)

    # Publish the typeset white paper PDF next to its in-browser page.
    if have_wp:
        shutil.copyfile(WHITEPAPER_PDF, out / "whitepaper.pdf")
        if args.verbose:
            print(f"  copied whitepaper.pdf ({WHITEPAPER_PDF.stat().st_size} bytes)", flush=True)

    (out / ".nojekyll").write_text("")

    problems = verify(out, pages, refs, args.include_transcripts)
    if problems:
        print("\nVERIFY FAILED:", flush=True)
        for p in problems:
            print(f"  ✗ {p}", flush=True)
        raise SystemExit(1)

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\n✓ {len(pages)} pages, {len(refs)} thumbnails, {total / 1e6:.1f} MB → {out}/")
    print("  verify OK · serve locally with:  python3 -m http.server -d "
          f"{out} 8000")


if __name__ == "__main__":
    main()
