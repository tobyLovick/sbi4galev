#!/usr/bin/env python3
"""Build the Concepts and Tools indexes for the public archive.

Two stages, on-box (the local model only):

  extract  alan (config [llm]) reads the APPROVED talks' summaries (titles, key
           points, topics, abstracts) and the cross-talk topic synthesis, and
           returns two lists: concepts (methods/terms like NPE, NLE, evidence)
           and tools (software/codes), each with aliases for matching, a short
           definition/description, and — for tools — candidate URLs. alan can't
           browse, so every proposed tool URL is then CHECKED by actually fetching
           it; broken/hallucinated links are blanked and alan is asked for fresh
           candidates (also fetched) so no dead link is ever published. Written to
           transcripts/concepts_tools.raw.json for review.

  count    A deterministic pass: for every concept/tool alias, count mentions in
           each talk's transcript and in the per-slide descriptions
           (slides_audit.json), and record which talks mention it. Writes the
           final transcripts/concepts.json and transcripts/tools.json that
           export_static renders.

  urls     Re-verify (and repair, via alan) the tool links in an existing
           raw.json, then recount — fixes broken links without re-extracting.

    python3 concepts_tools.py build       # extract then count (the usual path)
    python3 concepts_tools.py extract     # just the LLM stage -> raw.json
    python3 concepts_tools.py count       # just recount from raw.json (no LLM)
    python3 concepts_tools.py urls        # re-check/repair tool links, recount

Only the APPROVED talks (public_talks.txt) are considered, so the indexes — and
the related-talk links they render — match exactly what is published.
"""
import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import transcript_server as ts
import export_static as es

RAW_PATH = ts.SAVE_DIR / "concepts_tools.raw.json"
CONCEPTS_PATH = ts.SAVE_DIR / "concepts.json"
TOOLS_PATH = ts.SAVE_DIR / "tools.json"

EXTRACT_SYSTEM = (
    "You are analysing a simulation-based inference (SBI) astrophysics conference. "
    "From the talk summaries you are given, extract TWO lists and reply with ONLY a "
    "JSON object {\"concepts\": [...], \"tools\": [...]}.\n\n"
    "concepts = the key technical concepts, methods and terms discussed across the "
    "meeting (e.g. Neural Posterior Estimation, Neural Likelihood Estimation, "
    "Neural Ratio Estimation, simulation-based inference, approximate Bayesian "
    "computation, normalizing flows, the Bayesian evidence, posterior, MCMC, "
    "simulation-based calibration). Each: "
    '{"name": "<canonical name>", "abbr": "<short form or \'\'>", '
    '"aliases": ["<every spelling/abbrev used in talks>", ...], '
    '"definition": "<=20 words"}.\n'
    "Do NOT list the conference's overarching subject itself — simulation-based "
    "inference (SBI) — as a concept; it applies to essentially every talk. List the "
    "specific methods and terms used WITHIN it.\n\n"
    "tools = named software packages, codes, frameworks or pipelines (e.g. sbi, "
    "swyft, lampe, emcee, JAX, numpyro, pop-cosmos) AND named simulation suites / "
    "simulation datasets used as data or training sets (e.g. CAMELS, IllustrisTNG, "
    "SIMBA, EAGLE, Magneticum, FLAMINGO, UniverseMachine, L-Galaxies, pop-cosmos). "
    "Include both kinds "
    "in this one list. Each: "
    '{"name": "<canonical name>", "aliases": ["<spellings used>", ...], '
    '"url": "<official site or repo if you are confident, else \'\'>", '
    '"description": "<=15 words; note if it is a simulation suite/dataset>"}.\n\n'
    "Aliases must be the literal strings to search for in transcripts (include the "
    "abbreviation AND the spelled-out form). Be generous but precise; do not invent "
    "tools or suites that were not mentioned. Order each list by how central it was."
)


def _corpus(talks):
    blocks = []
    for t in talks:
        s = t.get("summary") or {}
        if not s:
            continue
        kp = "; ".join(s.get("key_points", []) or [])
        tp = ", ".join(s.get("topics", []) or [])
        ab = (s.get("abstract") or "")[:400]
        blocks.append(f"### {t['title']} — {s.get('speaker','')}\n"
                      f"key points: {kp}\ntopics: {tp}\nabstract: {ab}")
    # Cross-talk topic synthesis, if present, gives extra concept signal.
    topics = ts._load_topics_summary() or {}
    if topics.get("topics"):
        names = "; ".join(tp.get("title", "") for tp in topics["topics"])
        blocks.append(f"### Cross-talk key topics\n{names}")
    return "\n\n".join(blocks)


def _url_ok(url, timeout=8):
    """True if the URL actually resolves (final HTTP status < 400). alan cannot
    browse, so the pipeline fetches each candidate itself — this is what makes a
    proposed link 'checked', guaranteeing no dead links reach the page."""
    if not (isinstance(url, str) and url.startswith("http")):
        return False
    headers = {"User-Agent": "Mozilla/5.0 (ambient-ai link check)"}
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(url, method=method, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return getattr(r, "status", 200) < 400
        except urllib.error.HTTPError as e:
            if method == "HEAD" and e.code in (403, 405, 406, 429):
                continue                       # server rejects HEAD/bots → try GET
            return False
        except Exception:
            if method == "HEAD":
                continue                       # HEAD unsupported → try GET
            return False
    return False


def _verify_and_fill_urls(tools, timeout=180):
    """Guarantee every published tool link works. alan PROPOSES URLs; this fetches
    each to CHECK it. Broken URLs (incl. alan's hallucinations) are blanked, then
    alan is asked for fresh candidates for anything still missing and those are
    verified too. Anything that never resolves is left blank rather than dead."""
    blanked = []
    for t in tools:
        u = t.get("url", "")
        if u and not _url_ok(u):
            blanked.append(t.get("name", ""))
            t["url"] = ""
    missing = [t.get("name", "") for t in tools if not t.get("url") and t.get("name")]
    filled = 0
    if missing:
        prompt = ("For each of these software tools/packages used at a "
                  "simulation-based inference astrophysics conference, list up to 3 "
                  "candidate official URLs (code repository or project homepage), "
                  "most likely first. Reply with ONLY a JSON object mapping each "
                  "exact name to a list of URL strings:\n"
                  + "\n".join(f"- {m}" for m in missing))
        try:
            raw = ts._llm_chat([{"role": "user", "content": prompt}],
                               max_tokens=1500, temperature=0.0, timeout=timeout)
            s = raw.strip()
            cand = json.loads(s[s.find("{"):s.rfind("}") + 1])
        except Exception as e:
            print(f"  (url candidate request failed: {e})", file=sys.stderr)
            cand = {}
        for t in tools:
            if t.get("url"):
                continue
            cands = cand.get(t.get("name", ""), []) or []
            if isinstance(cands, str):
                cands = [cands]
            for u in cands:
                if isinstance(u, str) and _url_ok(u):
                    t["url"] = u.strip()
                    filled += 1
                    break
    still = [t.get("name", "") for t in tools if not t.get("url")]
    print(f"  url check: blanked {len(blanked)} broken, verified+filled {filled}, "
          f"{len(still)} left with no working URL")
    if blanked:
        print(f"    dropped broken link(s): {', '.join(blanked)}")
    if still:
        print(f"    no verified URL for: {', '.join(still)}")


def extract(talks, timeout=300):
    corpus = _corpus(talks)
    raw = ts._llm_chat(
        [{"role": "system", "content": EXTRACT_SYSTEM},
         {"role": "user", "content": corpus}],
        max_tokens=3000, temperature=0.1, timeout=timeout)
    s = raw.strip()
    i, j = s.find("{"), s.rfind("}")
    data = json.loads(s[i:j + 1])
    data = {"concepts": data.get("concepts", []), "tools": data.get("tools", [])}
    _verify_and_fill_urls(data["tools"], timeout=timeout)
    RAW_PATH.write_text(json.dumps(data, indent=2))
    print(f"extracted {len(data['concepts'])} concepts, {len(data['tools'])} tools -> {RAW_PATH.name}")
    return data


# ── deterministic counting ───────────────────────────────────────────────────

def _aliases(item):
    al = list(item.get("aliases") or [])
    for k in ("name", "abbr"):
        if item.get(k):
            al.append(item[k])
    # unique, longest first (so multi-word forms are tried before bare abbrevs)
    seen, out = set(), []
    for a in sorted({x.strip() for x in al if x and x.strip()}, key=len, reverse=True):
        lo = a.lower()
        if lo not in seen:
            seen.add(lo)
            out.append(a)
    return out


def _patterns(aliases):
    # Match each alias tolerating hyphen/space/underscore interchange, so a
    # canonical "pop-cosmos" also catches the spoken "pop cosmos" in transcripts
    # (and "Illustris-TNG" ~ "Illustris TNG", etc.). Single-word aliases are
    # unaffected (they reduce to \bword\b).
    pats = []
    for a in aliases:
        parts = [p for p in re.split(r"[\s\-_]+", a.strip()) if p]
        if not parts:
            continue
        body = r"[\s\-_]+".join(re.escape(p) for p in parts)
        pats.append(re.compile(r"\b" + body + r"\b", re.I))
    return pats


# The conference's overarching subject is not a useful "method" to list — it
# applies to essentially every talk — so it is excluded from the Methods page.
_EXCLUDE_CONCEPTS = {"simulation-based inference", "simulation based inference", "sbi"}


def _excluded_concept(it):
    nm = re.sub(r"\(.*?\)", "", it.get("name", "")).strip().lower()
    ab = (it.get("abbr", "") or "").strip().lower()
    return nm in _EXCLUDE_CONCEPTS or ab in _EXCLUDE_CONCEPTS


def count(raw, talks):
    audit = {}
    try:
        audit = json.loads((ts.SAVE_DIR / "slides_audit.json").read_text())
    except Exception:
        pass

    # Pre-load each talk's transcript text + concatenated slide descriptions.
    corpus = {}
    for t in talks:
        folder = t["folder"]
        tf = ts.SAVE_DIR / folder / "transcript.txt"
        text = tf.read_text() if tf.exists() else ""
        descs = [r.get("desc", "") for r in audit.get(folder, {}).values()]
        corpus[folder] = {"text": text, "slides": descs,
                          "title": t["title"], "date": ts._talk_day(t)}

    def tally(items, is_tool):
        out = []
        for it in items:
            pats = _patterns(_aliases(it))
            tment = sment = 0
            talks_hit = []
            for folder, c in corpus.items():
                t_n = sum(len(p.findall(c["text"])) for p in pats)
                s_n = sum(1 for d in c["slides"] if any(p.search(d) for p in pats))
                if t_n or s_n:
                    talks_hit.append({"folder": folder, "title": c["title"],
                                      "date": c["date"],
                                      "t": t_n, "s": s_n})
                tment += t_n
                sment += s_n
            talks_hit.sort(key=lambda x: -(x["t"] + x["s"]))
            rec = {"name": it.get("name", ""),
                   "aliases": _aliases(it),
                   "transcript_mentions": tment, "slide_mentions": sment,
                   "talks": talks_hit}
            if is_tool:
                rec["url"] = it.get("url", "")
                rec["description"] = it.get("description", "")
            else:
                rec["abbr"] = it.get("abbr", "")
                rec["definition"] = it.get("definition", "")
            out.append(rec)
        # drop never-mentioned items; sort by total mentions
        out = [r for r in out if r["talks"]]
        out.sort(key=lambda r: -(r["transcript_mentions"] + r["slide_mentions"]))
        return out

    concepts = tally([c for c in raw.get("concepts", []) if not _excluded_concept(c)],
                     is_tool=False)
    tools = tally(raw.get("tools", []), is_tool=True)
    CONCEPTS_PATH.write_text(json.dumps({"concepts": concepts}, indent=2))
    TOOLS_PATH.write_text(json.dumps({"tools": tools}, indent=2))
    print(f"counted {len(concepts)} concepts, {len(tools)} tools "
          f"-> {CONCEPTS_PATH.name}, {TOOLS_PATH.name}")
    miss = [t["name"] for t in tools if not t.get("url")]
    if miss:
        print(f"  tools missing a URL ({len(miss)}): {', '.join(miss)}")


def _approved_talks():
    es.apply_allowlist(False)        # restrict ts._load_talks() to public_talks.txt
    return ts._load_talks()


def main():
    ap = argparse.ArgumentParser(description="Build Concepts & Tools indexes.")
    ap.add_argument("cmd", choices=["build", "extract", "count", "urls"],
                    nargs="?", default="build")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()
    talks = _approved_talks()
    if args.cmd == "urls":
        # Re-verify/repair tool links in the existing extraction, then recount.
        raw = json.loads(RAW_PATH.read_text())
        _verify_and_fill_urls(raw.get("tools", []), timeout=args.timeout)
        RAW_PATH.write_text(json.dumps(raw, indent=2))
        count(raw, talks)
        return
    if args.cmd in ("build", "extract"):
        raw = extract(talks, timeout=args.timeout)
    else:
        raw = json.loads(RAW_PATH.read_text())
    if args.cmd in ("build", "count"):
        count(raw, talks)


if __name__ == "__main__":
    sys.exit(main())
