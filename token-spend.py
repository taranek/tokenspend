#!/usr/bin/env python3
"""token-spend — where do your billed Claude Code tokens go, per tool / per CLI command?

Parses Claude Code session transcripts (~/.claude/projects/**/*.jsonl) and attributes
REAL billed token usage (from the API's own `usage` fields) to tools and commands,
hierarchically: Bash: git -> checkout -> <args>, Edit -> <file>, mcp:server -> tool.

How attribution works:
  Each assistant API call reports usage. `input_tokens + cache_creation_input_tokens`
  is the NEW prompt content the API tokenized this call. The only new content since
  the previous call is (a) the previous call's own output (known exactly via its
  `output_tokens`) and (b) tool results / user messages injected in between. So:

      tool-result tokens ≈ new_input − previous_output

  measured by Anthropic's tokenizer, split proportionally by size when several tool
  results landed between two calls. Output tokens are likewise split across each
  call's text vs tool_use blocks.

Usage:
  token-spend.py                      # text report: current project, all sessions
  token-spend.py --serve              # interactive report, lazy-loaded levels (recommended)
  token-spend.py --all --serve
  token-spend.py --html [FILE]        # static self-contained report (data embedded)
  token-spend.py --json [FILE]        # full detailed report as JSON ('-' for stdout)
  token-spend.py --project blog --last
"""
import argparse
import json
import os
import re
import sys
import threading
import webbrowser
from collections import defaultdict, OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

PROJECTS_DIR = Path.home() / ".claude" / "projects"

SKIP_WORDS = {
    "cd", "echo", "printf", "export", "set", "source", ".", "exit", "true",
    "false", "then", "do", "if", "fi", "done", "else", "elif", "while", "for",
    "sudo", "command", "env", "time", "nohup", "xargs",
}

ASSISTANT_TEXT = "(assistant text + thinking)"
USER_MSGS = "(user messages)"
SYS_ATTACH = "(system attachments)"
SESSION_START = "(session start)"
OVERHEAD = "(context overhead)"

SS_CLAUDE_MD = "CLAUDE.md (est. from current file)"
SS_MEMORY = "memory MEMORY.md (est. from current file)"
SS_SYSTEM = "base system prompt + rest"
SS_FIRST_MSG = "first user message / replayed history"
SS_ATTACH = "context attachments"


# ---------------------------------------------------------------- labelling

def trunc(s, n=60):
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def short_path(p):
    home = str(Path.home())
    if p.startswith(home):
        p = "~" + p[len(home):]
    return p


def path_like(seg):
    """Heuristic: a segment that is a single filesystem path."""
    return "/" in seg and " " not in seg and "://" not in seg


def expand_seg(seg):
    """Explode a path-like segment into drillable directory components.
    '~/code/loco/web/src/App.tsx' -> ['~/code/loco', 'web', 'src', 'App.tsx']"""
    if not path_like(seg):
        return [seg]
    parts = [x for x in seg.split("/") if x]
    if seg.startswith("~/code/") and len(parts) >= 3:
        return ["~/code/" + parts[2]] + parts[3:]
    if seg.startswith("~/") and len(parts) >= 2:
        return ["~/" + parts[1]] + parts[2:]
    if seg.startswith("/"):
        return ["/" + parts[0]] + parts[1:]
    return parts


def command_words(cmdline):
    """Words of the most interesting simple command in a shell line."""
    cmdline = cmdline.replace("$(", "; ").replace("`", "; ")
    segments = re.split(r"&&|\|\||[|;\n]", cmdline)
    fallback = None
    for seg in segments:
        words = seg.strip().split()
        while words and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0]):
            words.pop(0)
        if not words:
            continue
        w = os.path.basename(words[0].strip("(){}"))
        if w.startswith("-") or not re.match(r"^[A-Za-z0-9._+-]+$", w):
            continue
        words = [w] + words[1:]
        if w not in SKIP_WORDS:
            return words
        if fallback is None:
            fallback = words
    return fallback or []


def label_path(tool_name, tool_input):
    """Hierarchical path for a tool call, e.g. ('Bash: git', 'checkout', '-b foo')."""
    if tool_name == "Bash":
        words = command_words(tool_input.get("command", ""))
        if not words:
            return ("Bash: (unknown)",)
        path = [f"Bash: {words[0]}"]
        if len(words) > 1:
            path.append(trunc(words[1], 40))
        if len(words) > 2:
            path.append(trunc(" ".join(words[2:]), 60))
        return tuple(path)
    if tool_name in ("Read", "Edit", "Write", "NotebookEdit"):
        fp = tool_input.get("file_path") or tool_input.get("notebook_path")
        return (tool_name, short_path(fp)) if fp else (tool_name,)
    if tool_name and tool_name.startswith("mcp__"):
        parts = tool_name.split("__")
        if len(parts) > 2:
            return (f"mcp:{parts[1]}", "__".join(parts[2:]))
        return (tool_name,)
    if tool_name == "WebSearch":
        q = tool_input.get("query")
        return ("WebSearch", trunc(q, 60)) if q else ("WebSearch",)
    if tool_name == "WebFetch":
        u = tool_input.get("url")
        return ("WebFetch", trunc(u, 60)) if u else ("WebFetch",)
    if tool_name == "Agent":
        sub = tool_input.get("subagent_type") or "general-purpose"
        desc = tool_input.get("description")
        return ("Agent", sub, trunc(desc, 60)) if desc else ("Agent", sub)
    if tool_name == "Skill":
        sk = tool_input.get("skill")
        return ("Skill", sk) if sk else ("Skill",)
    return (tool_name or "(unknown tool)",)


def content_chars(content):
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        n = 0
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    n += len(block.get("text", ""))
                elif block.get("type") == "image":
                    n += 6000  # rough: an image ≈ 1.5k tokens
            elif isinstance(block, str):
                n += len(block)
        return n
    return len(str(content))


# ---------------------------------------------------------------- analysis

_ctx_cache = {}


def session_context(cwd):
    """Estimated token sizes of files Claude Code injects at session start."""
    if cwd in _ctx_cache:
        return _ctx_cache[cwd]

    def file_tokens(p):
        try:
            return len(p.read_text(encoding="utf-8", errors="replace")) // 4
        except OSError:
            return 0

    claude_md = 0
    if cwd:
        # project CLAUDE.md files from cwd up to home, plus the global one
        d = Path(cwd)
        home = Path.home()
        seen = set()
        while True:
            f = d / "CLAUDE.md"
            if f not in seen:
                seen.add(f)
                if f.is_file():
                    claude_md += file_tokens(f)
            if d == home or d == d.parent:
                break
            d = d.parent
        g = home / ".claude" / "CLAUDE.md"
        if g not in seen and g.is_file():
            claude_md += file_tokens(g)

    memory = 0
    if cwd:
        slug = re.sub(r"[^A-Za-z0-9]", "-", cwd)
        m = PROJECTS_DIR / slug / "memory" / "MEMORY.md"
        if m.is_file():
            memory = file_tokens(m)

    _ctx_cache[cwd] = {"claude_md": claude_md, "memory": memory}
    return _ctx_cache[cwd]


def analyze_file(fpath, stats, totals, prefix=()):
    by_uuid = {}
    calls = OrderedDict()   # requestId -> {"head", "usage", "blocks"}
    tool_paths = {}         # tool_use_id -> path tuple
    session_cwd = None

    def S(path):
        return stats[prefix + path]

    try:
        fh = open(fpath, encoding="utf-8")
    except OSError as e:
        print(f"warning: cannot read {fpath}: {e}", file=sys.stderr)
        return
    with fh:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            uuid = d.get("uuid")
            if uuid:
                by_uuid[uuid] = d
            if session_cwd is None and d.get("cwd"):
                session_cwd = d["cwd"]
            if d.get("type") != "assistant":
                continue
            msg = d.get("message") or {}
            rid = d.get("requestId")
            usage = msg.get("usage") or {}
            if rid:
                call = calls.setdefault(rid, {"head": d, "usage": {}, "blocks": []})
                if usage:
                    call["usage"] = usage
            for block in msg.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    inp = block.get("input") or {}
                    path = label_path(block.get("name"), inp)
                    tool_paths[block.get("id")] = path
                    S(path)["calls"] += 1
                    if rid:
                        calls[rid]["blocks"].append((path, max(1, len(json.dumps(inp)))))
                elif block.get("type") in ("text", "thinking") and rid:
                    txt = block.get("text") or block.get("thinking") or ""
                    calls[rid]["blocks"].append(((ASSISTANT_TEXT,), max(1, len(txt))))

    for rid, call in calls.items():
        usage = call["usage"]
        if not usage:
            continue
        totals["input"] += usage.get("input_tokens", 0)
        totals["cache_write"] += usage.get("cache_creation_input_tokens", 0)
        totals["cache_read"] += usage.get("cache_read_input_tokens", 0)
        totals["output"] += usage.get("output_tokens", 0)
        totals["api_calls"] += 1

        # output side: split this call's output_tokens across its blocks
        out_tok = usage.get("output_tokens", 0)
        blocks = call["blocks"]
        if blocks and out_tok:
            w = sum(c for _, c in blocks)
            for path, c in blocks:
                S(path)["out"] += out_tok * c / w

        # input side: walk back to the previous API call, collect what was
        # injected in between, attribute new_input − prev_output across it
        new_input = usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
        items = []
        prev_out = None
        node = by_uuid.get(call["head"].get("parentUuid"))
        hops = 0
        while node is not None and hops < 10000:
            hops += 1
            t = node.get("type")
            if t == "assistant":
                prid = node.get("requestId")
                if prid in calls and calls[prid]["usage"]:
                    prev_out = calls[prid]["usage"].get("output_tokens", 0)
                    break
            elif t == "user":
                content = (node.get("message") or {}).get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            path = tool_paths.get(block.get("tool_use_id"), ("(orphan result)",))
                            items.append((path, max(1, content_chars(block.get("content")))))
                        elif isinstance(block, dict) and block.get("type") == "text":
                            items.append(((USER_MSGS,), max(1, len(block.get("text", "")))))
                elif isinstance(content, str):
                    items.append(((USER_MSGS,), max(1, len(content))))
            elif t == "attachment":
                att = node.get("attachment") or {}
                apath = [SYS_ATTACH, att.get("type") or "unknown"]
                if att.get("hookName"):
                    apath.append(att["hookName"])
                items.append((tuple(apath), max(1, len(json.dumps(att)))))
            node = by_uuid.get(node.get("parentUuid"))

        if prev_out is None:
            # First API call of a chain: new_input covers the system prompt
            # (incl. CLAUDE.md + memory, never logged in the transcript), the
            # first user message, and context attachments. Itemize with
            # chars/4 estimates for the known pieces; sizes of CLAUDE.md and
            # MEMORY.md come from the files as they exist on disk today.
            ctx = session_context(session_cwd)
            est = {SS_FIRST_MSG: 0, SS_ATTACH: 0}
            for path, c in items:
                key = SS_ATTACH if path[0] == SYS_ATTACH else SS_FIRST_MSG
                est[key] += c // 4
            remainder = max(0, new_input - est[SS_FIRST_MSG] - est[SS_ATTACH])
            claude_est = min(remainder, ctx["claude_md"])
            remainder -= claude_est
            mem_est = min(remainder, ctx["memory"])
            remainder -= mem_est
            for key, tok in ((SS_FIRST_MSG, est[SS_FIRST_MSG]),
                             (SS_ATTACH, est[SS_ATTACH]),
                             (SS_CLAUDE_MD, claude_est),
                             (SS_MEMORY, mem_est),
                             (SS_SYSTEM, remainder)):
                if tok:
                    S((SESSION_START, key))["in"] += min(tok, new_input)
            continue
        attributable = max(0, new_input - prev_out)
        if not attributable:
            continue
        if not items:
            S((OVERHEAD,))["in"] += attributable
            continue
        w = sum(c for _, c in items)
        for path, c in items:
            S(path)["in"] += attributable * c / w


class Analysis:
    """Lazily-computed stats over a set of transcript files."""

    def __init__(self, jobs, scope):
        self.jobs = jobs          # list of (file, path_prefix)
        self.scope = scope
        self.stats = None
        self.totals = None
        self._dir_stats = None
        self._lock = threading.Lock()

    def ensure(self):
        with self._lock:
            if self.stats is None:
                stats = defaultdict(lambda: {"calls": 0, "in": 0.0, "out": 0.0})
                totals = defaultdict(int)
                for f, prefix in self.jobs:
                    analyze_file(f, stats, totals, prefix)
                self.stats, self.totals = stats, totals
        return self

    def meta(self):
        return (f"{self.scope} · {len(self.jobs)} session file(s)"
                f" · {self.totals['api_calls']:,} API calls")

    def view_stats(self, view):
        """Path-keyed stats for a view: 'plain' as parsed, or 'dir' with
        path-like segments exploded into directory components."""
        if view != "dir":
            return self.stats
        with self._lock:
            if self._dir_stats is None:
                d = defaultdict(lambda: {"calls": 0, "in": 0.0, "out": 0.0})
                for path, s in self.stats.items():
                    key = tuple(x for seg in path for x in expand_seg(seg))
                    t = d[key]
                    t["calls"] += s["calls"]
                    t["in"] += s["in"]
                    t["out"] += s["out"]
                self._dir_stats = d
        return self._dir_stats

    def children(self, prefix, view="plain"):
        """One level of the hierarchy under `prefix` (a list of segments)."""
        stats = self.view_stats(view)
        prefix = tuple(prefix)
        depth = len(prefix)
        groups = defaultdict(lambda: {"c": 0, "i": 0.0, "o": 0.0, "more": False})
        total = 0.0
        for path, s in stats.items():
            if path[:depth] != prefix:
                continue
            total += s["in"] + s["out"]
            if len(path) <= depth:
                continue  # stats landing exactly at this node
            g = groups[path[depth]]
            g["c"] += s["calls"]
            g["i"] += s["in"]
            g["o"] += s["out"]
            if len(path) > depth + 1:
                g["more"] = True
        rows = [{"n": name, "c": g["c"], "i": round(g["i"]), "o": round(g["o"]),
                 "more": g["more"]}
                for name, g in groups.items()]
        rows.sort(key=lambda r: -(r["i"] + r["o"]))
        return {"total": round(total), "rows": rows}

    def search(self, q, view="plain", limit=60):
        """Nodes anywhere in the tree whose segment contains q (case-insensitive)."""
        q = q.lower()
        stats = self.view_stats(view)
        groups = {}
        grand = 0.0
        matched = 0.0
        for path, s in stats.items():
            grand += s["in"] + s["out"]
            hit = False
            for i, seg in enumerate(path):
                if q in seg.lower():
                    hit = True
                    key = path[: i + 1]
                    g = groups.setdefault(key, {"c": 0, "i": 0.0, "o": 0.0, "more": False})
                    g["c"] += s["calls"]
                    g["i"] += s["in"]
                    g["o"] += s["out"]
                    if len(path) > i + 1:
                        g["more"] = True
            if hit:
                matched += s["in"] + s["out"]
        rows = [{"path": list(k), "n": k[-1], "c": g["c"], "i": round(g["i"]),
                 "o": round(g["o"]), "more": g["more"]}
                for k, g in groups.items()]
        rows.sort(key=lambda r: -(r["i"] + r["o"]))
        return {"total": round(grand), "rows": rows[:limit], "matches": len(rows),
                "matched_total": round(matched)}


# ---------------------------------------------------------------- scopes

def find_project_dirs(pattern=None):
    if not PROJECTS_DIR.is_dir():
        sys.exit(f"error: {PROJECTS_DIR} not found — is Claude Code installed?")
    dirs = sorted(p for p in PROJECTS_DIR.iterdir() if p.is_dir())
    if pattern:
        dirs = [p for p in dirs if pattern.lower() in p.name.lower()]
    return dirs


def cwd_project_dir():
    slug = re.sub(r"[^A-Za-z0-9]", "-", os.getcwd())
    p = PROJECTS_DIR / slug
    return p if p.is_dir() else None


# ---------------------------------------------------------------- text report

def print_text_report(analysis, top):
    analysis.ensure()
    root = analysis.children([])
    rows, grand = root["rows"], root["total"] or 1
    t = analysis.totals
    print(f"\n  {analysis.meta()}")
    print(f"  billed totals:  output {t['output']:,}  ·  fresh input {t['input']:,}"
          f"  ·  cache write {t['cache_write']:,}  ·  cache read {t['cache_read']:,}\n")
    name_w = max((len(r["n"]) for r in rows[:top]), default=10) + 2
    header = f"  {'tool / command':<{name_w}} {'calls':>6} {'out tok':>9} {'in tok':>10} {'total':>9} {'share':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows[:top]:
        tot = r["i"] + r["o"]
        pct = 100.0 * tot / grand
        bar = "█" * max(1, round(pct / 2.5)) if pct >= 1 else "·"
        print(f"  {r['n']:<{name_w}} {r['c']:>6,} {r['o']:>9,} {r['i']:>10,}"
              f" {tot:>9,} {pct:>6.1f}%  {bar}")
    hidden = len(rows) - top
    if hidden > 0:
        rest = sum(r["i"] + r["o"] for r in rows[top:])
        print(f"  {'(+' + str(hidden) + ' more)':<{name_w}} {'':>6} {'':>9} {'':>10}"
              f" {rest:>9,} {100.0*rest/grand:>6.1f}%")
    print("\n  tip: --serve opens an interactive drill-down report\n")


# ---------------------------------------------------------------- html ui

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..700&display=swap" rel="stylesheet">
<title>token spend</title>
<style>
  :root {
    --bg-0: #08090a;          /* window */
    --bg-1: #0f1011;          /* panel */
    --bg-2: #141516;          /* raised */
    --bg-3: #191a1b;          /* card */
    --fg: #f7f8f8;
    --muted: #8a8f98;
    --subtle: #d0d6e0;
    --faint: #62666d;
    --border: #23252a;
    --hover: #ffffff0d;
    --acc: #84dcb7;
    --acc-bright: #95e3c3;
    --acc-tint: rgba(132, 220, 183, 0.12);
    --acc-deep: #4faf8a;
    --acc2: #e2c495;
    --acc2-deep: #b18d5c;
    --radius: 8px;
    --font-ui: "Inter Variable", "Inter", "SF Pro Display", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    --font-mono: "Berkeley Mono", ui-monospace, "SF Mono", Menlo, monospace;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  *, *::before, *::after { corner-shape: squircle; }
  html { background: var(--bg-0); }
  body {
    font-family: var(--font-ui);
    letter-spacing: -0.011em;
    color: var(--fg);
    -webkit-font-smoothing: antialiased;
    min-height: 100vh;
    background: var(--bg-0);
    max-width: 1080px;
    margin: 0 auto;
    padding: 40px clamp(20px, 4vw, 56px) 64px;
    user-select: none;
  }
  header { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; }
  h1 { font-size: 20px; font-weight: 590; letter-spacing: -0.022em; }
  h1::before {
    content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: var(--acc-deep); margin-right: 10px; vertical-align: 8%;
  }
  h1 em { font-style: normal; color: var(--acc); }
  .meta { color: var(--muted); font-size: 12px; }

  .stats {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px; margin: 22px 0 26px;
  }
  .stat {
    background: var(--bg-2); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 12px 14px 11px;
    animation: rise 0.35s cubic-bezier(0.2, 0.7, 0.2, 1) both;
  }
  .stat:nth-child(2) { animation-delay: 40ms; }
  .stat:nth-child(3) { animation-delay: 80ms; }
  .stat:nth-child(4) { animation-delay: 120ms; }
  .stat:nth-child(5) { animation-delay: 160ms; }
  .stat b { display: block; font-size: 19px; font-weight: 590; letter-spacing: -0.012em; font-variant-numeric: tabular-nums; }
  .stat span {
    display: block; color: var(--muted); font-size: 10px; font-weight: 500;
    letter-spacing: 0.08em; text-transform: uppercase; margin-top: 4px;
  }
  .stat.hot b { color: var(--acc-bright); }

  .panel {
    background: var(--bg-1); border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden;   /* rows: 6px radius + 6px gutter */
  }
  .scope-line {
    display: flex; justify-content: space-between; align-items: center; gap: 12px;
    padding: 11px 16px; background: var(--bg-2); border-bottom: 1px solid var(--border);
  }
  nav { display: flex; align-items: center; flex-wrap: wrap; gap: 2px; min-height: 24px; }
  nav .crumb {
    background: none; border: 0; font: inherit; font-size: 12.5px;
    color: var(--muted); cursor: pointer; padding: 4px 9px; border-radius: 999px;
  }
  nav .crumb:hover { background: var(--hover); color: var(--fg); }
  nav .sep { color: var(--muted); opacity: 0.45; padding: 0 1px; font-size: 12px; }
  nav .here {
    color: var(--acc-bright); font-size: 12.5px; font-weight: 510; padding: 4px 10px;
    background: var(--acc-tint); border-radius: 999px;
  }
  .searchwrap { position: relative; margin-left: auto; }
  #search {
    width: 190px;
    background: #ffffff08; border: 1px solid transparent; border-radius: 6px;
    font: inherit; font-size: 12.5px; color: var(--fg);
    padding: 4px 26px 4px 9px; outline: none;
  }
  #search-clear {
    position: absolute; right: 4px; top: 50%; translate: 0 -50%;
    width: 18px; height: 18px; border: 0; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    background: transparent; color: var(--faint); cursor: pointer;
    font-size: 10px; line-height: 1; padding: 0;
  }
  #search-clear:hover { background: #ffffff0d; color: var(--fg); }
  #search:placeholder-shown + #search-clear { display: none; }
  #search::placeholder { color: var(--faint); }
  #search:focus { border-color: var(--border); background: #ffffff0d; }
  #search::-webkit-search-cancel-button { -webkit-appearance: none; }
  .name .rowpath { color: var(--faint); font-size: 11px; margin-left: 8px; font-weight: 400; }
  .scope-total { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; white-space: nowrap; margin-left: auto; }
  .scope-total b { color: var(--fg); font-weight: 590; }

  .legend {
    display: flex; gap: 16px; color: var(--muted); font-size: 11px;
    padding: 10px 16px; border-bottom: 1px solid var(--border);
  }
  .legend { align-items: center; }
  .legend i { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }
  .legend .li i { background: var(--acc); }
  .legend .lo i { background: var(--acc2); }
  .seg {
    display: flex; gap: 2px; margin-left: 14px;
    background: #ffffff08; border-radius: 999px; padding: 2px;
  }
  .seg button {
    border: 0; background: none; font: inherit; font-size: 11px; font-weight: 500;
    color: var(--muted); padding: 3px 11px; border-radius: 999px; cursor: pointer;
  }
  .seg button:hover { color: var(--fg); }
  .seg button.active { background: #28282c; color: var(--fg); }

  .board { position: relative; min-height: 180px; }
  #rows { list-style: none; padding: 6px; }
  .row {
    display: grid;
    grid-template-columns: minmax(200px, 360px) 1fr 150px;
    gap: 14px; align-items: center;
    padding: 8px 12px; border-radius: 6px;
    animation: rise 0.2s ease both;
  }
  @keyframes rise { from { opacity: 0; } }
  .row.deeper { cursor: pointer; }
  .row.deeper:hover { background: var(--hover); }
  .name {
    font-size: 13px; color: var(--subtle); font-weight: 400;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    display: flex; align-items: center;
  }
  .name .arrow { color: var(--faint); margin-right: 7px; font-size: 10px; flex: none; }
  .name .calls {
    color: var(--faint); font-size: 10.5px; font-weight: 500;
    background: #ffffff08; border-radius: 999px; padding: 1px 7px; margin-left: 8px; flex: none;
  }
  .bar-track { height: 6px; border-radius: 999px; background: #ffffff08; overflow: hidden; }
  .bar { display: flex; height: 100%; width: 0%; border-radius: 999px; overflow: hidden; transition: width 0.5s cubic-bezier(0.25, 0.8, 0.25, 1); }
  .seg-in  { background: linear-gradient(180deg, var(--acc), var(--acc-deep)); }
  .seg-out { background: linear-gradient(180deg, var(--acc2), var(--acc2-deep)); }
  .nums {
    text-align: right; font-size: 12.5px;
    font-variant-numeric: tabular-nums; white-space: nowrap;
  }
  .nums b { font-weight: 590; }
  .nums .pct { display: inline-block; width: 48px; color: var(--acc-bright); font-size: 11.5px; }
  .nums .tok { color: var(--muted); }

  #loader {
    position: absolute; inset: 0; z-index: 5;
    display: flex; align-items: center; justify-content: center;
    background: rgba(8, 9, 10, 0.6); backdrop-filter: blur(4px);
    opacity: 0; pointer-events: none; transition: opacity 0.18s;
  }
  #loader.on { opacity: 1; pointer-events: auto; }
  .loader-card {
    display: flex; flex-direction: column; align-items: center; gap: 12px;
    background: var(--bg-3); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 18px 24px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
  }
  .scan { width: 160px; height: 4px; border-radius: 999px; background: #ffffff10; overflow: hidden; position: relative; }
  .scan::before {
    content: ""; position: absolute; top: 0; bottom: 0; width: 40%; border-radius: 999px;
    background: linear-gradient(90deg, transparent, var(--acc), transparent);
    animation: sweep 0.9s linear infinite;
  }
  @keyframes sweep { from { left: -40%; } to { left: 100%; } }
  #loader .msg { color: var(--muted); font-size: 11px; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; }
  #loader .msg::after { content: ""; animation: dots 1.2s steps(4) infinite; }
  @keyframes dots { 0% { content: ""; } 25% { content: "."; } 50% { content: ".."; } 75% { content: "..."; } }

  .hint { color: var(--muted); opacity: 0.75; font-size: 11px; padding: 10px 16px; border-top: 1px solid var(--border); }
  footer { margin-top: 18px; color: var(--muted); font-size: 11px; line-height: 1.7; max-width: 76ch; text-wrap: pretty; }
  footer b { color: var(--subtle); font-weight: 600; }
  footer code { font-family: var(--font-mono); color: var(--subtle); }

  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
      transition-duration: 0.01ms !important;
      animation-duration: 0.01ms !important;
    }
  }
</style>
</head>
<body>
<header>
  <h1>token <em>spend</em></h1>
  <div class="meta" id="meta"></div>
</header>
<div class="stats" id="stats"></div>

<main class="panel">
  <div class="scope-line">
    <nav id="crumbs"></nav>
    <div class="searchwrap">
      <input id="search" type="search" placeholder="Search…  /" spellcheck="false" autocomplete="off" autofocus>
      <button id="search-clear" title="Clear (Esc)" aria-label="Clear search">✕</button>
    </div>
  </div>
  <div class="legend">
    <span class="li"><i></i>in — tool result entering context</span>
    <span class="lo"><i></i>out — writing the call (args)</span>
    <div class="scope-total" id="scope-total"></div>
    <div class="seg" id="viewseg" title="how path-like arguments are grouped">
      <button data-v="plain" class="active">args</button>
      <button data-v="dir">by directory</button>
    </div>
  </div>
  <div class="board">
    <ul id="rows"></ul>
    <div id="loader"><div class="loader-card"><div class="scan"></div><div class="msg" id="loader-msg">loading</div></div></div>
  </div>
  <div class="hint" id="hint"></div>
</main>

<footer>
  <b>method</b> — real billed usage from transcript <code>usage</code> fields. For each API
  call, <code>input + cache_creation</code> minus the previous call's <code>output</code> is
  attributed to the tool results injected between the two calls (split by size).
  Output tokens are split across each call's text vs tool-call blocks. Session-start rows
  for CLAUDE.md and memory are estimated from the files as they exist on disk today.
  Cache reads (~0.1× price re-reads of history) are shown in the totals but not attributed per-tool.
</footer>

<script>
const BOOT = __BOOT__;   // {mode:"serve"} or {mode:"static", tree, totals, meta}

const fmt = n => n.toLocaleString("en-US");
const fmtK = n => n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n >= 1e4 ? Math.round(n / 1e3) + "k" : fmt(n);
const esc = s => String(s).replace(/[&<>"']/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

let loaderCount = 0, loaderTimer = null;
function loading(on, msg) {
  const el = document.getElementById("loader");
  if (on) {
    loaderCount++;
    document.getElementById("loader-msg").textContent = msg || "loading";
    // only show the overlay if the wait is perceptible
    if (!loaderTimer) loaderTimer = setTimeout(() => el.classList.add("on"), 120);
  } else {
    loaderCount = Math.max(0, loaderCount - 1);
    if (!loaderCount) {
      clearTimeout(loaderTimer); loaderTimer = null;
      el.classList.remove("on");
    }
  }
}

// ---- data access: lazy fetch in serve mode, local walk in static mode ----
async function apiSummary() {
  if (BOOT.mode === "static") return { totals: BOOT.totals, meta: BOOT.meta };
  const r = await fetch("/api/summary");
  return r.json();
}

async function apiSearch(term, v) {
  if (BOOT.mode === "static") {
    const q = term.toLowerCase();
    const rows = [];
    let matchedTotal = 0;
    const walk = (node, path, covered) => {
      for (const c of node.ch) {
        const p = [...path, c.n];
        const hit = c.n.toLowerCase().includes(q);
        if (hit) {
          rows.push({ path: p, n: c.n, c: c.c, i: c.i, o: c.o, more: c.ch.length > 0 });
          if (!covered) matchedTotal += c.i + c.o;
        }
        walk(c, p, covered || hit);
      }
    };
    walk(BOOT.trees[v], [], false);
    rows.sort((a, b) => (b.i + b.o) - (a.i + a.o));
    return { total: BOOT.trees[v].i + BOOT.trees[v].o, rows: rows.slice(0, 60),
             matches: rows.length, matched_total: matchedTotal };
  }
  const r = await fetch("/api/search?q=" + encodeURIComponent(term) + "&view=" + v);
  return r.json();
}

async function apiChildren(path, v) {
  if (BOOT.mode === "static") {
    let node = BOOT.trees[v];
    for (const seg of path) node = node && node.ch.find(c => c.n === seg);
    if (!node) return { total: 0, rows: [] };
    return {
      total: node.i + node.o,
      rows: node.ch.map(c => ({ n: c.n, c: c.c, i: c.i, o: c.o, more: c.ch.length > 0 })),
    };
  }
  const r = await fetch("/api/children?path=" + encodeURIComponent(JSON.stringify(path)) + "&view=" + v);
  return r.json();
}

// ---- rendering ----
let stack = [];        // segments of the current drill path
let view = "plain";    // "plain" args vs "dir" (paths exploded by directory)
let searchQ = "";      // active search term ("" = normal drill view)
const cache = new Map();

async function level(path) {
  const key = view + ":" + JSON.stringify(path);
  if (!cache.has(key)) cache.set(key, await apiChildren(path, view));
  return cache.get(key);
}

// switching views: keep as much of the drill path as still exists in the other view
async function setView(v) {
  if (v === view) return;
  view = v;
  document.querySelectorAll("#viewseg button").forEach(b =>
    b.classList.toggle("active", b.dataset.v === v));
  loading(true, "regrouping");
  try {
    const kept = [];
    for (const seg of stack) {
      const data = await level(kept);
      const hit = data.rows.find(r => r.n === seg);
      if (!hit || !hit.more) break;
      kept.push(seg);
    }
    stack = kept;
  } finally { loading(false); }
  render();
}
document.querySelectorAll("#viewseg button").forEach(b =>
  b.addEventListener("click", () => setView(b.dataset.v)));

function renderCrumbs() {
  document.getElementById("crumbs").innerHTML =
    ["all tools", ...stack].map((label, idx) => {
      if (idx === stack.length) return `<span class="here">${esc(label)}</span>`;
      return `<button class="crumb" data-idx="${idx}">${esc(label)}</button><span class="sep">/</span>`;
    }).join("");
  document.querySelectorAll(".crumb").forEach(b =>
    b.addEventListener("click", () => { stack = stack.slice(0, +b.dataset.idx); render(); }));
}

async function render() {
  if (searchQ.length >= 2) return renderSearch();
  renderCrumbs();
  loading(true, stack.length ? "loading level" : "crunching transcripts");
  let data;
  try { data = await level(stack); } finally { loading(false); }

  const total = data.total || 1;
  document.getElementById("scope-total").innerHTML = `<b>${fmt(data.total)}</b> tok in scope`;

  const rows = data.rows;
  const max = rows.length ? rows[0].i + rows[0].o : 1;
  const ul = document.getElementById("rows");
  ul.innerHTML = rows.map((r, idx) => {
    const t = r.i + r.o;
    const pct = 100 * t / total;
    return `<li class="row${r.more ? " deeper" : ""}" data-idx="${idx}" style="animation-delay:${Math.min(idx * 18, 400)}ms">
      <div class="name"><span class="arrow">${r.more ? "▸" : "·"}</span>${esc(r.n)}${r.c ? `<span class="calls">×${fmt(r.c)}</span>` : ""}</div>
      <div class="bar-track"><div class="bar" data-w="${(100 * t / max).toFixed(2)}">
        <div class="seg-in" style="flex:${r.i}"></div><div class="seg-out" style="flex:${r.o}"></div>
      </div></div>
      <div class="nums"><span class="pct">${pct >= 0.1 ? pct.toFixed(1) : "<0.1"}%</span> <b>${fmtK(t)}</b> <span class="tok">tok</span></div>
    </li>`;
  }).join("");

  ul.querySelectorAll(".row.deeper").forEach(li =>
    li.addEventListener("click", () => { stack = [...stack, rows[+li.dataset.idx].n]; render(); }));

  requestAnimationFrame(() => requestAnimationFrame(() =>
    ul.querySelectorAll(".bar").forEach(b => { b.style.width = b.dataset.w + "%"; })));

  document.getElementById("hint").textContent = rows.some(r => r.more)
    ? "▸ click a row to drill into the next argument level — breadcrumbs to go back"
    : "leaf level — nothing deeper here";
}

async function renderSearch() {
  document.getElementById("crumbs").innerHTML =
    `<span class="here">search: ${esc(searchQ)}</span>`;
  loading(true, "searching");
  let data;
  try { data = await apiSearch(searchQ, view); } finally { loading(false); }

  const total = data.total || 1;
  const mpct = 100 * data.matched_total / (data.total || 1);
  document.getElementById("scope-total").innerHTML =
    `<b>${fmt(data.matches)}</b> match${data.matches === 1 ? "" : "es"} · ` +
    `<b>${fmt(data.matched_total)}</b> tok (${mpct >= 0.1 ? mpct.toFixed(1) : "<0.1"}%)`;

  const rows = data.rows;
  const max = rows.length ? rows[0].i + rows[0].o : 1;
  const ul = document.getElementById("rows");
  ul.innerHTML = rows.map((r, idx) => {
    const t = r.i + r.o;
    const pct = 100 * t / total;
    const parents = r.path.slice(0, -1).join(" / ");
    return `<li class="row deeper" data-idx="${idx}" style="animation-delay:${Math.min(idx * 12, 300)}ms">
      <div class="name"><span class="arrow">${r.more ? "▸" : "·"}</span>${esc(r.n)}${r.c ? `<span class="calls">×${fmt(r.c)}</span>` : ""}${parents ? `<span class="rowpath">${esc(parents)}</span>` : ""}</div>
      <div class="bar-track"><div class="bar" data-w="${(100 * t / max).toFixed(2)}">
        <div class="seg-in" style="flex:${r.i}"></div><div class="seg-out" style="flex:${r.o}"></div>
      </div></div>
      <div class="nums"><span class="pct">${pct >= 0.1 ? pct.toFixed(1) : "<0.1"}%</span> <b>${fmtK(t)}</b> <span class="tok">tok</span></div>
    </li>`;
  }).join("");

  ul.querySelectorAll(".row").forEach(li =>
    li.addEventListener("click", () => {
      const r = rows[+li.dataset.idx];
      stack = r.more ? r.path : r.path.slice(0, -1);
      searchQ = "";
      document.getElementById("search").value = "";
      render();
    }));

  requestAnimationFrame(() => requestAnimationFrame(() =>
    ul.querySelectorAll(".bar").forEach(b => { b.style.width = b.dataset.w + "%"; })));

  document.getElementById("hint").textContent = rows.length
    ? "click a result to jump to it in the tree — clear the search to go back"
    : "no nodes match — token totals are searched by name at every level";
}

const searchEl = document.getElementById("search");
let searchTimer = null;
searchEl.addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    searchQ = searchEl.value.trim();
    render();
  }, 180);
});
searchEl.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { searchEl.value = ""; searchQ = ""; searchEl.blur(); render(); }
});
document.getElementById("search-clear").addEventListener("click", () => {
  searchEl.value = ""; searchQ = ""; searchEl.focus(); render();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "/" && document.activeElement !== searchEl) { e.preventDefault(); searchEl.focus(); }
});

async function boot() {
  loading(true, "crunching transcripts");
  let s;
  try { s = await apiSummary(); } finally { loading(false); }
  document.getElementById("meta").textContent = s.meta || "";
  document.getElementById("stats").innerHTML = [
    ["output", s.totals.output, 1],
    ["fresh input", s.totals.input, 0],
    ["cache write", s.totals.cache_write, 0],
    ["cache read", s.totals.cache_read, 0],
    ["api calls", s.totals.api_calls, 0],
  ].map(([label, v, hot]) =>
    `<div class="stat${hot ? " hot" : ""}"><b>${fmtK(v || 0)}</b><span>${label}</span></div>`
  ).join("");
  await render();
}

boot();
</script>
</body>
</html>
"""


def render_html(boot):
    return HTML_TEMPLATE.replace("__BOOT__", json.dumps(boot))


# static export: full tree embedded (loads everything up front; --serve is lazy)

def full_tree(analysis, view):
    def expand(path):
        data = analysis.children(path, view)
        return [{"n": r["n"], "c": r["c"], "i": r["i"], "o": r["o"],
                 "ch": expand(path + [r["n"]]) if r["more"] else []}
                for r in data["rows"]]
    root = analysis.children([], view)
    return {"n": "all", "c": 0, "i": root["total"], "o": 0, "ch": expand([])}


def write_static_html(analysis, out_path):
    analysis.ensure()
    boot = {"mode": "static",
            "trees": {"plain": full_tree(analysis, "plain"),
                      "dir": full_tree(analysis, "dir")},
            "totals": dict(analysis.totals), "meta": analysis.meta()}
    Path(out_path).write_text(render_html(boot), encoding="utf-8")


# ---------------------------------------------------------------- json export

def write_json_report(analysis, out_path):
    analysis.ensure()

    def readable(node):
        out = {
            "name": node["n"],
            "calls": node["c"],
            "input_tokens": node["i"],
            "output_tokens": node["o"],
            "total_tokens": node["i"] + node["o"],
        }
        if node["ch"]:
            out["children"] = [readable(c) for c in node["ch"]]
        return out

    t = analysis.totals
    report = {
        "generator": "tokenspend (https://github.com/taranek/tokenspend)",
        "scope": analysis.scope,
        "session_files": len(analysis.jobs),
        "api_calls": t["api_calls"],
        "billed_totals": {
            "output_tokens": t["output"],
            "fresh_input_tokens": t["input"],
            "cache_write_tokens": t["cache_write"],
            "cache_read_tokens": t["cache_read"],
        },
        "notes": [
            "attribution: per API call, input+cache_creation minus the previous call's "
            "output is attributed to the tool results injected between the two calls",
            "cache reads are in billed_totals but not attributed per node",
            "session-start CLAUDE.md/memory rows are estimated from current file sizes",
        ],
        "attribution": readable(full_tree(analysis, "plain")),
        "attribution_by_directory": readable(full_tree(analysis, "dir")),
    }
    body = json.dumps(report, indent=2)
    if out_path == "-":
        print(body)
    else:
        Path(out_path).write_text(body + "\n", encoding="utf-8")
        print(f"JSON report written to {Path(out_path).resolve()}")


# ---------------------------------------------------------------- server

def serve(analysis, port, open_browser=True):
    page = render_html({"mode": "serve"}).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            try:
                if url.path == "/":
                    self._send(page, "text/html; charset=utf-8")
                elif url.path == "/api/summary":
                    analysis.ensure()
                    body = json.dumps({"totals": dict(analysis.totals),
                                       "meta": analysis.meta()}).encode()
                    self._send(body, "application/json")
                elif url.path == "/api/children":
                    analysis.ensure()
                    q = parse_qs(url.query)
                    path = json.loads(q.get("path", ["[]"])[0])
                    view = q.get("view", ["plain"])[0]
                    body = json.dumps(analysis.children(path, view)).encode()
                    self._send(body, "application/json")
                elif url.path == "/api/search":
                    analysis.ensure()
                    q = parse_qs(url.query)
                    term = q.get("q", [""])[0]
                    view = q.get("view", ["plain"])[0]
                    body = json.dumps(analysis.search(term, view)).encode()
                    self._send(body, "application/json")
                else:
                    self.send_error(404)
            except BrokenPipeError:
                pass
            except Exception as e:
                self.send_error(500, str(e))

    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError:
        # requested port is busy — let the OS pick a free one
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"serving {analysis.scope} at {url}  (ctrl-c to stop)")
    if open_browser:
        # webbrowser uses `open` on macOS and xdg-open/$BROWSER on Linux;
        # if neither can open a window, the printed URL above is the fallback
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Billed token spend per tool/command in Claude Code sessions")
    ap.add_argument("files", nargs="*", help="explicit .jsonl transcript files")
    ap.add_argument("--project", help="substring match on project dir name")
    ap.add_argument("--all", action="store_true", help="all projects")
    ap.add_argument("--last", action="store_true", help="only the most recent session")
    ap.add_argument("--top", type=int, default=20, help="rows to show in text mode (default 20)")
    ap.add_argument("--serve", action="store_true",
                    help="serve the interactive report with lazy-loaded levels")
    ap.add_argument("--port", type=int, default=8765, help="port for --serve (default 8765)")
    ap.add_argument("--no-open", action="store_true", help="don't auto-open the browser")
    ap.add_argument("--html", nargs="?", const="token-spend.html", metavar="FILE",
                    help="write a static self-contained report instead (data embedded)")
    ap.add_argument("--json", nargs="?", const="token-spend.json", metavar="FILE",
                    help="write the full detailed report as JSON (use '-' for stdout)")
    args = ap.parse_args()

    def project_label(dirname):
        if "-code-" in dirname:
            return dirname.split("-code-", 1)[-1]
        home_slug = re.sub(r"[^A-Za-z0-9]", "-", str(Path.home()))
        return "~ (home)" if dirname == home_slug else dirname.lstrip("-")

    if args.files:
        jobs = [(Path(f), ()) for f in args.files]
        scope = f"{len(jobs)} file(s)"
    else:
        if args.all:
            dirs = find_project_dirs()
            scope = "all projects"
        elif args.project:
            dirs = find_project_dirs(args.project)
            if not dirs:
                sys.exit(f"error: no project dir matching '{args.project}'")
            scope = " + ".join(project_label(d.name) for d in dirs)
        else:
            d = cwd_project_dir()
            if not d:
                sys.exit("error: no transcripts for the current directory; use --project or --all")
            dirs = [d]
            scope = os.path.basename(os.getcwd())
        # multiple projects in scope -> group everything under a project level
        dirs = [d for d in dirs if any(d.glob("*.jsonl"))]
        multi = len(dirs) > 1
        jobs = [(f, (project_label(d.name),) if multi else ())
                for d in dirs for f in d.glob("*.jsonl")]
        if args.last:
            jobs = sorted(jobs, key=lambda j: j[0].stat().st_mtime)[-1:]
    if not jobs:
        sys.exit("error: no transcript files found")

    analysis = Analysis(jobs, scope)

    if args.serve:
        serve(analysis, args.port, open_browser=not args.no_open)
    elif args.html:
        write_static_html(analysis, args.html)
        print(f"report written to {Path(args.html).resolve()}")
    elif args.json:
        write_json_report(analysis, args.json)
    else:
        print_text_report(analysis, args.top)


if __name__ == "__main__":
    main()
