# tokenspend

Where do your billed Claude Code tokens actually go — per tool, per CLI command, per file?

`tokenspend` parses your local Claude Code session transcripts (`~/.claude/projects/**/*.jsonl`)
and attributes **real billed token usage** (from the API's own `usage` fields, not estimates)
to a drillable hierarchy:

```
project → tool → command → argument
loco    → Bash: git → add → -A
loco    → Edit → ~/code/loco → web → src → App.tsx
blog    → (session start) → CLAUDE.md
```

## Quick start

No install, no dependencies (Python 3 stdlib only) — run it straight from GitHub:

```bash
curl -fsSL https://raw.githubusercontent.com/taranek/tokenspend/master/token-spend.py | python3 - --all --serve
```

This parses all your Claude Code projects and opens the interactive report in your browser.

## Usage

```bash
./token-spend.py                    # text report: project inferred from $PWD
./token-spend.py --serve            # interactive report at http://127.0.0.1:8765
./token-spend.py --all --serve      # every project, lazy-loaded drill-down
./token-spend.py --project loco     # match project dir by substring
./token-spend.py --last             # most recent session only
./token-spend.py --html report.html # static self-contained export (data embedded)
```

No dependencies — Python 3 stdlib only. The interactive report is a single dark,
Linear-styled page: stacked in/out bars, breadcrumbs, an `args / by directory`
toggle that regroups path-like arguments into a drillable directory tree, and
lazy per-level loading served by a tiny built-in HTTP server.

## How attribution works

Each assistant message in a transcript carries the exact usage the API billed for
that call. `input_tokens + cache_creation_input_tokens` is the **new** prompt
content tokenized on that call. The only new content since the previous call is
(a) the previous call's own output — known exactly via its `output_tokens` — and
(b) the tool results and messages injected in between. So:

```
tool-result tokens ≈ new_input − previous_output
```

measured by Anthropic's own tokenizer, split proportionally by size when several
tool results land between two calls. Output tokens are likewise split across each
call's text vs tool-call blocks, so the cost of *writing* a large `Write` or
`Edit` is attributed too.

The **(session start)** bucket is itemized the same way: the first call's new
input covers the system prompt, CLAUDE.md, memory, attachments, and the first
user message. Known pieces are sized from the transcript; CLAUDE.md and
`MEMORY.md` rows are estimated from the files as they exist on disk today, and
the remainder is labeled "base system prompt".

### Caveats

- Cache reads (re-reading history on every call at ~0.1× price) are shown in the
  totals but not attributed per-tool — note that early-session content is
  re-read hundreds of times, so session-start tokens are the most amplified.
- Session-start CLAUDE.md/memory rows use current file sizes (chars ÷ 4), not
  the sizes at the time the sessions ran.
- Skill payloads injected as user-role turns show up under "(user messages)".
