# tokenspend

Where do your billed Claude Code tokens go — per project, tool, command, file.
Parses `~/.claude/projects/**/*.jsonl` locally. Python 3 stdlib only, nothing leaves your machine.

## Run

```bash
# interactive report (macOS/Linux, opens your browser)
curl -fsSL https://raw.githubusercontent.com/taranek/tokenspend/master/token-spend.py | python3 - --all --serve

# detailed JSON report
curl -fsSL https://raw.githubusercontent.com/taranek/tokenspend/master/token-spend.py | python3 - --all --json tokenspend.json
```

## Flags

```bash
./token-spend.py                      # text report for the project in $PWD
./token-spend.py --all                # every project
./token-spend.py --project loco       # project dir matching substring
./token-spend.py --last               # most recent session only
./token-spend.py --serve              # interactive web report (lazy-loaded drill-down, search)
./token-spend.py --serve --port 9000  # custom port (falls back to a free one if busy)
./token-spend.py --serve --no-open    # don't auto-open the browser
./token-spend.py --html report.html   # static self-contained export
./token-spend.py --json -             # JSON to stdout, pipe into jq
./token-spend.py --top 30             # more rows in text mode
./token-spend.py path/to/session.jsonl  # explicit transcript files
```

## How

- Real billed usage from transcript `usage` fields: per API call, `input + cache_creation` minus the previous call's `output` ≈ the tool results injected in between.
- Session start is itemized: CLAUDE.md and memory rows estimated from current file sizes; remainder is the base system prompt.
- Cache reads (~0.1× re-reads of history on every call) are shown in totals, not attributed per node.
