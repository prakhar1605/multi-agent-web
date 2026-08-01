# multi-agent-web — Phase 1: the single-agent foundation

A vision-language browser agent loop:

```
screenshot ──▶ policy.predict(task, history, screenshot, page) ──▶ action ──▶ browser ──┐
     ▲                                                                                  │
     └──────────────────────────────────────────────────────────────────────────────────┘
```

**Phase 1 is only this loop.** No manager LLM, no orchestration, no UI. The
point is to nail down the environment (browser + action space) and the policy
interface, so that swapping in MolmoWeb-4B — and later running several of these
agents in parallel — is a drop-in change rather than a rewrite.

---

## Setup

```bash
python -m pip install -r requirements.txt
python -m playwright install chromium     # downloads the browser binary (~150 MB)
```

## Run

```bash
# Zero network, zero GPU: MockPolicy against the bundled demo page.
python scripts/run_single.py --task "type a query on the demo page"

# Watch it happen.
python scripts/run_single.py --task "..." --headed --slow-mo 300

# Any site.
python scripts/run_single.py --task "find the pricing page" --start-url https://example.com
```

Output lands in `runs/<timestamp>/`:

```
runs/20260801-143012/
  meta.json        task, config, start time
  step_000.png     what the policy SAW before choosing step 0
  step_001.png
  final.png        the page after the last action
  steps.jsonl      {thought, action, url, title, screenshot, error, timestamp}
  summary.json     status, answer, step count
```

## Test

```bash
pytest -q
```

Runs the full loop headless against `tests/fixtures/demo_page.html` over
`file://` and asserts the trajectory was written to disk.

---

## Layout

| File | Role |
|---|---|
| `config.py` | `RunConfig`: viewport, step budget, timeouts, run dir |
| `actions.py` | Pydantic discriminated union of browser actions |
| `browser.py` | `BrowserSession`: async Playwright wrapper (the environment) |
| `policy/base.py` | `AgentPolicy` ABC — the swap point |
| `policy/mock.py` | `MockPolicy`: replays a scripted action list |
| `policy/molmoweb.py` | **stub**, raises `NotImplementedError` |
| `agent.py` | the ReAct loop |
| `trajectory.py` | `Step` + on-disk logging |

Import graph is acyclic: `agent → policy → trajectory, browser → actions → config`.

---

## Design decisions

**Actions carry absolute viewport pixels; the policy converts.**
`Click(x=640, y=360)` always means "CSS pixel 640,360 from the viewport's
top-left". Molmo-family models emit *normalized* points (usually 0–100), other
web agents emit raw pixels, others emit accessibility-tree indices. Rather than
bake one model's convention into the shared action space, the action layer fixes
one convention and each policy adapter translates into it. Swapping the policy
then cannot silently change what a coordinate means. Full spec in the
`actions.py` module docstring.

**`device_scale_factor = 1`.** On a Retina display the default is 2, the
screenshot comes back at 2560×1440 for a 1280×720 viewport, and every predicted
click lands at half its intended position — a bug that looks like a bad model
rather than a bad config. Pinning it to 1 makes one screenshot pixel exactly one
action pixel.

**`predict()` is `async`.** Phase 2 runs several agents concurrently; a
blocking `predict` would serialize them on the event loop. Free now, expensive
to retrofit. (In-process GPU inference should still be wrapped in
`asyncio.to_thread`.)

**The screenshot logged with a step is the one taken *before* the action.**
That is the (observation, action) pairing you need for evaluation or training.
Logging the after-screenshot instead would look fine in a demo and be useless as
data.

**`Step` lives in `trajectory.py`, not `agent.py`.** Both the policy and the
logger need it; putting it in `agent.py` would make `agent → policy → agent` a
circular import.

**Failures are recorded, not raised.** A failed click sets `step.error` and the
loop continues — flaky pages shouldn't discard a whole run. The one exception is
`max_consecutive_errors` (default 3) back-to-back failures, which means the
agent is genuinely stuck and further steps would just burn the budget.

**Trajectories are written incrementally.** If a run dies at step 9 of 15, the
first nine steps and their screenshots are already on disk — exactly the runs
you most want to debug.

**The environment does no repair.** No retries, no "the model probably meant the
button 20px away". A dumb environment makes policy failures show up in the
trajectory as failures, which is what you want when evaluating a policy.

---

## Why `policy/molmoweb.py` is empty

It raises `NotImplementedError` on purpose. Implementing it needs the model's
exact output format, and a guessed parser would look plausible while
mis-grounding every click — worse than no parser, because it fails silently.
The file lists the six things needed to finish it (output format, coordinate
convention, image preprocessing, prompt template, action vocabulary, serving
mode). All of them are answerable in that one file; nothing else changes.

## Next (not built)

- **Phase 1b**: implement `MolmoWebPolicy` once the format is known.
- **Phase 2**: a manager LLM that decomposes a goal into subtasks, one
  `BrowserSession` + `Agent` per subtask running concurrently, and a result
  aggregator. `BrowserSession` already owns no global state and each agent gets
  its own browser context, so this is additive.
