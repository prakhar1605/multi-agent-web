# multi-agent-web — vision-language browser agents

**Phase 1** — the single-agent loop. **Phase 2** — the parallel substrate and
best-of-N. Phase 3 (a manager LLM decomposing goals into a DAG of subtasks) is
not built.

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

**Phase 2** adds the substrate for running many of those loops at once:

```
                    ┌── agent_0: own browser context + own policy ──┐
task ── strategy ───┼── agent_1: own browser context + own policy ──┼── judge ── answer
                    └── agent_2: own browser context + own policy ──┘
```

with two independent concurrency limits (browsers are local and memory-bound;
the model server is shared and GPU-bound), per-step timing that separates model
time from browser time, and one crashed agent never sinking the run.

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

# With the real model. The GPU can be on another machine; this is just a URL.
export MOLMOWEB_ENDPOINT=http://gpu-host:8001
python scripts/run_single.py --policy molmoweb --task "find the pricing page" \
    --start-url https://example.com
```

### Several agents on one task (Phase 2)

```bash
# Four agents race the same task; a judge picks the winner. MockPolicy default.
python scripts/run_multi.py --task "find the pricing page" --n 4

# The interesting case: browsers run in parallel, generations queue single-file.
python scripts/run_multi.py --task "..." --n 8 --max-browsers 8 --max-model 1
```

Output lands in `runs/<timestamp>/agent_0/`, `agent_1/`, … each keeping the
single-agent format above, plus a top-level `run.json`
([format](docs/run_json.md)) holding the task, per-agent outcome, judge decision
and the timing breakdown.

### Grounding gate

Before trusting the model — or building anything on top of it — check that a
predicted click actually lands on the element it names:

```bash
python scripts/check_grounding.py --endpoint http://gpu-host:8001 --trials 5
```

It loads a page with four separated buttons, asks for one by label, and reports
predicted percent → converted pixels → target rect → HIT/MISS. This isolates the
percent→pixel mapping, which is the part most likely to be silently wrong: a
convention that is off by a factor, an axis, or a device-scale-factor still
produces perfectly plausible JSON — it just misses. With no endpoint configured
it skips (exit 0), so it is safe without a GPU.

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
| `config.py` | `RunConfig` (browser) and `MolmoWebConfig` (model endpoint) |
| `actions.py` | Pydantic discriminated union of browser actions |
| `browser.py` | `BrowserSession`: async Playwright wrapper (the environment) |
| `policy/base.py` | `AgentPolicy` ABC — the swap point |
| `policy/mock.py` | `MockPolicy`: replays a scripted action list |
| `policy/molmoweb.py` | MolmoWeb-4B adapter: HTTP client + strict parser |
| `agent.py` | the ReAct loop |
| `trajectory.py` | `Step` + on-disk logging |
| `orchestrator/session.py` | one isolated agent: own context, own policy |
| `orchestrator/runner.py` | N sessions concurrently under two limits; `run.json` |
| `orchestrator/strategy/` | `Strategy` ABC + `BestOfN` |
| `orchestrator/judge/` | `Judge` ABC + deterministic `MockJudge` |
| `docs/molmoweb_format.md` | the model's wire format, verified against training data |
| `docs/run_json.md` | the multi-agent run format (stable; the UI reads it) |
| `scripts/check_grounding.py` | acceptance gate: does a predicted click land? |

Import graph is acyclic: `orchestrator → agent → policy → trajectory, browser →
actions → config`.

---

## Design decisions

**Actions carry absolute viewport pixels; the policy converts.**
`Click(x=640, y=360)` always means "CSS pixel 640,360 from the viewport's
top-left". MolmoWeb emits percentages 0–100 of the screenshot; other web agents
emit raw pixels, others emit accessibility-tree indices. Rather than bake one
model's convention into the shared action space, the action layer fixes one
convention and each policy adapter translates into it. Swapping the policy then
cannot silently change what a coordinate means. Full spec in the `actions.py`
module docstring.

Ai2 landed on the same split independently — their executor's docstring reads
"Each agent is responsible for converting its own coordinate system to pixels"
(`utils/envs/action_executor.py:1-6`).

**The MolmoWeb adapter fails loudly.** An unknown action name, an unexpected
key, a non-`left` mouse button, a malformed payload — all raise, with the full
raw generation in the message. Nothing is coerced, nothing falls back to a
default action. A wrong format assumption then shows up as a readable error on
step 1 instead of a mystery misclick on step 5. Actions the model supports but
we don't model yet (`hover_at`, `scroll_at`, `browser_nav`, …) raise a
`NotImplementedError` naming the action, and are logged so you can see which
ones the model actually reaches for.

The one exception is coordinate clamping to `[1, dim-2]`, which is *conversion*,
not repair: the reference implementation defines the percent→pixel mapping that
way, and real training targets contain clipped values like `"y": 100` that would
otherwise convert to exactly `dim` and be rejected as out-of-viewport.

**`Scroll` is a signed `(delta_x, delta_y)` vector, not direction + magnitude.**
MolmoWeb can scroll both axes in one action. A direction enum would force the
adapter to drop an axis, and a silently dropped axis surfaces later as an
unexplained model failure — which is exactly the no-repair principle being
violated. `Scroll.by("down", 600)` keeps hand-written scripts readable.

**The MolmoWeb policy keeps its own history in the model's coordinate space.**
The loop's `history` is pixel-space `Step`s; the prompt needs the model's own
percent-space dicts. Re-deriving percentages from our pixels would drift by a
rounding step every turn, so the adapter keeps a parallel `past_actions` list,
exactly as the reference `MultimodalAgent` does.

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

## The MolmoWeb wire format

Full spec with `path:line` citations: [`docs/molmoweb_format.md`](docs/molmoweb_format.md).
The short version — the model emits **one JSON object per step**:

```json
{"thought": "...", "action": {"name": "click", "x": 71.1, "y": 3.1, "button": "left", "click_type": "single"}}
```

- No regex, no DSL — the reference parses with `json.loads`.
- Coordinates are **percentages 0–100** (one decimal) of the **screenshot**
  dimensions, never of the model's internal preprocessed image.
- `delta_y: 100` on a scroll is exactly one viewport height.
- Completion is `send_msg_to_user` with an `[ANSWER]` or `[EXIT]` prefix — there
  is no `done` action. We terminate on either and record which fired on the step.
- Prompt history renders past actions as **Python dict repr** (single quotes),
  because that is what training used.

These strings are not guesses. They were verified by streaming rows from the
public `allenai/MolmoWeb-SyntheticTrajs` dataset and tracing them through the
reference serializer — the training target string *is* the generation format.
`tests/test_molmoweb_format.py` pins the parser and the prompt against them, and
runs with no GPU and no network.

## Phase 2 design decisions

**Isolation is by construction, not convention.** `Runner` takes a *policy
factory* (`Callable[[int], AgentPolicy]`), never a policy — there is no
parameter through which two agents could receive the same one. This matters
because policies are stateful: `MolmoWebPolicy` carries the `past_actions` it
renders into every prompt, so a shared instance would prompt each agent with
steps another agent took. That bug produces plausible-looking trajectories and
is near-impossible to spot afterwards. A factory that returns a cached instance
is caught by an identity check and reported as a crashed session.

**One browser process, N contexts.** A *context* is Playwright's isolation
unit — cookies, localStorage, IndexedDB and auth all live there. Two agents on
separate pages of one context would share a login; on separate contexts they
cannot. N contexts cost far less than N browser processes and isolate just as
well. `tests/test_orchestrator.py` proves it behaviourally: state written in one
session is invisible in another.

**Two concurrency limits, because there are two resources.** Browsers are local
and memory-bound; the model server is remote and GPU-bound. Conflating them
hides the interesting behaviour — with 8 browsers and 1 model slot, all 8 agents
browse in parallel and then line up single-file at the model. That is what
`model_queue_seconds` measures, and it is the headline plot: as N grows, queue
time rises roughly linearly while browser time stays flat, which is the point
where adding agents stops helping.

**Failures are results, not omissions.** A crashed agent becomes a
`SessionResult` with `status="crashed"`, is reported in `run.json`, and is still
handed to the judge — labelled. Filtering failures out before judging would make
"three of four agents crashed" indistinguishable from "one agent answered".

**The `Strategy` interface was designed for the second implementation.** A
strategy receives the `Runner` and decides what to launch and when, so a Phase 3
DAG can run a wave, read it, and build the next wave from the results —
`Runner.run_sessions` is re-entrant and keeps assigning fresh agent indices.
`contributing_indices` is a list, not a winner, so a DAG answer synthesised from
several agents needs no schema change to a format the UI already reads.

## Next (not built)

- **Phase 3**: a manager LLM that decomposes a goal into a DAG of subtasks, and
  an LLM judge. Both slot into existing interfaces — `Strategy` and `Judge` —
  without changes here.
- Run `scripts/check_grounding.py` against a live endpoint before trusting any
  of this with the real model. It is unrun so far: no hardware yet. It does not
  block Phase 2, because the orchestrator only ever talks to `AgentPolicy`.
