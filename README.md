# multi-agent-web

A research prototype for **multi-agent web browsing**: several vision-language
browser agents, driven by [MolmoWeb-4B](https://github.com/allenai/molmoweb),
working in parallel on one user goal. Each agent observes a screenshot, predicts
a browser action, and executes it — the system runs many such loops
concurrently under isolated browser contexts and picks the best result.

> **Status.** The full pipeline is verified end-to-end against a mock policy and
> a mock model server. **The MolmoWeb model itself has not been run** — no GPU
> access yet. No results in this repo come from live model inference. See
> [What is and isn't built](#what-is-and-isnt-built).

---

## The core loop

```
screenshot ──▶ policy.predict(task, history, screenshot, page) ──▶ action ──▶ browser ──┐
     ▲                                                                                  │
     └──────────────────────────────────────────────────────────────────────────────────┘
```

and N of those loops at once:

```
                    ┌── agent_0: own browser context + own policy ──┐
task ── strategy ───┼── agent_1: own browser context + own policy ──┼── judge ── answer
                    └── agent_2: own browser context + own policy ──┘
```

## Architecture

Three layers, each ignorant of the ones above it.

| Layer | What it does | Key type |
|---|---|---|
| **Policy** | Chooses the next action from a screenshot | `AgentPolicy` (ABC) |
| **Browser** | Executes actions; the environment | `BrowserSession` |
| **Orchestration** | Runs N agents concurrently, picks a winner | `Runner`, `Strategy`, `Judge` |

**Why this split.** The policy interface is one method —
`predict(task, history, screenshot, page) -> Step` — and nothing below it
imports a model runtime. That buys three things:

- The whole system is exercisable with `MockPolicy`: no GPU, no API key, no
  network. Every test in this repo runs that way.
- Model-specific quirks stay in one file. MolmoWeb emits percentage
  coordinates; a different model emitting pixels or element IDs is a new
  adapter, not a refactor.
- The orchestrator only ever talks to `AgentPolicy`, so parallel execution was
  built and tested without the model existing.

```
multi_agent_web/
  actions.py        Pydantic discriminated union of browser actions
  browser.py        async Playwright wrapper (the environment)
  agent.py          the single-agent loop
  trajectory.py     per-step logging to disk
  policy/
    base.py         AgentPolicy ABC — the swap point
    mock.py         scripted policy, for testing without a model
    molmoweb.py     MolmoWeb-4B adapter: HTTP client + strict parser
  orchestrator/
    session.py      one isolated agent: own browser context, own policy
    runner.py       N sessions concurrently under two limits; run.json
    strategy/       Strategy ABC + BestOfN
    judge/          Judge ABC + deterministic MockJudge
```

## What is and isn't built

**Built and verified without a model:**

- Single-agent loop, trajectory logging, run artifacts
- MolmoWeb wire-format adapter — prompt construction, strict JSON parsing,
  percent→pixel conversion. Pinned by tests against **real generations traced
  from Ai2's public training data**, byte-for-byte.
- HTTP transport, exercised end-to-end against a fake server implementing the
  reference server's contract
- Parallel orchestration: isolated contexts, two concurrency limits, per-step
  model-vs-browser timing, crash containment, best-of-N with a deterministic
  judge

**Not built:**

- Manager LLM and DAG decomposition of a goal into subtasks
- LLM judge (the interface has a documented seam; `MockJudge` fills it)
- Any UI

**Built but never run against the real model:**

- `scripts/check_grounding.py` — the acceptance gate. It asks the model to click
  a labelled button and asserts the predicted click lands inside that button's
  rectangle. Until it passes against a live endpoint, treat every claim about
  MolmoWeb's behaviour here as *derived from its published code and training
  data*, not observed.

## Install

```bash
python -m pip install -r requirements.txt
python -m playwright install chromium
```

## Run

```bash
pytest        # 43 pass; 1 skips unless a model endpoint is configured

# One agent, mock policy, bundled local page — zero network.
python scripts/run_single.py --task "type a query on the demo page"
python scripts/run_single.py --task "..." --headed --slow-mo 300

# Four agents race the same task; a judge picks the winner.
python scripts/run_multi.py --task "find the pricing page" --n 4

# Browsers parallel, generations queued single-file — the interesting case.
python scripts/run_multi.py --task "..." --n 8 --max-browsers 8 --max-model 1
```

With a model server (not yet exercised — no hardware):

```bash
export MOLMOWEB_ENDPOINT=http://gpu-host:8001   # may be a different machine
python scripts/check_grounding.py --trials 5    # run this FIRST
python scripts/run_multi.py --task "..." --n 4 --policy molmoweb
```

Output goes to `runs/<timestamp>/agent_0/`, `agent_1/`, … each with a screenshot
per step and a `steps.jsonl`, plus a top-level `run.json`
([format](docs/run_json.md)) holding per-agent outcomes, the judge's decision
and the timing breakdown.

## Design decisions

**Absolute viewport pixels in the action layer; conversion in the adapter.**
`Click(x=640, y=360)` always means the same thing. MolmoWeb emits percentages
0–100, other agents emit raw pixels or accessibility-tree indices — so each
adapter converts into one fixed convention rather than the convention bending
per model. Swapping policies then cannot silently change what a coordinate
means. Ai2's own executor makes the same split.

**`device_scale_factor = 1`.** On a HiDPI display the default is 2: a 1280×720
viewport screenshots at 2560×1440 and every predicted click lands at half its
intended position — a bug that reads as a bad model rather than a bad config.
Pinning it makes one screenshot pixel exactly one action pixel.

**Policy *factories*, not policy instances.** The orchestrator takes
`Callable[[int], AgentPolicy]`, so two agents sharing one policy is not
expressible. This matters because policies are stateful: the MolmoWeb adapter
carries the action history it renders into every prompt, and a shared instance
would prompt each agent with steps another agent took — producing plausible
trajectories that are nearly impossible to debug afterwards.

**Separate browser and model concurrency limits.** Browsers are local and
memory-bound; the model server is shared and usually GPU-bound. Conflating them
hides the behaviour that matters — with 8 browsers and 1 model slot, agents
browse in parallel then queue single-file at the model. Each step records model
time, model *queue* time, and browser time separately, so where wall-clock goes
as N scales is a property of the logs rather than an afterthought.

**The parser fails loudly.** An unknown action name, an unexpected key, a
non-left mouse button — all raise, with the full raw generation in the message.
Nothing is coerced, nothing falls back to a default action. A wrong assumption
about the model's output surfaces as a readable error on step 1 instead of a
mystery misclick on step 5. Actions MolmoWeb supports that this action space
doesn't model yet raise by name and are logged, so it's visible which ones the
model actually reaches for.

## The format spec

**[`docs/molmoweb_format.md`](docs/molmoweb_format.md)** — a complete, verified
specification of MolmoWeb-4B's wire format: prompt template, action vocabulary,
coordinate convention, termination signalling, and how history is fed back in.

Every claim cites `path:line` in Ai2's repository. The raw generation format was
verified rather than guessed: the model's training target string *is* its
generation format by construction, so rows were streamed from the public
`allenai/MolmoWeb-SyntheticTrajs` dataset and traced through the reference
serializer to recover exact output strings. That process settled several things
reading alone could not — that clipped coordinates serialize as `int` rather
than `float`, that a scroll delta of 100 is exactly one viewport, and that
prompt history uses Python `repr` rather than JSON.

It is the strongest artifact here, and it is what the adapter's tests are pinned
against.

## Next

1. **Run the grounding gate** against a live endpoint. Until a predicted click
   provably lands on its target, nothing above the adapter is trustworthy.
2. **Manager LLM + DAG decomposition** — decompose a goal into subtasks with
   dependencies and schedule them in waves. Slots in as another `Strategy`; the
   interface and `run.json` were shaped to accept it without changes.
3. **LLM judge**, replacing `MockJudge` at the documented seam.
4. **UI** for replaying trajectories from `run.json` side by side.

## Credits

The format specification in `docs/molmoweb_format.md` was derived by reading
[**allenai/molmoweb**](https://github.com/allenai/molmoweb) (Ai2), which is
licensed under **Apache License 2.0**, together with the public
`allenai/MolmoWeb-SyntheticTrajs` dataset. No code was copied from it; the
adapter here is an independent implementation of the same wire format against a
different action space, and the files that derive from reading it carry
provenance comments.

MolmoWeb-4B and the Molmo family are Ai2's work.
