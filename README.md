# multi-agent-web

A research prototype for **multi-agent web browsing**: several vision-language
browser agents, driven by [MolmoWeb-4B](https://github.com/allenai/molmoweb),
working in parallel on one user goal. Each agent observes a screenshot, predicts
a browser action, and executes it — the system runs many such loops
concurrently under isolated browser contexts and picks the best result.

> **Status.** The pipeline runs end-to-end against a **live vision-language
> model** (Qwen3.5-27B over an OpenAI-compatible API): 100% on the grounding
> gate, and multi-agent best-of-N completing real browsing tasks. The
> **MolmoWeb adapter has never been run** — no GPU access — so every claim
> about MolmoWeb here is derived from its published code and training data, not
> observed. See [What is and isn't built](#what-is-and-isnt-built).

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
imports a model runtime. Three adapters now sit behind it, and they are as
different as adapters get: a scripted mock, a purpose-trained web agent with its
own output grammar (MolmoWeb, percent coordinates), and a general VLM told to
emit our schema (Qwen, 0–1000 normalized points over a metered HTTP API).
Adding the third required no change to the browser layer, the agent loop, or the
orchestrator. That buys three things:

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
    molmoweb.py     MolmoWeb-4B adapter: percent coords, self-hosted server
    qwen.py         Qwen VLM adapter: normalized points, metered API, budget
  orchestrator/
    session.py      one isolated agent: own browser context, own policy
    runner.py       N sessions concurrently under two limits; run.json
    strategy/       Strategy ABC + BestOfN
    judge/          Judge ABC + deterministic MockJudge
```

## What is and isn't built

**Verified against a live model (Qwen3.5-27B):**

- **Grounding gate: 6/6.** Six targets — top edge, mid-screen, bottom band, a
  34px icon, and one between two near-identical twins — every predicted click
  landed inside its target rectangle. Annotated PNGs are written per target.
- **Multi-agent best-of-N completing a real task.** Two agents independently
  clicked a search field, typed, submitted, read the result back, and reported
  it; the judge picked a winner. 8 API calls, 14.5k tokens.
- First real timing data: **model 64.1s vs browser 3.5s** in that run, an 18:1
  ratio. Adding agents does not help once the model saturates, which is exactly
  what the two separate concurrency limits exist to expose.

**Built and verified without a model:**

- Single-agent loop, trajectory logging, run artifacts
- MolmoWeb wire-format adapter — prompt construction, strict JSON parsing,
  percent→pixel conversion. Pinned by tests against **real generations traced
  from Ai2's public training data**, byte-for-byte, and exercised end-to-end
  against a fake server implementing the reference server's contract.
- Parallel orchestration: isolated contexts, two concurrency limits, per-step
  model-vs-browser timing, crash containment, best-of-N with a deterministic
  judge
- Manager LLM and DAG decomposition: plan validation (cycles, dangling
  dependencies and unstartable graphs are rejected, never repaired), wave
  execution with results passed down dependency edges, budgeted replanning, and
  DAG-growth/replan-rate reporting. Driven in tests by a stub client, so the
  whole planning path is exercised with no API key and no spend.
- LLM judge, filling the seam `judge/base.py` documented. Opt-in;
  `MockJudge` stays the default.

**Not built:**

- Set-of-marks prompting — deliberately not built. The gate result says raw
  coordinate grounding is sufficient on this class of page, so SoM would be
  complexity without evidence.
- Any UI

**Built but never run:**

- `policy/molmoweb.py`. No GPU access. Treat every claim about MolmoWeb's
  behaviour as *derived from its published code and training data*, not
  observed. The gate is policy-agnostic and will validate it in one command
  when hardware appears.

## Install

```bash
python -m pip install -r requirements.txt
python -m playwright install chromium
```

## Run

```bash
pytest        # 132 pass; 1 skips unless a model endpoint is configured

# One agent, mock policy, bundled local page — zero network.
python scripts/run_single.py --task "type a query on the demo page"
python scripts/run_single.py --task "..." --headed --slow-mo 300

# Four agents race the same task; a judge picks the winner.
python scripts/run_multi.py --task "find the pricing page" --n 4

# Browsers parallel, generations queued single-file — the interesting case.
python scripts/run_multi.py --task "..." --n 8 --max-browsers 8 --max-model 1
```

With a live model. Put `PPAPI_KEY` and `PPAPI_BASE_URL` in `.env` (gitignored):

```bash
python scripts/smoke_api.py                       # verify the API contract
python scripts/check_grounding.py --policy qwen   # run this FIRST
python scripts/run_multi.py --task "..." --n 4 --policy qwen --max-calls 60

# A manager LLM decomposes the task into a dependency graph and runs it in
# waves, passing each subtask's finding down to the subtasks that depend on it.
python scripts/run_multi.py --task "..." --strategy dag --policy qwen

# The planning ablation: decompose, then never revise. Same code path.
python scripts/run_multi.py --task "..." --strategy dag --policy qwen \
    --planning-budget 0

# An LLM judge instead of the deterministic one (best-of-N only).
python scripts/run_multi.py --task "..." --n 4 --policy qwen --judge llm
```

`--strategy dag` needs a key even with `--policy mock`, because the manager is
an LLM whatever the agents are — which makes `--policy mock --strategy dag` a
cheap way to inspect a decomposition (one or two calls) without paying for the
browsing.

`--max-calls` is a per-run ceiling shared across agents; the run aborts rather
than overspending, and actual usage is written to `run.json`. The grounding gate
skips (exit 0) when nothing is configured, and the live test in the suite only
runs under `GROUNDING_LIVE=1` — a `pytest` that silently bills a metered key is
a trap.

For MolmoWeb against a self-hosted server (never exercised — no hardware):

```bash
export MOLMOWEB_ENDPOINT=http://gpu-host:8001
python scripts/check_grounding.py --policy molmoweb
```

Output goes to `runs/<timestamp>/agent_0/`, `agent_1/`, … each with a screenshot
per step and a `steps.jsonl`, plus a top-level `run.json`
([format](docs/run_json.md)) holding per-agent outcomes, the judge's decision
and the timing breakdown. A `dag` run adds the initial and final graphs, every
replan with its reason, and the planning budget consumed.

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

**Meet the model's prior; don't fight it.** The Qwen adapter was first written to
ask for absolute pixel coordinates, on the reasoning that the model sees the
screenshot at full size. The grounding gate scored **0/6** — but every failure
was a *schema* error, not a miss. The model was cramming a coordinate pair into
the `x` field, even breaking JSON to do it. Scoring the numbers it emitted
against the targets showed all six matched the true centres to within 0.6%
*after dividing by 1000*: Qwen-VL is trained to emit points on a 0–1000 grid,
and that prior beats a system prompt. Switching the wire format to normalized
`[x, y]` pairs and converting in the adapter took the gate to **6/6**. The gate
paid for itself on its first run — a units bug that would otherwise have
surfaced as "the model is bad at clicking".

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

1. **Harder grounding pages.** 6/6 on a clean synthetic layout is a floor, not a
   ceiling. Real sites have dense nav bars, overlapping z-indexes and ambiguous
   labels. Set-of-marks stays unbuilt until a page defeats raw coordinates.
2. **Scaling study** — model-vs-browser time as N grows, now that there is a
   real 18:1 measurement to extrapolate from.
3. **Planning ablation** — `--planning-budget 0` against the default 10 on the
   same tasks. The budget is a config value precisely so the two runs differ in
   nothing else; `growth.replan_rate` and `growth.net_growth` in `run.json` are
   the numbers to compare.
4. **A stronger manager.** `--manager-model` is the knob MACU found mattered
   most, and it is one flag: the manager is a single model with two hooks and no
   other responsibilities, so swapping it changes planning quality and nothing
   else.
5. **UI** for replaying trajectories from `run.json` side by side.

## Credits

The format specification in `docs/molmoweb_format.md` was derived by reading
[**allenai/molmoweb**](https://github.com/allenai/molmoweb) (Ai2), which is
licensed under **Apache License 2.0**, together with the public
`allenai/MolmoWeb-SyntheticTrajs` dataset. No code was copied from it; the
adapter here is an independent implementation of the same wire format against a
different action space, and the files that derive from reading it carry
provenance comments.

MolmoWeb-4B and the Molmo family are Ai2's work.
