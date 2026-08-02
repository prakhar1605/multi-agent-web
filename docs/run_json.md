# `run.json` — the multi-agent run format

**Stable format.** The UI and the demo tooling both read this. Add fields rather
than renaming or removing them; bump `RUN_JSON_SCHEMA_VERSION`
(`multi_agent_web/orchestrator/runner.py`) if the shape changes incompatibly,
and check `schema_version` before parsing.

## Directory layout

```
runs/20260801-195032/
  run.json          <- this file: the whole multi-agent run
  agent_0/          <- one directory per agent, Phase 1 format, UNCHANGED
    meta.json
    steps.jsonl
    step_000.png
    ...
    final.png
    summary.json
  agent_1/
  agent_2/
```

Per-agent directories are byte-identical in structure to a single-agent run, so
anything that reads a Phase 1 run directory keeps working on each agent inside a
multi-agent one. `agents[i].dir` is the directory name, relative to `run.json`.

A crashed agent may have no directory at all (if it failed before its first
step); `dir` is then `null`. Always check it before joining a path.

## Top level

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | Currently `1`. Check before parsing. |
| `task` | string | The goal, as given by the user. |
| `strategy` | string | `"best_of_n"` today; `"dag"` in Phase 3. |
| `started_at` / `finished_at` | ISO-8601 UTC | Wall-clock bounds of the whole run. |
| `answer` | string \| null | The reported answer. **`null` is a legitimate outcome** — every agent may have failed. |
| `contributing_agents` | int[] | Indices whose work the answer rests on. Exactly one for best-of-N; several for a Phase 3 DAG. Empty when there is no answer. |
| `reason` | string | Why this answer — the judge's rationale, or an explanation of failure. Always populated. |
| `details` | object | Strategy-specific. Shape depends on `strategy`; see below. |
| `agents` | object[] | One per agent launched, in index order. Never filtered. |
| `timing` | object | Where wall-clock went. |
| `usage` | object \| null | API spend, for metered policies. `null` for local ones. |

## `agents[]`

| Field | Type | Meaning |
|---|---|---|
| `index` | int | Agent index, matching `agent_<index>/`. Unique within a run. |
| `dir` | string \| null | Directory name, or `null` if the agent produced none. |
| `task` | string | This agent's task. Same as the top-level task for best-of-N; a subtask in Phase 3. |
| `label` | string \| null | Human-readable role, e.g. `"candidate 2/4"`. |
| `status` | enum | `done` \| `max_steps` \| `aborted` \| `crashed`. See below. |
| `answer` | string \| null | Only meaningful when `status == "done"`. |
| `num_steps` | int | Steps recorded. |
| `num_errors` | int | Steps whose action failed but the run continued. |
| `error` | string \| null | Set when `status == "crashed"`. |
| `timing` | object | `wall_seconds`, `model_seconds`, `model_queue_seconds`, `browser_seconds`. |

### `status`

| Value | Meaning |
|---|---|
| `done` | The agent emitted `Done` — it decided it was finished. The only status whose `answer` should be trusted. |
| `max_steps` | Ran out of step budget without deciding it was finished. May still hold a partial answer. |
| `aborted` | Stopped after `max_consecutive_errors` back-to-back failures. |
| `crashed` | Raised before producing a result. `error` says what happened. |

**Failed agents are never dropped.** A run where three of four agents crashed is
a valid, reportable outcome, and it must be distinguishable from a run where one
agent answered. Anything consuming this file should expect mixed statuses.

## `timing`

Model and browser time are tracked separately because they are different
resources: the model is a shared, usually GPU-bound queue, while browsers are
local and parallelise freely.

| Field | Meaning |
|---|---|
| `wall_seconds` | Real elapsed time for the whole run. |
| `total_model_seconds` | Summed across agents: time inside `policy.predict()`, queueing included. |
| `total_model_queue_seconds` | Of the above, time spent **waiting for a model slot** rather than generating. |
| `total_browser_seconds` | Summed across agents: screenshotting, page info, and executing actions. |
| `sum_agent_wall_seconds` | Summed per-agent wall-clock — what a serial run would have cost. |
| `max_agent_wall_seconds` | The slowest single agent — the floor on `wall_seconds`. |
| `peak_concurrent_browsers` | Highest number of contexts live at once. |
| `peak_inflight_model_requests` | Highest number of generations in flight at once. |
| `speedup_vs_serial` | `sum_agent_wall_seconds / wall_seconds`. Concurrency actually achieved: `1.0` means fully serialized, `N` means perfect overlap across N agents. |

`total_model_queue_seconds` is the headline number as N scales. With one GPU it
grows roughly linearly in N while `total_browser_seconds` stays flat — the plot
that shows adding agents stops helping once the model saturates. Note the
`total_*` fields sum across agents and so exceed `wall_seconds` whenever
anything ran in parallel; that is expected, not a bug.

## `usage`

Present when the policy talks to a metered API (`qwen`), `null` otherwise
(`mock`, `molmoweb` against a self-hosted server). The budget is **shared across
all agents in a run**, so these are run totals, not per-agent.

```json
{
  "calls": 8, "limit": 20, "retries": 0,
  "prompt_tokens": 13989, "completion_tokens": 466,
  "reasoning_tokens": 0, "image_tokens": 7056, "total_tokens": 14455,
  "calls_by_agent": {"0": 4, "1": 4}
}
```

`calls` counts every HTTP attempt including retries, because a retried call is
still a billed call. Reaching `limit` aborts the run deliberately rather than
continuing to spend. `image_tokens` is usually the dominant cost — roughly half
the prompt tokens in the example above — which is why history is text-only and
only the current screenshot is sent.

**This object never contains credentials.** The API key is held as a Pydantic
`SecretStr` and is not part of anything serialized here.

## `details`, by strategy

### `best_of_n`

```json
{
  "n": 4,
  "judge": {"name": "mock", "winner": 1, "reason": "..."},
  "candidates": [
    {"index": 0, "answer": "...", "status": "done", "num_steps": 4,
     "num_errors": 0, "error": null, "label": "candidate 1/4"}
  ]
}
```

`judge.winner` is `null` when no candidate was acceptable. `candidates` contains
**every** agent, failures included — that is what the judge saw.

### `dag` (Phase 3, not built)

Will carry the subtask graph. `contributing_agents` is already a list so that
an answer synthesised from several leaf agents needs no schema change.
