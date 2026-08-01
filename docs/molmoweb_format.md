# MolmoWeb-4B output format — spec

Extracted by reading `allenai/molmoweb` (cloned to `reference/molmoweb`, gitignored,
Apache 2.0). Every claim below cites `path:line` in that clone. Nothing here has
been copied into our source tree; this is a written description of the wire format.

Clone: `git clone --depth 1 https://github.com/allenai/molmoweb.git reference/molmoweb`

> **Headline:** the model emits **a single JSON object**, not a DSL or a
> `<point>` tag. There is no regex parser — the code calls `json.loads`.
> Coordinates are **percentages 0–100** of the **screenshot** dimensions.

---

## 1. Prompt template

The repo matches the GOAL / PREVIOUS STEPS / CURRENTLY ACTIVE PAGE structure, and
the `molmo_web_think` system prefix, exactly as described on the HF model card.

**Training-time template** — `train/olmo/models/molmo/data_formatter.py:152-169`,
selected when `style == "molmo_web_think"` (`data_formatter.py:2116-2119`):

```jinja
# GOAL
{{ task_description }}

# PREVIOUS STEPS
{% for action in past_actions: -%}
## Step {{ action['index'] }}
THOUGHT: {{ action['thought'] }}
ACTION: {{ action['action'] }}
{% endfor %}
# CURRENTLY ACTIVE PAGE
Page {{ page_index }}: {{ page_title }} | {{ page_url }}

# NEXT STEP

```

(The template string starts and ends with a newline — see the `Template("""\n...`
literal at `data_formatter.py:152-153` and the trailing blank line at `:166-168`.)

**Inference-time template** — `agent/multimodal_agent.py:42-59`, as
`USER_MSG_TEMPLATE`. **It is byte-identical to the training template.** No skew.

A sibling `MOLMOWEB_BASE_TEMPLATE` (`data_formatter.py:135-151`) is the same minus
the `THOUGHT:` line; it pairs with `style == "molmo_web_base"`, which emits an
action with no thought. Training sometimes mixes the two 50/50
(`train/olmo/data/web_datasets.py:1204-1205`).

### The system message prefix

`agent/multimodal_agent.py:229`:

```python
prompt = f"{self.system_message}: {user_message}"
```

`system_message` defaults to `"molmo_web_think"` (`agent/multimodal_agent.py:166`),
and is passed explicitly by both the client (`inference/client.py:101`) and the
benchmark runner (`benchmarks/evaluate.py:18`).

So the literal text sent to the model begins:

```
molmo_web_think: 
# GOAL
...
```

Note the `": "` separator and that the template's own leading newline follows it.
This is a *style tag*, not a chat system role — it is prepended to the user string
and, in the native backend, passed as `style=` to the preprocessor
(`agent/model_backends.py:217`).

### Sampling defaults

`temperature=0.7`, `top_p=0.8`, `max_new_tokens=1024`
(`agent/multimodal_agent.py:171-172`, `agent/model_backends.py:96`, `:149-155`).

---

## 2. Raw generation format

There is **no regex and no grammar**. `agent/multimodal_agent.py:261-281`:

```python
try:
    assert pred_text is not None
    pred_json: dict[str, Any] = json.loads(pred_text)
    if "action" in pred_json:
        action_json: dict[str, Any] = pred_json["action"]
        thought: str = pred_json.get("thought", "")
        action_desc = pred_json.get("action_description", None)
    elif "name" in pred_json:
        action_json = pred_json
        thought = ""
        action_desc = None
    else:
        raise ValueError(f"Expected 'action' or 'name' key in parsed JSON but didnt get it.")
except Exception as e:
    pred_json = dict(
        thought=f"Could not parse predicted action: {e}",
        action=dict(name="report_infeasible", infeasibility_reason=f"Unparseable model output: {pred_text}"),
    )
```

Accepted shapes:

| Shape | Meaning |
|---|---|
| `{"thought": str, "action": {...}}` | `molmo_web_think` — the normal case |
| `{"name": ..., ...}` (no `action` key) | `molmo_web_base` — bare action, `thought = ""` |
| anything else / invalid JSON | swallowed → `report_infeasible`, run continues |

An optional `"action_description"` key is used verbatim for logging if present,
truncated to 400 chars (`agent/multimodal_agent.py:266`, `:294`).

**Parse failure never raises.** It degrades to `report_infeasible` carrying the
unparseable text. Worth copying — a malformed generation shouldn't kill an episode.

### The generation is exactly `json.dumps(answer_dict)`

Training target construction, `train/olmo/data/web_datasets.py:1207-1216`:

```python
if effective_style == "molmo_web_think":
    answer_dict = {
        "thought": traj_step["action"]["action_output"]["thought"].strip(),
        "action": formatted_action,
    }
else:  # molmo_web_base
    answer_dict = {"action": formatted_action}

message = dict(
    answer=json.dumps(answer_dict, ensure_ascii=False),
    ...
)
```

So key order in the generation is `thought`, then `action`; and within `action`,
`name` first (`web_datasets.py:1041`), then the type-specific keys in the order
they are assigned (for a click: `x`, `y`, `button`, `click_type` —
`web_datasets.py:1063-1066`).

---

## 3. Action vocabulary

Authoritative list = what `convert_action_json_to_action_obj` accepts,
`agent/multimodal_agent.py:73-147`. `pct` marks a 0–100 percentage (see §4).

| `name` emitted | Arguments | Executed as |
|---|---|---|
| `click` / `dblclick` / `mouse_click` | `x` pct, `y` pct, `button` = `left`\|`right`\|`middle` (default `left`) | `MouseClick`; `click_type="double"` iff name is `dblclick` (`:88-94`) |
| `hover_at` | `x` pct, `y` pct, `duration` seconds (default 1.0) | `HoverAt` (`:95-100`) |
| `drag_and_drop` / `mouse_drag_and_drop` | `from_x`, `from_y`, `to_x`, `to_y` — all pct | `MouseDragAndDrop` (`:101-107`) |
| `scroll` | `delta_x` pct, `delta_y` pct, **signed** | `Scroll` (`:108-112`) |
| `scroll_at` | `x`, `y` pct + `delta_x`, `delta_y` pct | `ScrollAt` (`:113-119`) |
| `type` / `keyboard_type` | `text` | `KeyboardType` (`:120-121`) |
| `keypress` / `keyboard_press` | `key`, **must be in `ALLOWED_KEYS`** | `KeyboardPress` (`:122-127`) |
| `gemini_type_text_at` | `x`, `y`, `text`, `press_enter` (default `True`), `clear_before_typing` (default `True`) | `GeminiTypeTextAt` — **see the ÷10 quirk below** |
| `goto` | `url` | `Goto` (`:136-137`) |
| `send_msg_to_user` | `msg` (truncated to 1000 chars, `:288-289`) | `SendMsgToUser` (`:138-139`) — **this is the terminal action** |
| `browser_nav` | `nav_type` = `go_back`\|`new_tab`\|`tab_focus`, `index` (int, `-1` when unused) | `BrowserNav` (`:140-141`) |
| `noop` | `noop_reason` = `loading`\|`captcha`\|`unsupported_keypress`\|`retrying_after_api_error` | `Noop` — always sleeps 5 s (`NOOP_WAIT_MS`, `agent/actions.py:7`, `:142-143`) |
| `report_infeasible` | `infeasibility_reason` | `ReportInfeasible` (`:144-145`) |
| *anything else* | — | `ReportInfeasible("Unsupported action type: ...")` (`:146-147`) |

`ALLOWED_KEYS` (`agent/multimodal_agent.py:27-40`) — matched case-insensitively,
anything else becomes `report_infeasible` (`:124-127`):

```
Enter, Escape, Backspace, Tab, ArrowUp, ArrowDown, ArrowLeft, ArrowRight,
ControlOrMeta+a, ControlOrMeta+c, ControlOrMeta+v, F5
```

**Two `Click` classes exist — don't confuse them.** `agent/actions.py:10-36`
defines a `bid`-based `Click` for the *accessibility-tree* baselines
(`gemini_axtree`, `gpt_axtree`). MolmoWeb's `click` maps to `MouseClick`
(`agent/actions.py:39-57`), which is pixel-based. Only `MouseClick` is relevant.

### The `gemini_type_text_at` ÷10 quirk

`agent/multimodal_agent.py:283-286`, applied *before* percentage conversion:

```python
if action_json.get("name") == "gemini_type_text_at":
    if "x" in action_json and "y" in action_json:
        action_json["x"] = action_json["x"] / 10.0
        action_json["y"] = action_json["y"] / 10.0
```

That is a Gemini-CUA compatibility path (Gemini emits 0–1000; ÷10 → 0–100). It
is *not* MolmoWeb's convention, but it lives in the shared MolmoWeb code path, so
if MolmoWeb ever emits `gemini_type_text_at` its coordinates get divided by 10.
Treat this action as Gemini-only unless proven otherwise.

---

## 4. Coordinate convention — **percent of screenshot dimensions**

### Range: 0–100, one decimal place

Training-time normalization, `train/olmo/data/web_datasets.py:47-68`:

```python
def normalize_click_coords(x, y, image_w, image_h, upper_bound=100, num_digits=1):
    x = round(x / image_w * upper_bound, num_digits)
    y = round(y / image_h * upper_bound, num_digits)
    # add min and max clipping to ensure normalized coords are between 0 and upperbound
    x = max(0, min(x, upper_bound))
    y = max(0, min(y, upper_bound))
    return x, y
```

So `x = 50.2` means "50.2% across the image", i.e. a **float with one decimal**,
not an int, not 0–1, not 0–1000.

### Denominator: the SCREENSHOT, not the model's preprocessed image

This was the critical question, and the answer is unambiguous on both sides.

**Training** — `image_w, image_h` come from the raw screenshot file
(`train/olmo/data/web_datasets.py:1171-1174`):

```python
if "image_w" in traj_step and "image_h" in traj_step:
    image_w, image_h = traj_step["image_w"], traj_step["image_h"]
else:
    image_w, image_h = Image.open(image).size
```

...and are passed straight into `get_formatted_action(action_output, image_w, image_h)`
(`:1202`), which calls `normalize_click_coords(x, y, image_w, image_h)` (`:1062`).

**Inference** — `agent/multimodal_agent.py:62-70` and `:86`:

```python
def _pct_to_coord(pct: float, dim: int) -> float:
    """Convert percentage to pixel coordinate, clamped to [1, dim-2] so edge
    predictions still land inside the viewport."""
    px = round((pct / 100.0) * dim, 1)
    return max(1.0, min(px, dim - 2.0))

...
h, w = (screenshot.shape[:2] if screenshot is not None else (720, 1280))
```

`screenshot` is the numpy array handed to the model (`:291`). **The model's
internal preprocessing (tiling/resize inside the Molmo preprocessor) is never
part of the denominator.** Whatever resizing the vision tower does internally is
the model's problem; the contract is expressed against the image you send it.

### The image you send is forced to 1280×720

`inference/client.py:65-66` and `:155-161`:

```python
VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 720
...
h, w = obs["screenshot"].shape[:2]
if (w, h) != (self.VIEWPORT_WIDTH, self.VIEWPORT_HEIGHT):
    img = Image.fromarray(obs["screenshot"]).resize(
        (self.VIEWPORT_WIDTH, self.VIEWPORT_HEIGHT), Image.LANCZOS,
    )
    obs["screenshot"] = np.array(img)
```

Their env sets `viewport_size` (`utils/envs/browser_env.py:111`) but **never sets
`device_scale_factor`** — so on a HiDPI host the CDP screenshot
(`utils/envs/browser_env.py:61-69`) could come back at 2×. This resize is the
safety net that fixes it. We solve the same problem at the source by pinning
`device_scale_factor=1`; the resize is a belt-and-braces version of the same idea.

### Scroll deltas normalize differently — signed, and NOT clamped

`train/olmo/data/web_datasets.py:71-83`:

```python
def normalize_scroll_deltas(delta_x, delta_y, image_w, image_h, upper_bound=100, num_digits=1):
    def _normalize(delta, dim):
        if dim == 0:
            return 0.0  # avoid divide by zero
        normalized = abs(delta) / dim * upper_bound
        normalized = round(normalized, num_digits)
        return normalized if delta >= 0 else -normalized
```

At inference, deltas use `_pct_to_px` (`agent/multimodal_agent.py:62-63`) which
does **no clamping**, unlike `_pct_to_coord`. So:

- `delta_y: 100` = scroll down by exactly one viewport height (720 px)
- `delta_y: -50` = scroll up half a viewport (−360 px)
- values outside ±100 are legal and mean multi-viewport scrolls

### One train/inference asymmetry (harmless, but know it)

Training clamps positions to `[0, 100]` → `[0, dim]` px. Inference clamps to
`[1, dim-2]` px (`_pct_to_coord`). A predicted `0.0` becomes pixel `1.0`, and
`100.0` becomes `dim-2`. Deliberate — it stops edge predictions landing outside
the viewport — and sub-2px, so irrelevant in practice.

---

## 5. Thought / action separation

They are **two keys of one JSON object**, not delimited text. There is no
`<think>` tag, no `THOUGHT:` prefix, no split on newline.

- Generation: `{"thought": "...", "action": {...}}` (`web_datasets.py:1207-1211`)
- Parse: `pred_json.get("thought", "")` (`multimodal_agent.py:266`)
- `molmo_web_base` style omits `thought` entirely; the parser then falls back to
  the `"name"`-at-top-level shape with `thought = ""` (`multimodal_agent.py:268-271`)

The `THOUGHT:` prefix appears **only in the prompt**, when past steps are
rendered back in (§6) — never in the model's own output.

---

## 6. History formatting

`self.past_actions` entries are appended at `agent/multimodal_agent.py:304-313`
and rendered by `get_user_message` (`:214-225`) using the template's:

```jinja
## Step {{ action['index'] }}
THOUGHT: {{ action['thought'] }}
ACTION: {{ action['action'] }}
```

`action['action']` is the **raw action dict** (`action_json`), not a string
(`multimodal_agent.py:300`, `:311`). Jinja renders a Python dict via `str()`, so
history lines carry **Python repr with single quotes**, not JSON:

```
## Step 1
THOUGHT: I need to search for the product, so I will click the search box.
ACTION: {'name': 'click', 'x': 50.2, 'y': 16.7, 'button': 'left', 'click_type': 'single'}
```

Training does the same thing — `past_actions.append({**answer_dict, "index": step_idx})`
(`train/olmo/data/web_datasets.py:1302`), where `answer_dict["action"]` is the dict
`formatted_action`. **So this single-quoted repr is what the model was trained on.
Do not "fix" it to JSON.**

Windowing:

| Setting | `MultimodalAgent` default | Client override |
|---|---|---|
| `max_past_steps` | 3 (`multimodal_agent.py:180`) | **10** (`inference/client.py:103`) |
| `max_past_images` | 0 (`:181`) | 0 (`inference/client.py:104`) |

`max_past_images=0` means **only the current screenshot is sent** — history is
text-only. Slicing is `self.past_actions[-self.max_past_steps:]` (`:222`, `:242`).

`index` is 1-based at inference (`len(self.past_actions) + 1`, `:306`). At training
it is the trajectory's own step key (`step_idx`, `web_datasets.py:1283`, `:1302`),
whose base I could not confirm from the repo alone — the HF dataset is not vendored.
Minor: it only affects the `## Step N` label.

---

## 7. Task completion and the final answer

There is **no `done` action**. Completion is `send_msg_to_user` with a **prefix
convention in the `msg` string**.

`inference/client.py:194-198` — the client's terminal check:

```python
if isinstance(step.prediction.action, SendMsgToUser) and (
    step.prediction.action.msg.startswith("[EXIT]")
    or step.prediction.action.msg.startswith("[ANSWER]")
):
    return traj
```

The answer is carried **inline in `msg`, after the `[ANSWER] ` prefix**. The repo
strips it by string replacement — `demo.ipynb` cell 15:

```python
AUTHORS = last_step.prediction.action.msg.replace("[ANSWER] ", "").split(", ")
```

The convention is spelled out in the Gemini baseline's prompt
(`agent/gemini_axtree_agent.py:69-82`), and MolmoWeb reproduces it because its
training trajectories were distilled from those agents:

> ```
> send_msg_to_user(text='[ANSWER] <your single answer here>')
> send_msg_to_user(text='[EXIT]')
> ```
> - The `[ANSWER]` message must contain exactly ONE concise answer—no multiple
>   options, no explanations, no error messages.
> - The `[EXIT]` message must be sent as a SEPARATE action AFTER the answer,
>   containing ONLY `[EXIT]`.
> - Never combine `[ANSWER]` and `[EXIT]` in the same message.
> - For tasks that don't require a textual response, send
>   `send_msg_to_user(text='[ANSWER] Done')` then `send_msg_to_user(text='[EXIT]')`.

### ⚠ The two harnesses disagree about which prefix terminates

| Harness | Terminates on | Cite |
|---|---|---|
| Inference client | `[ANSWER]` **or** `[EXIT]` | `inference/client.py:194-198` |
| Benchmark episode runner | `[EXIT]` **only** | `utils/eval_utils/episode.py:136-142` |

So under the client the `[EXIT]` step usually never executes — `[ANSWER]` already
ended the run. Under the benchmark runner the agent must emit both. This is a real
inconsistency in their codebase, and a decision point for us: terminating on
`[ANSWER]` (client behaviour) is the right default for a Phase-1 loop, since it
saves a step and we have no benchmark harness expecting the two-step form.

**Failure signal:** `report_infeasible` with `infeasibility_reason`. Note it does
*not* terminate the loop in either harness — it just executes as a no-op
(`utils/envs/action_executor.py:102-103`) and the loop keeps stepping until
`max_steps`.

---

## 8. Verbatim examples

### 8a. Verbatim, from the repo — post-parse (`demo.ipynb`, cell outputs 3, 5, 11, 12)

These are `ActionOutput.to_str()` renderings, i.e. **after** percent→pixel
conversion. Real committed notebook output, quoted exactly:

```
[03:11:20] Step  1: goto(url='https://allenai.org')
[03:11:27] Step  2: mouse_click(x=640.0, y=270.0, button='left')
[03:11:31] Step  3: send_msg_to_user(text='[ANSWER] Done')
```

```
Thought: The goal was to navigate to allenai.org and open the top project on the homepage. I have already navigated to the site and clicked on 'MolmoBot', which is the top featured project. The task is now complete.
Action: send_msg_to_user(text='[ANSWER] Done')
```

```
[03:11:52] Step  1: mouse_click(x=643.8, y=465.8, button='left')
[03:11:58] Step  2: mouse_click(x=632.3, y=409.7, button='left')
```

Note `x=640.0, y=270.0` on a 1280×720 viewport = exactly `x: 50.0, y: 37.5` in
percent — consistent with §4. And `643.8` = 50.3% × 1280, showing the one-decimal
percentage surviving into a one-decimal pixel value.

### 8b. Reconstructed raw generation — **not** a repo quote

I could **not** find a literal raw generation string committed anywhere (the only
`print` of `pred_text` is commented out at `agent/multimodal_agent.py:249`, and
`raw_output` at `utils/eval_utils/episode.py:20` is only populated at runtime).

The following is reconstructed from the serialization code
(`json.dumps({"thought": ..., "action": formatted_action})`, `web_datasets.py:1207-1216`,
with key order from `web_datasets.py:1041,1063-1066`). It is what the format
*must* produce, not something copied out of the repo:

```json
{"thought": "The search box is at the top of the page. I will click it to focus it before typing the query.", "action": {"name": "click", "x": 50.2, "y": 16.7, "button": "left", "click_type": "single"}}
```

```json
{"thought": "The results are below the fold, so I will scroll down by about one viewport.", "action": {"name": "scroll", "delta_x": 0.0, "delta_y": 100.0}}
```

```json
{"thought": "The rating is visible in the table. I have the answer and will report it.", "action": {"name": "send_msg_to_user", "msg": "[ANSWER] Magnus Carlsen, 2839"}}
```

**Verify 8b against one real generation before trusting the key names.** Cheapest
check: uncomment `agent/multimodal_agent.py:249` in the clone, or POST to
`/predict` per `README.md:158-169` and print `resp.json()`.

---

## 9. Open questions

1. **A real raw generation.** Everything in §8b is inferred from serialization
   code. One captured `pred_text` would confirm key order, spacing and whether the
   model ever wraps output in a code fence.
2. **`step_idx` base at training time** (§6) — 0- or 1-based.
3. **Whether MolmoWeb ever emits `gemini_type_text_at`.** If it does, the ÷10 at
   `multimodal_agent.py:283-286` applies and its coordinates are *not* 0–100.
4. **`[ANSWER]` vs `[EXIT]`** — we should pick the client's behaviour (§7), but
   confirm the model reliably emits `[ANSWER]` first rather than a bare `[EXIT]`.
