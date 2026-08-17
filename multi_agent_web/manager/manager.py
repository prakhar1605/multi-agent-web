"""The manager LLM: decompose a goal into a graph, then revise it as work lands.

TWO HOOKS, ONE MODEL
====================
Following MACU, the manager is a single model with two jobs:

    decompose(task)              -> DAG
    replan(dag, outcomes)        -> Replan | None

and nothing else. It never browses, never judges, and never writes the final
answer. That narrowness is what makes the "stronger manager" knob meaningful:
swapping the model changes planning quality and nothing else, so an ablation
measures what it claims to measure. ``ManagerConfig.model`` is the knob, and it
defaults to the same endpoint the policy uses so the default run needs no extra
configuration.

THE PLANNING BUDGET IS A CONFIG VALUE, NOT A CODE PATH
======================================================
``planning_budget`` (default 10, as in the paper) counts DAG *edits*, and
``B = 0`` is a first-class setting rather than a separate mode: the initial
decomposition still runs, and ``replan`` returns ``None`` without calling the
model. So "no replanning" is one flag away from "replanning", the two paths
share every line of code between them, and the difference between the runs is
attributable to replanning rather than to whatever else a separate code path
would have changed. That is the ablation the paper reports, run as a config
change.

A replan costing more than the budget has left is rejected whole rather than
truncated. Applying the first two edits of a three-edit plan is how you end up
adding a subtask whose dependency was never added -- and the graph would then
be rejected as dangling anyway, having spent the budget getting there.

PARSING DISCIPLINE
==================
Same as the policy adapters: strict JSON, ``json.loads``, no coercion, no
default plan, and every failure carries the raw generation. Markdown fences are
the one tolerance, because general models emit them regardless of instructions.
A malformed plan is a prompt bug, and prompt bugs should arrive as stack traces
with the evidence attached, not as a quietly degraded run.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from ..config import ManagerConfig
from ..ppapi import ChatClient, PPAPIError, parse_json_object
from .plan import DAG, InvalidPlan, Replan, Subtask, SubtaskOutcome

logger = logging.getLogger(__name__)


class ManagerProtocolError(PPAPIError):
    """The manager's reply did not match the requested schema.

    Carries the full raw generation, because the only useful response to this
    is reading what the model actually said and fixing the prompt.
    """


@dataclass
class PlanningBudget:
    """How many DAG edits this run may still make.

    Separate from the API ``CallBudget``: that one bounds *spend*, this one
    bounds *how much the plan may change*. They are different failure modes --
    a manager can burn money without editing anything, and can wreck a plan in
    two cheap edits -- so conflating them would make both unreadable.
    """

    limit: int
    spent: int = 0
    #: Edits proposed but refused for want of budget. Reported, not silent.
    refused: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.spent)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    def can_afford(self, edits: int) -> bool:
        return edits <= self.remaining

    def spend(self, edits: int) -> None:
        if not self.can_afford(edits):  # pragma: no cover - callers check first
            raise ValueError(
                f"planning budget cannot afford {edits} edit(s); "
                f"{self.remaining} of {self.limit} remain"
            )
        self.spent += edits

    def as_dict(self) -> dict[str, Any]:
        return {
            "limit": self.limit,
            "spent": self.spent,
            "remaining": self.remaining,
            "refused_edits": self.refused,
        }


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------
DECOMPOSE_SYSTEM = """\
You are the manager of a team of web browsing agents. You do not browse \
yourself. You break one goal into subtasks and arrange them into a dependency \
graph, which the system then executes: subtasks with no unmet dependencies run \
AT THE SAME TIME, in separate browsers.

Each agent starts at the same start page, sees only screenshots of its own \
browser, and knows nothing about the others except what you pass down through \
dependencies. Write every instruction so it makes sense to somebody who can \
read only that one instruction.

Reply with a single JSON object and NOTHING else. No markdown, no code fence, \
no commentary before or after. The object has exactly one key:

  "subtasks": a list of objects, each with exactly these three keys:
      "id"         a short unique slug, lowercase with underscores, e.g. \
"find_product_page"
      "instruction" what that one agent must do, in plain English, ending in \
what it should report back
      "depends_on" a list of ids that must finish first. Use [] for subtasks \
that can start immediately.

Rules:
  - At least one subtask MUST have "depends_on": [].
  - Dependencies must refer to ids that exist in this same reply.
  - The graph must be acyclic. Nothing may depend on itself, directly or \
through a chain.
  - Use at most {max_subtasks} subtasks. Fewer is better: every subtask is a \
whole browser and a whole model budget.
  - Only add a dependency when the later subtask genuinely NEEDS the earlier \
one's finding. Independent subtasks run in parallel, so a needless dependency \
just makes the run slower.
  - Prefer breadth: several independent subtasks that each answer part of the \
goal, plus at most one that combines them.
  - If the goal is genuinely one step, emit exactly one subtask. Do not invent \
work.
  - Each instruction must end by stating what the agent should report, since \
that text is what gets passed on.

Example, for "compare the price of the same book on two sites":
{{"subtasks": [
  {{"id": "price_site_a", "instruction": "On site A, find the book \\"Dune\\" \
and report its price.", "depends_on": []}},
  {{"id": "price_site_b", "instruction": "On site B, find the book \\"Dune\\" \
and report its price.", "depends_on": []}},
  {{"id": "compare", "instruction": "Using the two prices already found, report \
which site is cheaper and by how much.", "depends_on": ["price_site_a", \
"price_site_b"]}}
]}}
"""

REPLAN_SYSTEM = """\
You are the manager of a team of web browsing agents. A wave of subtasks has \
just finished. You decide whether the remaining plan should change.

Reply with a single JSON object and NOTHING else. No markdown, no code fence, \
no commentary. The object has exactly three keys:

  "reason"  one sentence on why you are or are not changing the plan
  "add"     a list of NEW subtasks, each with exactly "id", "instruction" and \
"depends_on"
  "remove"  a list of ids to drop from the plan

Rules:
  - Changing nothing is usually right. Emit "add": [] and "remove": [] unless \
the results genuinely call for a change.
  - You may spend at most {remaining} edit(s) in this reply. Each added \
subtask and each removed subtask counts as one. A reply that costs more than \
that is discarded whole, so keep it small.
  - You may only remove subtasks that have NOT started. Anything already done, \
failed or blocked stays in the record.
  - New ids must not clash with existing ones. Dependencies may point at \
existing subtasks or at other subtasks you are adding in this same reply.
  - The graph must stay acyclic and every dependency must exist.

Change the plan when, and only when:
  - a subtask failed and a different approach could still get the information;
  - a finding makes a planned subtask pointless -- remove it;
  - a finding reveals work that is genuinely needed and was not planned.

Do NOT add a subtask merely to double-check a result that already looks fine.
"""


class Manager:
    """One LLM, two hooks. Owns the planning budget for the run."""

    name = "manager"

    def __init__(
        self,
        client: ChatClient,
        config: ManagerConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or ManagerConfig()
        self.budget = PlanningBudget(limit=self.config.planning_budget)
        #: Every replan attempt, applied or not, for run.json.
        self.history: list[dict[str, Any]] = []

    @property
    def model(self) -> str:
        """The manager's model. Empty config value means "same as the policy",
        which is the client's default -- so the default run needs no extra
        configuration, and a stronger manager is one setting away."""
        return self.config.model or self.client.default_model

    # --- hook 1: decompose -------------------------------------------------

    async def decompose(self, task: str, start_url: str | None = None) -> DAG:
        """Turn the goal into a validated graph, or fail loudly.

        ``start_url`` is where every agent will begin. Telling the manager
        lets it write instructions that name the site rather than instructions
        that assume the agent will find it -- and it costs nothing when the
        strategy has no start URL to give.
        """
        start = f"# START PAGE\nEvery agent begins at {start_url}\n\n" if start_url else ""
        user = (
            f"# GOAL\n{task}\n\n{start}"
            f"# YOUR JOB\nBreak this into at most {self.config.max_subtasks} "
            f"subtasks and reply with the JSON object."
        )
        raw = await self._ask(
            DECOMPOSE_SYSTEM.format(max_subtasks=self.config.max_subtasks), user
        )
        dag = self._parse_dag(raw)
        logger.info("manager decomposed the task into %s", dag.summary())
        return dag

    # --- hook 2: replan ----------------------------------------------------

    async def replan(
        self,
        dag: DAG,
        outcomes: Sequence[SubtaskOutcome],
        wave: int | None = None,
    ) -> Replan | None:
        """Propose edits after a wave, or ``None`` if there are to be none.

        ``None`` covers three cases, all of which mean "carry on with the plan
        you have": the budget is spent (no call is made -- that is the B=0
        ablation), the model proposed nothing, or the model proposed more than
        the budget can afford. Every one of them is recorded in ``history``, so
        run.json can distinguish "the manager was happy" from "the manager was
        overruled". ``wave`` is stamped on the record so a viewer can place it.
        """
        if self.budget.exhausted:
            logger.info(
                "planning budget exhausted (%d/%d edits); not replanning",
                self.budget.spent, self.budget.limit,
            )
            self.history.append(
                {
                    "applied": False,
                    "reason": (
                        f"planning budget exhausted "
                        f"({self.budget.spent}/{self.budget.limit} edits used)"
                    ),
                    "called_model": False,
                    "wave": wave,
                    "outcome": "budget exhausted",
                }
            )
            return None

        raw = await self._ask(
            REPLAN_SYSTEM.format(remaining=self.budget.remaining),
            _replan_user_message(dag, outcomes, self.budget.remaining),
        )
        replan = self._parse_replan(raw)

        if replan.is_noop:
            logger.info("manager kept the plan unchanged: %s", replan.reason)
            self.history.append(
                {**replan.as_dict(), "applied": False, "called_model": True,
                 "wave": wave, "outcome": "no change proposed"}
            )
            return None

        if not self.budget.can_afford(replan.edits):
            # Rejected whole. See the module docstring: a partly-applied graph
            # edit is how dangling dependencies get built.
            logger.warning(
                "refusing a %d-edit replan: only %d edit(s) of budget remain",
                replan.edits, self.budget.remaining,
            )
            self.budget.refused += replan.edits
            self.history.append(
                {**replan.as_dict(), "applied": False, "called_model": True,
                 "wave": wave,
                 "outcome": (
                     f"refused: costs {replan.edits} edit(s), "
                     f"{self.budget.remaining} remain")}
            )
            return None

        return replan

    def record_applied(self, replan: Replan, wave: int) -> None:
        """Charge an applied replan to the budget and record it.

        Called by the strategy *after* the edit has been accepted by
        ``DAG.apply``, so a replan that produced an invalid graph never gets
        billed -- the run did not get the plan it paid for.
        """
        self.budget.spend(replan.edits)
        self.history.append(
            {**replan.as_dict(), "applied": True, "called_model": True, "wave": wave}
        )

    def record_rejected(self, replan: Replan, wave: int, error: str) -> None:
        """Record a replan whose edits would not have left a valid graph."""
        self.history.append(
            {**replan.as_dict(), "applied": False, "called_model": True,
             "wave": wave, "outcome": f"rejected: {error}"}
        )

    # --- plumbing ----------------------------------------------------------

    async def _ask(self, system: str, user: str) -> str:
        raw = await self.client.complete(
            system=system,
            user=user,
            model=self.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        logger.debug("manager raw generation: %s", raw)
        return raw

    def _parse_dag(self, raw: str) -> DAG:
        payload = parse_json_object(raw, "manager decompose", ManagerProtocolError)
        unexpected = set(payload) - {"subtasks"}
        if unexpected:
            raise ManagerProtocolError(
                f"manager decompose: reply has unexpected top-level key(s) "
                f"{sorted(unexpected)}; expected exactly 'subtasks'."
                f"\nRaw generation: {raw}"
            )
        if "subtasks" not in payload:
            raise ManagerProtocolError(
                f"manager decompose: reply has no 'subtasks' key."
                f"\nRaw generation: {raw}"
            )
        try:
            return DAG(subtasks=payload["subtasks"])
        except ValidationError as exc:
            raise ManagerProtocolError(
                f"manager decompose: subtasks do not match the schema: {exc}"
                f"\nRaw generation: {raw}"
            ) from exc
        except InvalidPlan as exc:
            # Re-raised as itself, not wrapped: a structural failure is a
            # different bug from a schema failure, and the caller may well want
            # to tell them apart. The generation rides along either way.
            raise InvalidPlan(f"{exc}\nRaw generation: {raw}") from exc

    def _parse_replan(self, raw: str) -> Replan:
        payload = parse_json_object(raw, "manager replan", ManagerProtocolError)
        unexpected = set(payload) - {"reason", "add", "remove"}
        if unexpected:
            raise ManagerProtocolError(
                f"manager replan: reply has unexpected top-level key(s) "
                f"{sorted(unexpected)}; expected 'reason', 'add' and 'remove'."
                f"\nRaw generation: {raw}"
            )
        try:
            return Replan.model_validate(payload)
        except ValidationError as exc:
            raise ManagerProtocolError(
                f"manager replan: reply does not match the schema: {exc}"
                f"\nRaw generation: {raw}"
            ) from exc

    async def close(self) -> None:
        await self.client.close()

    def as_dict(self) -> dict[str, Any]:
        """Manager-level facts for run.json. No credentials, by construction."""
        return {
            "name": self.name,
            "model": self.model,
            "max_subtasks": self.config.max_subtasks,
            "budget": self.budget.as_dict(),
        }


def _replan_user_message(
    dag: DAG, outcomes: Sequence[SubtaskOutcome], remaining: int
) -> str:
    """The current graph and what just came back from it.

    Failed subtasks are shown with their error text, and pending ones with
    their dependencies, because those two together are what a decision to
    replan is actually made from.
    """
    lines = ["# CURRENT PLAN"]
    for subtask in dag.topological_order():
        deps = ", ".join(subtask.depends_on) or "none"
        lines.append(
            f"- {subtask.id} [{subtask.status}] (depends on: {deps})\n"
            f"    {subtask.instruction}"
        )

    lines.append("\n# RESULTS SO FAR")
    if outcomes:
        for outcome in outcomes:
            lines.append(f"- {outcome.id} [{outcome.status}]")
            if outcome.answer:
                lines.append(f"    reported: {outcome.answer}")
            if outcome.error:
                lines.append(f"    error: {outcome.error}")
            if outcome.status != "done" and not outcome.error:
                lines.append(
                    f"    no answer after {outcome.num_steps} step(s), "
                    f"{outcome.num_errors} of which failed"
                )
    else:
        lines.append("(nothing has finished yet)")

    lines.append(
        f"\n# YOUR JOB\nDecide whether the remaining plan should change. You "
        f"may spend at most {remaining} edit(s). Reply with the JSON object."
    )
    return "\n".join(lines)


def build_subtask(subtask_id: str, instruction: str, depends_on: list[str] | None = None) -> Subtask:
    """Small constructor, mostly for tests and for hand-written plans."""
    return Subtask(
        id=subtask_id, instruction=instruction, depends_on=list(depends_on or [])
    )


def format_context(entries: Sequence[tuple[str, str]]) -> str:
    """Render completed dependency results as a block for a dependent's prompt.

    Lives here rather than in the strategy because it is prompt text, and
    prompt text belongs next to the other prompt text. ``json.dumps`` on the
    answer keeps a multi-line finding from being mistaken for more instructions.
    """
    lines = ["# WHAT EARLIER AGENTS ALREADY FOUND"]
    for subtask_id, answer in entries:
        lines.append(f"- {subtask_id}: {json.dumps(answer, ensure_ascii=False)}")
    lines.append(
        "Treat the above as established. Do not redo that work; use it to "
        "complete your own task."
    )
    return "\n".join(lines)


__all__ = [
    "Manager",
    "ManagerProtocolError",
    "PlanningBudget",
    "build_subtask",
    "format_context",
]
