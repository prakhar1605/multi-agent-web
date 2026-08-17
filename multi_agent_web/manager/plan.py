"""The plan: subtasks, the graph they form, and edits to it.

VALIDATION IS THE POINT OF THIS FILE
====================================
A ``DAG`` cannot be constructed unless it is a DAG. Every structural rule is
checked in the constructor, and a violation raises :class:`InvalidPlan` rather
than being quietly repaired:

* every ``depends_on`` id must name a subtask that exists  (no dangling edges)
* the graph must be acyclic
* at least one subtask must have no dependencies       (something can start)
* ids must be unique and non-empty

Repairing would be easy -- drop the unknown edge, break the back edge, promote
an arbitrary node to root -- and that is exactly why it is not done. A manager
LLM that emits a cyclic graph has a prompt problem, and a strategy that silently
straightens the graph out turns "my prompt is wrong" into "the answers are
slightly worse for reasons I cannot see". The graph is the manager's output; if
it is malformed, the manager is what needs fixing.

``InvalidPlan`` deliberately does NOT inherit from ``ValueError``. Pydantic
converts a ``ValueError`` raised inside a validator into a ``ValidationError``,
which would bury the explanation -- and the explanation ("subtask 'c' depends on
'e', which is not in the plan") is the whole value of failing here.

WHY EDITS ARE A MODEL AND NOT A NEW GRAPH
=========================================
Replanning returns a :class:`Replan` -- a set of additions and removals -- and
never a replacement graph. Two reasons. The planning budget is denominated in
edits, so edits have to be countable. And a manager handed the licence to
re-emit the whole graph will re-emit the whole graph, including the parts that
have already run; ``apply`` refuses to remove a subtask that is not still
pending, so completed work cannot be rewritten out of the record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Where a subtask is in its life.
#:
#: ``blocked`` is not a failure of the subtask itself -- it is a subtask that
#: can never start because something it depends on failed. Kept distinct from
#: ``failed`` so a run report can say "one agent failed and took three others
#: with it", which is a different story from "four agents failed".
SubtaskStatus = Literal["pending", "running", "done", "failed", "blocked"]

TERMINAL_STATUSES = ("done", "failed", "blocked")


class InvalidPlan(Exception):
    """A plan violated a structural rule. See the module docstring."""


class Subtask(BaseModel):
    """One unit of work for one browsing agent.

    The first three fields are the manager's to write. The rest are the
    execution record, filled in by the strategy as the graph runs, and are what
    lets ``run.json`` carry a final DAG that shows what actually happened rather
    than only what was intended.
    """

    # A manager that invents a field is a prompt problem worth seeing, same
    # discipline as the action schema in the policy adapters.
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, description="Short unique slug, e.g. 'find_price'.")
    instruction: str = Field(
        min_length=1,
        description="Self-contained instruction for one agent, in plain English.",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Ids of subtasks that must finish before this one starts.",
    )
    status: SubtaskStatus = "pending"

    # --- execution record, never set by the manager ------------------------
    agent_index: int | None = None
    wave: int | None = None
    answer: str | None = None
    error: str | None = None

    @property
    def finished(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class Replan(BaseModel):
    """A set of edits to a graph, plus why.

    An empty ``add`` and ``remove`` is a legitimate and common reply: it means
    the manager looked at the results and concluded the plan still holds. It
    costs nothing against the budget, which is deliberate -- charging for "no
    change" would make the cheapest correct answer the expensive one.
    """

    model_config = ConfigDict(extra="forbid")

    reason: str = ""
    add: list[Subtask] = Field(default_factory=list)
    remove: list[str] = Field(default_factory=list)

    @property
    def edits(self) -> int:
        """What this costs against the planning budget: one per node touched."""
        return len(self.add) + len(self.remove)

    @property
    def is_noop(self) -> bool:
        return self.edits == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "add": [s.as_dict() for s in self.add],
            "remove": list(self.remove),
            "edits": self.edits,
        }


class DAG(BaseModel):
    """A validated dependency graph of subtasks.

    Construction is the validation gate: if you are holding one of these, it is
    acyclic, fully connected to existing ids, and has somewhere to start.
    """

    model_config = ConfigDict(extra="forbid")

    subtasks: list[Subtask]

    # --- validation --------------------------------------------------------

    @model_validator(mode="after")
    def _validate_structure(self) -> "DAG":
        if not self.subtasks:
            raise InvalidPlan(
                "the plan is empty. A decomposition must contain at least one "
                "subtask; if the task needs no decomposition, emit a single one."
            )

        ids = [s.id for s in self.subtasks]
        seen: set[str] = set()
        duplicates = set()
        for subtask_id in ids:
            if subtask_id in seen:
                duplicates.add(subtask_id)
            seen.add(subtask_id)
        if duplicates:
            raise InvalidPlan(
                f"duplicate subtask id(s) {sorted(duplicates)}. Ids address nodes "
                f"in the graph, so they must be unique."
            )

        known = set(ids)
        for subtask in self.subtasks:
            unknown = [d for d in subtask.depends_on if d not in known]
            if unknown:
                raise InvalidPlan(
                    f"subtask {subtask.id!r} depends on {unknown}, which "
                    f"{'is' if len(unknown) == 1 else 'are'} not in the plan. "
                    f"Known ids: {sorted(known)}."
                )
            if subtask.id in subtask.depends_on:
                raise InvalidPlan(
                    f"subtask {subtask.id!r} depends on itself, so it can never "
                    f"start."
                )

        # Cycles are checked before roots on purpose. A graph with no root is
        # necessarily cyclic, so both rules would fire on it -- and "a -> b -> a"
        # tells you where to look, while "nothing can start" does not.
        cycle = self._find_cycle()
        if cycle:
            raise InvalidPlan(
                f"the plan is cyclic: {' -> '.join(cycle)}. Subtasks in a cycle "
                f"can never run, because each waits on the other."
            )

        # Unreachable on a finite acyclic graph, which is why it comes second.
        # Kept because "something can start" is the property the executor
        # actually relies on, and an invariant worth relying on is worth
        # asserting rather than inferring from another check.
        if not any(not s.depends_on for s in self.subtasks):  # pragma: no cover
            raise InvalidPlan(
                "every subtask has a dependency, so nothing can start. At least "
                "one subtask must have an empty 'depends_on'."
            )
        return self

    def _find_cycle(self) -> list[str] | None:
        """Return one cycle as a path, or None. Named nodes beat a bare bool:
        the error message is what makes a bad plan fixable."""
        by_id = {s.id: s for s in self.subtasks}
        # 0 = unvisited, 1 = on the current path, 2 = fully explored.
        state: dict[str, int] = {i: 0 for i in by_id}
        path: list[str] = []

        def walk(node: str) -> list[str] | None:
            state[node] = 1
            path.append(node)
            for dep in by_id[node].depends_on:
                if state.get(dep) == 1:
                    return path[path.index(dep):] + [dep]
                if state.get(dep) == 0:
                    found = walk(dep)
                    if found:
                        return found
            path.pop()
            state[node] = 2
            return None

        for node in by_id:
            if state[node] == 0:
                found = walk(node)
                if found:
                    return found
        return None

    # --- reading the graph -------------------------------------------------

    @property
    def by_id(self) -> dict[str, Subtask]:
        return {s.id: s for s in self.subtasks}

    def get(self, subtask_id: str) -> Subtask | None:
        return self.by_id.get(subtask_id)

    def dependents_of(self, subtask_id: str) -> list[Subtask]:
        return [s for s in self.subtasks if subtask_id in s.depends_on]

    def terminal(self) -> list[Subtask]:
        """Subtasks nothing depends on -- where the run's answer comes from."""
        depended_on = {d for s in self.subtasks for d in s.depends_on}
        return [s for s in self.subtasks if s.id not in depended_on]

    def ready(self) -> list[Subtask]:
        """Pending subtasks whose dependencies have all completed successfully.

        The next wave, in topological order. A dependency that *failed* does not
        satisfy anything -- see ``propagate_blocked``.
        """
        done = {s.id for s in self.subtasks if s.status == "done"}
        return [
            s
            for s in self.topological_order()
            if s.status == "pending" and all(d in done for d in s.depends_on)
        ]

    def propagate_blocked(self) -> list[Subtask]:
        """Mark every subtask that can never start, and return the newly marked.

        Runs to a fixpoint so a failure propagates all the way down the graph,
        not just one level. Called after each wave: the alternative is a loop
        that spins with pending work it will never be able to launch.
        """
        newly: list[Subtask] = []
        changed = True
        while changed:
            changed = False
            for subtask in self.subtasks:
                if subtask.status != "pending":
                    continue
                blockers = [
                    d
                    for d in subtask.depends_on
                    if (dep := self.get(d)) is not None
                    and dep.status in ("failed", "blocked")
                ]
                if blockers:
                    subtask.status = "blocked"
                    subtask.error = (
                        f"blocked: depends on {blockers}, which did not complete"
                    )
                    newly.append(subtask)
                    changed = True
        return newly

    def topological_order(self) -> list[Subtask]:
        """Dependencies before dependents. Safe: the graph is known acyclic.

        Ties are broken by the order the manager emitted them, so a plan
        serialized into run.json reads in the order it was written.
        """
        by_id = self.by_id
        order: list[Subtask] = []
        placed: set[str] = set()
        remaining = list(self.subtasks)
        while remaining:
            layer = [s for s in remaining if all(d in placed for d in s.depends_on)]
            if not layer:  # pragma: no cover - impossible on a validated DAG
                order.extend(remaining)
                break
            order.extend(layer)
            placed.update(s.id for s in layer)
            remaining = [s for s in remaining if s.id not in placed]
        return order

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for subtask in self.subtasks:
            counts[subtask.status] = counts.get(subtask.status, 0) + 1
        return counts

    # --- editing -----------------------------------------------------------

    def apply(self, replan: Replan) -> "DAG":
        """Return a NEW graph with ``replan`` applied, or raise ``InvalidPlan``.

        A new object rather than a mutation, because validation happens in the
        constructor: building the result is what proves the edit left a graph
        that is still runnable. An edit that would introduce a cycle or a
        dangling dependency raises here, before anything is launched.

        Surviving subtasks keep their execution record -- their status, answer
        and agent index all carry over -- so applying an edit mid-run does not
        lose what the completed agents found.
        """
        current = self.by_id

        unknown = [i for i in replan.remove if i not in current]
        if unknown:
            raise InvalidPlan(
                f"cannot remove {unknown}: no such subtask. Known ids: "
                f"{sorted(current)}."
            )
        started = [i for i in replan.remove if current[i].status != "pending"]
        if started:
            raise InvalidPlan(
                f"cannot remove {started}: already {', '.join(sorted({current[i].status for i in started}))}. "
                f"A replan may prune work that has not started; it may not "
                f"rewrite what already ran."
            )

        clashes = [s.id for s in replan.add if s.id in current and s.id not in replan.remove]
        if clashes:
            raise InvalidPlan(
                f"cannot add {clashes}: {'that id is' if len(clashes) == 1 else 'those ids are'} "
                f"already in the plan. To change a subtask, remove it and add it back."
            )

        removed = set(replan.remove)
        kept = [s.model_copy(deep=True) for s in self.subtasks if s.id not in removed]
        return DAG(subtasks=kept + [s.model_copy(deep=True) for s in replan.add])

    # --- serialization -----------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        """The shape written into run.json, in topological order."""
        return {
            "subtasks": [s.as_dict() for s in self.topological_order()],
            "counts": self.counts(),
        }

    def summary(self) -> str:
        counts = ", ".join(f"{n} {status}" for status, n in sorted(self.counts().items()))
        return f"{len(self.subtasks)} subtask(s): {counts}"


@dataclass(frozen=True)
class SubtaskOutcome:
    """What the manager is shown about a finished subtask when it replans.

    A plain dataclass, and deliberately not a ``SessionResult``: the manager
    reasons about *work*, not about browser sessions, and keeping the
    orchestrator's types out of this package is what stops the dependency
    running both ways (``orchestrator.strategy.dag`` imports ``manager``, never
    the reverse).

    Failures are included, with their error text. A manager that cannot see
    what went wrong cannot plan a way around it, which is most of what
    replanning is for.
    """

    id: str
    instruction: str
    status: str
    answer: str | None = None
    num_steps: int = 0
    num_errors: int = 0
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "done"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "answer": self.answer,
            "num_steps": self.num_steps,
            "num_errors": self.num_errors,
            "error": self.error,
        }


__all__ = [
    "DAG",
    "InvalidPlan",
    "Replan",
    "Subtask",
    "SubtaskOutcome",
    "SubtaskStatus",
]
