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
* every ``retry_of`` id must name a subtask that exists, and lineage must not
  loop                                                (no dangling lineage)

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
have already run; ``apply`` refuses to remove a subtask that has run or is
running, so the record of executed work cannot be rewritten.

A removal is allowed only for a subtask that never executed -- ``pending`` (not
started) or ``blocked`` (an upstream dependency failed, so it never will). That
distinction is the whole point of re-routing: when a failed lookup blocks a
join, the manager's fix is exactly to remove the blocked join and add one over
the inputs that did arrive. Refusing to remove a ``blocked`` node would forbid
the one edit the block was supposed to prompt.

LINEAGE IS DECLARED, NOT INFERRED
=================================
When a lookup fails, the manager's usual fix is to add another subtask that
tries the same thing a different way. Nothing in the graph said so: the retry
arrived as a fresh node in a later wave, and the only trace of the connection
was that the manager had picked a similar id. Inferring "retry" from id
similarity is guesswork, and it is wrong in both directions -- ``compare_v2``
supersedes ``compare`` while sharing no suffix convention with
``find_x_retry``, and two unrelated ``find_price_a`` / ``find_price_b`` lookups
share a prefix without either superseding the other.

So ``Subtask.retry_of`` lets the manager *say* it, and it is the manager's
field like ``id`` and ``instruction`` are. It is lineage, not dependency: a
retry does not wait for the attempt it replaces (that attempt is already
finished), so ``retry_of`` never affects scheduling, ``ready`` or
``propagate_blocked``. It exists so a report can group attempts of one logical
subtask instead of scattering them across waves.

It is optional and defaults to ``None``, which is what keeps every run recorded
before it existed parseable -- and what lets a manager that has nothing to
declare say nothing.
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

#: Statuses a replan may prune. A subtask in either never executed: ``pending``
#: has not started, and ``blocked`` never will (an upstream dependency failed),
#: so removing it rewrites nothing that ran. ``running``/``done``/``failed`` are
#: refused -- those are in flight or already happened, and a replan must not
#: edit the record of work that occurred.
REMOVABLE_STATUSES = ("pending", "blocked")


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
    #: Lineage, not dependency -- see the module docstring. Optional, and
    #: ``None`` on every subtask written before the field existed.
    retry_of: str | None = Field(
        default=None,
        description=(
            "Id of the subtask this one supersedes, when it is another attempt "
            "at the same work. Null for a subtask that supersedes nothing."
        ),
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
            if subtask.retry_of is not None:
                if subtask.retry_of not in known:
                    raise InvalidPlan(
                        f"subtask {subtask.id!r} declares retry_of "
                        f"{subtask.retry_of!r}, which is not in the plan. "
                        f"Lineage must point at a subtask that is still here. "
                        f"Known ids: {sorted(known)}."
                    )
                if subtask.retry_of == subtask.id:
                    raise InvalidPlan(
                        f"subtask {subtask.id!r} declares itself as its own "
                        f"retry_of. A subtask cannot supersede itself."
                    )

        lineage_cycle = self._find_lineage_cycle()
        if lineage_cycle:
            raise InvalidPlan(
                f"lineage loops: {' -> '.join(lineage_cycle)}. Following "
                f"retry_of must reach an original subtask, not come back round."
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

    def _find_lineage_cycle(self) -> list[str] | None:
        """Return one ``retry_of`` loop as a path, or None.

        Lineage is a chain, not a graph -- each subtask supersedes at most one
        other -- so following it is a walk, and the only way it fails is by
        coming back round. Cheap to check and worth checking: a loop would make
        "which attempt came first" unanswerable.
        """
        by_id = {s.id: s for s in self.subtasks}
        for start in by_id:
            seen: list[str] = []
            node: str | None = start
            while node is not None and node in by_id:
                if node in seen:
                    return seen[seen.index(node):] + [node]
                seen.append(node)
                node = by_id[node].retry_of
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
        ran = [i for i in replan.remove if current[i].status not in REMOVABLE_STATUSES]
        if ran:
            statuses = ", ".join(sorted({current[i].status for i in ran}))
            raise InvalidPlan(
                f"cannot remove {ran}: {statuses}. A replan may prune work that "
                f"never executed (pending, or blocked by an upstream failure); it "
                f"may not rewrite work that is running, done or failed."
            )

        clashes = [s.id for s in replan.add if s.id in current and s.id not in replan.remove]
        if clashes:
            raise InvalidPlan(
                f"cannot add {clashes}: {'that id is' if len(clashes) == 1 else 'those ids are'} "
                f"already in the plan. To change a subtask, remove it and add it back."
            )

        removed = set(replan.remove)
        # Caught here rather than left to the constructor: "you removed the
        # subtask this one says it retries" is a fixable instruction, while
        # "retry_of names something not in the plan" sends the manager looking
        # at the wrong edit.
        severed = [
            (s.id, s.retry_of)
            for s in [*self.subtasks, *replan.add]
            if s.id not in removed and s.retry_of in removed
        ]
        if severed:
            pairs = ", ".join(f"{i!r} retries {t!r}" for i, t in severed)
            raise InvalidPlan(
                f"cannot remove {sorted({t for _, t in severed})}: {pairs}. "
                f"Removing a subtask that another declares as its retry_of "
                f"would leave lineage pointing at nothing. Keep it, or drop "
                f"the retry_of."
            )

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
