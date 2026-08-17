"""DAG execution: run the manager's graph in waves, replanning as results land.

    decompose -> [wave: launch everything whose dependencies are met]
              -> replan on what came back
              -> repeat until nothing is runnable
              -> aggregate

WHY WAVES, AND WHAT A WAVE COSTS
================================
Each pass launches *every* subtask whose dependencies are satisfied, through
one ``Runner.run_sessions`` call, so independent subtasks run concurrently
under the same two limits everything else does. The wave is also the
replanning point: the manager sees a wave's results before the next wave is
built, which is the whole reason ``Runner`` was made re-entrant in Phase 2
(``test_runner_is_reentrant_for_phase_3_waves``).

A wave is a barrier, and barriers cost wall-clock: a two-subtask wave takes as
long as its slower subtask, even if the faster one's dependent could have
started earlier. The alternative -- launching each subtask the moment its own
dependencies land -- is faster and gives the manager nothing coherent to replan
against, since "the results so far" would be a different set for every
in-flight subtask. Planning quality is the thing being tested here, so the
barrier stays, and ``timing.speedup_vs_serial`` in run.json shows what it cost.

FAILURE IS CONTAINED, TWICE
===========================
A subtask that crashes does not sink the run: its siblings finish, the graph
keeps going, and the manager is shown the failure so it can plan around it.
What it does stop is its own dependents -- a subtask whose input never arrived
is marked ``blocked`` rather than launched against missing context, and the
distinction between ``failed`` (this agent failed) and ``blocked`` (someone
upstream did) survives into run.json.

The second containment is the replan. An edit that would leave an invalid graph
is refused by ``DAG.apply``; a manager that emits unparseable JSON, or an API
budget that runs dry mid-run, is caught the same way. In each case the failure
is logged at ERROR and recorded in ``details.replans`` with its reason, and the
run continues on the plan it already had. The graph is never silently
straightened out -- the rejection and the raw problem are both in the report --
but a manager typo at wave three does not throw away three waves of paid work.

``decompose`` is deliberately not contained: a failure there is fatal. There is
no earlier plan to fall back to, the plan *is* the run, and nothing has been
spent yet -- so raising costs nothing and hides nothing.

WHY THERE IS NO JUDGE HERE
==========================
Best-of-N judges N attempts at the *same* task, so "which is best" is a
question with an answer. A DAG's subtasks are different pieces of work; asking
which of "find the price" and "find the shipping cost" is better is not a
question. The answer is therefore composed, not chosen: terminal subtasks are
the ones nothing else depends on, so their answers are what the graph was built
to produce, and ``contributing_indices`` lists all of them -- which is why that
field was a list from Phase 2 onwards.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ...manager.manager import Manager, format_context
from ...manager.plan import DAG, InvalidPlan, Subtask, SubtaskOutcome
from ...ppapi import CallBudgetExceeded, PPAPIError
from ..session import SessionResult, SessionSpec
from .base import Strategy, StrategyOutcome

if TYPE_CHECKING:  # pragma: no cover
    from ..runner import Runner

logger = logging.getLogger(__name__)


class DagStrategy(Strategy):
    """Decompose with the manager, execute the graph in waves, aggregate."""

    name = "dag"

    def __init__(
        self,
        manager: Manager,
        start_url: str | None = None,
        max_waves: int | None = None,
    ) -> None:
        self.manager = manager
        self.start_url = start_url
        self.max_waves = max_waves or manager.config.max_waves

    async def run(self, task: str, runner: "Runner") -> StrategyOutcome:
        dag = await self.manager.decompose(task)
        initial = dag.as_dict()

        sessions: list[SessionResult] = []
        outcomes: list[SubtaskOutcome] = []
        waves: list[dict[str, Any]] = []
        stopped_early: str | None = None
        wave = 0

        while True:
            ready = dag.ready()
            if not ready:
                break
            wave += 1
            if wave > self.max_waves:
                # The edit budget already bounds how much the graph can grow;
                # this is the backstop for the case where it grows anyway.
                stopped_early = (
                    f"stopped after {self.max_waves} wave(s) with "
                    f"{len(ready)} subtask(s) still runnable"
                )
                logger.warning("dag: %s", stopped_early)
                break

            results = await self._run_wave(ready, dag, runner, wave)
            sessions.extend(results)
            outcomes.extend(
                _outcome(s, r) for s, r in zip(ready, results)
            )

            blocked = dag.propagate_blocked()
            if blocked:
                logger.warning(
                    "dag: %d subtask(s) blocked by an upstream failure: %s",
                    len(blocked), [s.id for s in blocked],
                )
            waves.append(
                {
                    "wave": wave,
                    "subtasks": [s.id for s in ready],
                    "statuses": {s.id: s.status for s in ready},
                    "blocked": [s.id for s in blocked],
                }
            )

            dag = await self._maybe_replan(dag, outcomes, wave)

        answer, contributing, reason = _aggregate(dag, stopped_early)
        return StrategyOutcome(
            answer=answer,
            contributing_indices=contributing,
            reason=reason,
            sessions=sessions,
            details={
                "manager": self.manager.as_dict(),
                # Both graphs, because the pair is the measurement: the paper
                # reports DAG growth, which is exactly final minus initial.
                "initial_dag": initial,
                "final_dag": dag.as_dict(),
                "replans": self.manager.history,
                "budget": self.manager.budget.as_dict(),
                "waves": waves,
                "growth": _growth(initial, dag, self.manager.history, waves),
                "stopped_early": stopped_early,
            },
        )

    # --- one wave ----------------------------------------------------------

    async def _run_wave(
        self, ready: list[Subtask], dag: DAG, runner: "Runner", wave: int
    ) -> list[SessionResult]:
        specs = [
            SessionSpec(
                task=self._instruction_with_context(subtask, dag),
                start_url=self.start_url,
                label=f"wave {wave}: {subtask.id}",
            )
            for subtask in ready
        ]
        logger.info(
            "dag wave %d: launching %d subtask(s): %s",
            wave, len(specs), [s.id for s in ready],
        )
        for subtask in ready:
            subtask.status = "running"

        results = await runner.run_sessions(specs)

        # run_sessions returns results in spec order, which is `ready` order.
        for subtask, result in zip(ready, results):
            subtask.status = "done" if result.succeeded else "failed"
            subtask.agent_index = result.index
            subtask.wave = wave
            subtask.answer = result.answer
            subtask.error = result.error or (
                None if result.succeeded else f"agent finished as {result.status}"
            )
        return results

    def _instruction_with_context(self, subtask: Subtask, dag: DAG) -> str:
        """The instruction, plus what this subtask's dependencies found.

        This is the payoff of having dependencies at all: a subtask that waited
        on another gets that other's answer in its own prompt, so the second
        agent can use the first agent's finding instead of rediscovering it.
        Only direct dependencies are passed -- transitive results reach here
        through the intermediate subtask's own answer, and pasting the whole
        graph's history into every prompt is how a context window gets spent on
        things nobody needs.
        """
        found = [
            (dep.id, dep.answer)
            for dep_id in subtask.depends_on
            if (dep := dag.get(dep_id)) is not None
            and dep.status == "done"
            and (dep.answer or "").strip()
        ]
        if not found:
            return subtask.instruction
        return f"{subtask.instruction}\n\n{format_context(found)}"

    # --- replanning --------------------------------------------------------

    async def _maybe_replan(
        self, dag: DAG, outcomes: list[SubtaskOutcome], wave: int
    ) -> DAG:
        """Ask the manager for edits and apply them, or keep the current graph.

        Returns a graph either way. The manager decides whether to propose
        anything (and refuses itself when the budget cannot cover it); this
        decides whether what it proposed is a graph we can still run.
        """
        try:
            replan = await self.manager.replan(dag, outcomes)
        except (PPAPIError, CallBudgetExceeded) as exc:
            # The manager broke or the API budget ran dry partway through. Both
            # are recorded and neither is fatal: waves that already ran are paid
            # for, and the remaining plan is still executable without further
            # advice. Contrast ``decompose``, where a failure IS fatal -- there
            # is no plan to fall back on, and nothing has been spent yet.
            logger.error("dag: manager failed to replan at wave %d -- %s", wave, exc)
            self.manager.history.append(
                {
                    "reason": "",
                    "add": [],
                    "remove": [],
                    "edits": 0,
                    "applied": False,
                    "called_model": True,
                    "wave": wave,
                    "outcome": f"failed: {type(exc).__name__}: {exc}",
                }
            )
            return dag

        if replan is None:
            return dag

        try:
            edited = dag.apply(replan)
        except InvalidPlan as exc:
            # Loud and recorded, but not fatal: the completed subtasks' work is
            # real and paid for. The run continues on the unedited plan, and
            # details.replans carries the rejection so a bad prompt is visible
            # afterwards rather than inferred from odd results.
            logger.error(
                "dag: refusing the manager's replan at wave %d -- %s", wave, exc
            )
            self.manager.record_rejected(replan, wave, str(exc))
            return dag

        self.manager.record_applied(replan, wave)
        logger.info(
            "dag: applied a %d-edit replan at wave %d (+%s -%s): %s",
            replan.edits, wave, [s.id for s in replan.add], replan.remove,
            replan.reason,
        )
        # A newly added subtask may depend on one that already ran, so re-run
        # blocking now rather than at the top of the next wave -- otherwise a
        # subtask hung off a failed dependency would look runnable for a moment.
        edited.propagate_blocked()
        return edited


# ---------------------------------------------------------------------------
# aggregation and reporting
# ---------------------------------------------------------------------------
def _outcome(subtask: Subtask, result: SessionResult) -> SubtaskOutcome:
    """What the manager sees about one finished subtask.

    Step and error counts come from the session rather than the graph: "failed
    after one step" and "failed after fifteen, eleven of them erroring" call for
    different replans, and the subtask record alone cannot tell them apart.
    """
    return SubtaskOutcome(
        id=subtask.id,
        instruction=subtask.instruction,
        status=subtask.status,
        answer=subtask.answer,
        num_steps=len(result.steps),
        num_errors=result.num_errors,
        error=subtask.error,
    )


def _aggregate(dag: DAG, stopped_early: str | None) -> tuple[str | None, list[int], str]:
    """Compose the run's answer from the subtasks nothing else depends on.

    Terminal subtasks are the graph's outputs by construction, so they are the
    default source. If none of them succeeded the run still may have found
    something -- a two-stage graph whose second stage crashed still has the
    first stage's answer -- so the fallback reports every answer there is,
    labelled, rather than reporting nothing. What it will not do is invent a
    synthesis: the text is the agents' own, quoted.
    """
    counts = dag.counts()
    tally = ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
    suffix = f". {stopped_early}" if stopped_early else ""

    def answered(subtasks: list[Subtask]) -> list[Subtask]:
        return [
            s for s in subtasks if s.status == "done" and (s.answer or "").strip()
        ]

    terminal = answered(dag.terminal())
    source, fallback = (terminal, False)
    if not source:
        source, fallback = (answered(dag.topological_order()), True)

    if not source:
        return (
            None,
            [],
            f"no subtask produced an answer ({len(dag.subtasks)} subtask(s): "
            f"{tally}){suffix}",
        )

    contributing = [s.agent_index for s in source if s.agent_index is not None]

    if len(source) == 1:
        # One output: report it verbatim rather than labelling a single line.
        answer = source[0].answer or ""
    else:
        answer = "\n".join(f"{s.id}: {s.answer}" for s in source)

    origin = (
        f"terminal subtask(s) {[s.id for s in source]}"
        if not fallback
        else (
            f"subtask(s) {[s.id for s in source]} -- no terminal subtask "
            f"succeeded, so every available answer is reported"
        )
    )
    return (
        answer,
        contributing,
        f"answer composed from {origin}; {len(dag.subtasks)} subtask(s): {tally}{suffix}",
    )


def _growth(
    initial: dict[str, Any],
    final: DAG,
    history: list[dict[str, Any]],
    waves: list[dict[str, Any]],
) -> dict[str, Any]:
    """The headline numbers the paper reports, computed once here.

    ``replan_rate`` is applied replans per wave: the fraction of decision points
    at which the manager actually changed the plan. Reported alongside the raw
    counts because the rate alone hides whether it was 1 of 2 waves or 8 of 16.
    """
    applied = [h for h in history if h.get("applied")]
    called = [h for h in history if h.get("called_model")]
    initial_n = len(initial["subtasks"])
    final_n = len(final.subtasks)
    return {
        "initial_subtasks": initial_n,
        "final_subtasks": final_n,
        "subtasks_added": sum(len(h.get("add", [])) for h in applied),
        "subtasks_removed": sum(len(h.get("remove", [])) for h in applied),
        "net_growth": final_n - initial_n,
        "waves": len(waves),
        "replans_proposed": len(called),
        "replans_applied": len(applied),
        "replan_rate": round(len(applied) / len(waves), 3) if waves else 0.0,
    }


__all__ = ["DagStrategy"]
