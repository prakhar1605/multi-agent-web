"""Phase 3 tests: plan validation, the manager, the DAG loop, the LLM judge.

NO API CALLS ANYWHERE IN THIS FILE. Every manager and judge is driven by
``FakeChatClient``, which returns canned strings from a queue and records the
prompts it was given; every browsing agent is a ``MockPolicy``. That is not a
convenience -- it is the reason the planning logic is testable at all. A
manager whose only failure mode is "the live model said something odd today"
cannot be regression-tested, so the model is the one thing stubbed out and
everything around it is real: real Pydantic validation, real graph algorithms,
real ``Runner``, real browsers.

The ordering test in particular asserts on *when* things ran, not merely that
they all ran. A DAG executed as one flat batch would also finish every subtask,
so completion proves nothing about dependencies being respected -- the same
reasoning as the Phase 2 concurrency tests.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

pytest.importorskip("playwright", reason="playwright is not installed")

from multi_agent_web.actions import Done, Wait  # noqa: E402
from multi_agent_web.config import JudgeConfig, ManagerConfig, RunConfig  # noqa: E402
from multi_agent_web.manager import (  # noqa: E402
    DAG,
    InvalidPlan,
    Manager,
    ManagerProtocolError,
    Replan,
    Subtask,
    SubtaskOutcome,
)
from multi_agent_web.manager.manager import (  # noqa: E402
    DECOMPOSE_SYSTEM,
    REPLAN_SYSTEM,
    _replan_user_message,
)
from multi_agent_web.orchestrator import (  # noqa: E402
    LLMJudge,
    OrchestratorConfig,
    Runner,
    orchestrate,
)
from multi_agent_web.orchestrator.judge.base import Candidate  # noqa: E402
from multi_agent_web.orchestrator.judge.llm import (  # noqa: E402
    JudgeProtocolError,
    parse_verdict,
)
from multi_agent_web.orchestrator.strategy.dag import DagStrategy  # noqa: E402
from multi_agent_web.policy.base import AgentPolicy  # noqa: E402
from multi_agent_web.policy.mock import MockPolicy  # noqa: E402
from multi_agent_web.ppapi import ChatClient  # noqa: E402


# ---------------------------------------------------------------------------
# doubles
# ---------------------------------------------------------------------------
class FakeChatClient(ChatClient):
    """Returns queued replies; records every (system, user) pair it was asked.

    The whole surface of ``ChatClient`` is one method, which is what makes a
    fake this small possible -- and is why the manager was given a text-in /
    text-out client rather than the full API object.
    """

    name = "fake"
    default_model = "fake-model"

    def __init__(self, replies: Sequence[str], repeat_last: bool = False) -> None:
        self.replies = list(replies)
        self.repeat_last = repeat_last
        self.calls: list[dict[str, str]] = []
        self.closed = False

    async def complete(
        self,
        system: str,
        user: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append({"system": system, "user": user, "model": model or ""})
        if not self.replies:
            if self.repeat_last and self.calls:
                return NO_CHANGE
            raise AssertionError(
                f"FakeChatClient ran out of replies on call {len(self.calls)}. "
                f"The code under test made more model calls than the test "
                f"expected, which is usually the bug."
            )
        return self.replies.pop(0)

    async def close(self) -> None:
        self.closed = True

    @property
    def num_calls(self) -> int:
        return len(self.calls)


NO_CHANGE = json.dumps({"reason": "the plan still holds", "add": [], "remove": []})


def plan_reply(*subtasks: tuple[str, list[str]]) -> str:
    """``("a", []), ("b", ["a"])`` -> the JSON a manager would emit."""
    return json.dumps(
        {
            "subtasks": [
                {"id": i, "instruction": f"do {i}", "depends_on": deps}
                for i, deps in subtasks
            ]
        }
    )


def replan_reply(
    add: Sequence[tuple[str, list[str]]] = (),
    remove: Sequence[str] = (),
    reason: str = "because",
) -> str:
    return json.dumps(
        {
            "reason": reason,
            "add": [
                {"id": i, "instruction": f"do {i}", "depends_on": deps}
                for i, deps in add
            ],
            "remove": list(remove),
        }
    )


def dag_of(*subtasks: tuple[str, list[str]]) -> DAG:
    return DAG(
        subtasks=[
            Subtask(id=i, instruction=f"do {i}", depends_on=deps) for i, deps in subtasks
        ]
    )


class RecordingPolicy(AgentPolicy):
    """MockPolicy that writes down which task it was given, and when.

    ``order`` is shared across every agent in a run, so the sequence it ends up
    holding is the real launch order across waves -- which is what the
    dependency test needs to assert on.
    """

    name = "recording"

    def __init__(
        self,
        index: int,
        order: list[str],
        answers: dict[str, str] | None = None,
        crash_on: str | None = None,
    ) -> None:
        self.index = index
        self.order = order
        self.answers = answers or {}
        self.crash_on = crash_on
        self.inner: MockPolicy | None = None

    async def predict(self, task, history, screenshot, page):
        if self.inner is None:
            # First sight of the task: record it and decide what to answer.
            key = _subtask_key(task)
            self.order.append(key)
            if self.crash_on and key == self.crash_on:
                raise RuntimeError(f"simulated failure in subtask {key}")
            self.inner = MockPolicy(
                [Wait(seconds=0.01), Done(answer=self.answers.get(key, f"did {key}"))]
            )
        return await self.inner.predict(task, history, screenshot, page)

    async def reset(self) -> None:
        self.inner = None


def _subtask_key(task: str) -> str:
    """Recover the subtask id from the instruction the strategy composed.

    Instructions are built as "do <id>" by the fake manager, so the second word
    of the first line is the id.
    """
    return task.splitlines()[0].split()[1]


def run_config(tmp_path: Path, **kw) -> RunConfig:
    return RunConfig(headless=True, max_steps=6, runs_dir=tmp_path / "runs", **kw)


def manager_with(replies: Sequence[str], **config) -> Manager:
    return Manager(FakeChatClient(replies), ManagerConfig(**config))


# ---------------------------------------------------------------------------
# 1. plan validation -- reject, never repair
# ---------------------------------------------------------------------------
class TestPlanValidation:
    def test_a_valid_plan_parses(self) -> None:
        dag = dag_of(("search", []), ("read", ["search"]))
        assert [s.id for s in dag.subtasks] == ["search", "read"]
        assert [s.id for s in dag.ready()] == ["search"]
        assert [s.id for s in dag.terminal()] == ["read"]

    def test_a_cycle_is_rejected_and_named(self) -> None:
        with pytest.raises(InvalidPlan) as exc:
            dag_of(("a", []), ("b", ["c"]), ("c", ["b"]))
        assert "cyclic" in str(exc.value)
        # The message must name the nodes, or it cannot be acted on.
        assert "b" in str(exc.value) and "c" in str(exc.value)

    def test_a_self_dependency_is_rejected(self) -> None:
        with pytest.raises(InvalidPlan, match="depends on itself"):
            dag_of(("a", []), ("b", ["b"]))

    def test_a_dangling_dependency_is_rejected(self) -> None:
        with pytest.raises(InvalidPlan) as exc:
            dag_of(("a", []), ("b", ["ghost"]))
        assert "ghost" in str(exc.value)
        # And it lists what WAS available, so the typo is obvious.
        assert "'a'" in str(exc.value) or "a" in str(exc.value)

    def test_duplicate_ids_are_rejected(self) -> None:
        with pytest.raises(InvalidPlan, match="duplicate"):
            dag_of(("a", []), ("a", []))

    def test_an_empty_plan_is_rejected(self) -> None:
        with pytest.raises(InvalidPlan, match="empty"):
            DAG(subtasks=[])

    def test_a_plan_with_no_root_is_rejected(self) -> None:
        """Every node depending on another means nothing can ever start."""
        with pytest.raises(InvalidPlan):
            dag_of(("a", ["b"]), ("b", ["a"]))

    def test_a_long_cycle_is_caught(self) -> None:
        with pytest.raises(InvalidPlan, match="cyclic"):
            dag_of(("root", []), ("a", ["c"]), ("b", ["a"]), ("c", ["b"]))

    def test_a_diamond_is_valid(self) -> None:
        """Two paths converging is not a cycle -- the common false positive."""
        dag = dag_of(("a", []), ("b", ["a"]), ("c", ["a"]), ("d", ["b", "c"]))
        assert [s.id for s in dag.terminal()] == ["d"]
        order = [s.id for s in dag.topological_order()]
        assert order.index("a") < order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")


# ---------------------------------------------------------------------------
# 1b. lineage -- declared by the manager, never inferred
# ---------------------------------------------------------------------------
class TestLineage:
    """``retry_of`` says which subtask an added one supersedes.

    The field exists because the alternative is guessing from id similarity,
    and that guess is wrong in both directions: ``compare_v2`` supersedes
    ``compare`` without sharing a suffix convention with ``find_x_retry``,
    while ``price_a`` and ``price_b`` share a prefix and supersede nothing.
    """

    def test_a_plan_without_lineage_still_parses(self) -> None:
        """Optional means optional: every run recorded before the field."""
        dag = DAG(subtasks=[{"id": "a", "instruction": "do a", "depends_on": []}])
        assert dag.get("a").retry_of is None
        assert dag.get("a").as_dict()["retry_of"] is None

    def test_lineage_is_carried_into_run_json(self) -> None:
        dag = DAG(subtasks=[Subtask(id="a", instruction="do a"),
                            Subtask(id="a2", instruction="do a again", retry_of="a")])
        by_id = {s["id"]: s for s in dag.as_dict()["subtasks"]}
        assert by_id["a2"]["retry_of"] == "a"
        assert by_id["a"]["retry_of"] is None

    def test_lineage_pointing_at_nothing_is_rejected(self) -> None:
        with pytest.raises(InvalidPlan) as exc:
            DAG(subtasks=[Subtask(id="a", instruction="do a"),
                          Subtask(id="a2", instruction="again", retry_of="ghost")])
        assert "ghost" in str(exc.value) and "retry_of" in str(exc.value)

    def test_a_subtask_cannot_supersede_itself(self) -> None:
        with pytest.raises(InvalidPlan, match="its own retry_of"):
            DAG(subtasks=[Subtask(id="a", instruction="do a", retry_of="a")])

    def test_a_lineage_loop_is_rejected_and_named(self) -> None:
        """Two subtasks each claiming to supersede the other has no first attempt."""
        a = Subtask(id="a", instruction="do a", retry_of="b")
        b = Subtask(id="b", instruction="do b", retry_of="a")
        with pytest.raises(InvalidPlan) as exc:
            DAG(subtasks=[a, b])
        assert "lineage loops" in str(exc.value)
        assert "a" in str(exc.value) and "b" in str(exc.value)

    def test_lineage_does_not_schedule(self) -> None:
        """A retry waits for nothing: the attempt it replaces already finished.

        This is the whole reason it is a separate field rather than a
        dependency -- an edge would make the retry wait on a failed subtask,
        which would block it instead of replacing it.
        """
        dag = DAG(subtasks=[Subtask(id="find", instruction="find it"),
                            Subtask(id="find_again", instruction="find it again",
                                    retry_of="find")])
        dag.get("find").status = "failed"
        assert [s.id for s in dag.ready()] == ["find_again"]
        assert dag.propagate_blocked() == [], "lineage propagated a block"

    def test_a_retry_of_a_failed_lookup_is_the_ordinary_case(self) -> None:
        """The pricecheck shape: the lookup failed, its join is blocked, and
        the manager adds another attempt plus a join over the retry."""
        dag = dag_of(("price_a", []), ("price_b", []), ("join", ["price_a", "price_b"]))
        dag.get("price_b").status = "failed"
        dag.get("price_a").status = "done"
        dag.propagate_blocked()
        assert dag.get("join").status == "blocked"
        edited = dag.apply(
            Replan(
                add=[
                    Subtask(id="price_b_retry", instruction="try b again",
                            retry_of="price_b"),
                    Subtask(id="join_v2", instruction="join",
                            depends_on=["price_a", "price_b_retry"]),
                ],
                remove=["join"],
            )
        )
        assert edited.get("price_b_retry").retry_of == "price_b"
        assert [s.id for s in edited.ready()] == ["price_b_retry"]

    def test_removing_the_subtask_a_retry_names_is_refused(self) -> None:
        """Lineage may not be left dangling by an edit, and the error says how
        to fix it -- otherwise the manager is sent to look at the wrong node."""
        dag = dag_of(("a", []), ("join", ["a"]))
        dag.get("a").status = "failed"
        dag.propagate_blocked()
        with pytest.raises(InvalidPlan) as exc:
            dag.apply(
                Replan(
                    add=[Subtask(id="join_v2", instruction="join", retry_of="join")],
                    remove=["join"],
                )
            )
        assert "retries" in str(exc.value) and "drop" in str(exc.value)

    def test_attempts_of_one_subtask_are_recoverable_from_the_final_graph(self) -> None:
        """What the viewer actually needs: walk retry_of back to the original.

        Three attempts spread across three waves are one logical subtask, and
        this is the only thing in run.json that says so.
        """
        dag = DAG(subtasks=[
            Subtask(id="find", instruction="find", status="failed", wave=1),
            Subtask(id="find_retry", instruction="again", retry_of="find",
                    status="failed", wave=2),
            Subtask(id="find_direct", instruction="direct url", retry_of="find_retry",
                    status="failed", wave=3),
            Subtask(id="other", instruction="other", status="done", wave=1),
        ])
        by_id = dag.by_id
        chains: dict[str, list[str]] = {}
        for subtask in dag.subtasks:
            root = subtask.id
            while by_id[root].retry_of:
                root = by_id[root].retry_of
            chains.setdefault(root, []).append(subtask.id)
        assert chains == {
            "find": ["find", "find_retry", "find_direct"],
            "other": ["other"],
        }

class TestGraphEdits:
    def test_applying_a_replan_returns_a_new_validated_graph(self) -> None:
        dag = dag_of(("a", []), ("b", ["a"]))
        edited = dag.apply(
            Replan(add=[Subtask(id="c", instruction="do c", depends_on=["b"])])
        )
        assert [s.id for s in edited.subtasks] == ["a", "b", "c"]
        assert len(dag.subtasks) == 2, "the original graph was mutated"

    def test_an_edit_that_would_dangle_is_rejected(self) -> None:
        dag = dag_of(("a", []), ("b", ["a"]))
        with pytest.raises(InvalidPlan):
            dag.apply(Replan(add=[Subtask(id="c", instruction="c", depends_on=["zz"])]))

    def test_an_edit_that_would_cycle_is_rejected(self) -> None:
        dag = dag_of(("a", []), ("b", ["a"]))
        # Removing 'b' and re-adding it depending on a new 'c' that depends on
        # 'b' closes a loop that neither edit creates on its own.
        with pytest.raises(InvalidPlan, match="cyclic"):
            dag.apply(
                Replan(
                    remove=["b"],
                    add=[
                        Subtask(id="b", instruction="b", depends_on=["c"]),
                        Subtask(id="c", instruction="c", depends_on=["b"]),
                    ],
                )
            )

    @pytest.mark.parametrize("status", ["running", "done", "failed"])
    def test_executed_work_cannot_be_removed(self, status: str) -> None:
        """A replan must not rewrite work that ran or is running."""
        dag = dag_of(("a", []), ("b", ["a"]))
        dag.get("a").status = status
        with pytest.raises(InvalidPlan) as exc:
            dag.apply(Replan(remove=["a"]))
        assert status in str(exc.value)
        assert "running, done or failed" in str(exc.value)

    @pytest.mark.parametrize("status", ["pending", "blocked"])
    def test_work_that_never_ran_can_be_removed(self, status: str) -> None:
        """pending never started; blocked never will. Both are prunable.

        The ``blocked`` case is the regression: refusing it forbids the exact
        re-route a block is meant to prompt.
        """
        dag = dag_of(("a", []), ("gone", ["a"]))
        dag.get("gone").status = status
        if status == "blocked":
            dag.get("gone").error = "blocked: depends on a, which did not complete"
        edited = dag.apply(Replan(remove=["gone"]))
        assert edited.get("gone") is None
        assert [s.id for s in edited.subtasks] == ["a"]

    def test_surviving_subtasks_keep_their_results(self) -> None:
        dag = dag_of(("a", []), ("b", []))
        dag.get("a").status = "done"
        dag.get("a").answer = "found it"
        dag.get("a").agent_index = 3
        edited = dag.apply(Replan(remove=["b"]))
        assert edited.get("a").answer == "found it"
        assert edited.get("a").agent_index == 3
        assert edited.get("a").status == "done"

    def test_a_failed_dependency_blocks_its_dependents_transitively(self) -> None:
        dag = dag_of(("a", []), ("b", ["a"]), ("c", ["b"]), ("d", []))
        dag.get("a").status = "failed"
        blocked = dag.propagate_blocked()
        assert {s.id for s in blocked} == {"b", "c"}
        assert dag.get("d").status == "pending", "an unrelated subtask was blocked"
        assert dag.ready() == [dag.get("d")]


# ---------------------------------------------------------------------------
# 2. the manager -- parsing and failing loudly
# ---------------------------------------------------------------------------
class TestManagerDecompose:
    @pytest.mark.asyncio
    async def test_valid_decomposition_parses(self) -> None:
        manager = manager_with([plan_reply(("find", []), ("read", ["find"]))])
        dag = await manager.decompose("do the thing")
        assert [s.id for s in dag.subtasks] == ["find", "read"]
        assert dag.get("read").depends_on == ["find"]

    @pytest.mark.asyncio
    async def test_the_goal_reaches_the_prompt(self) -> None:
        client = FakeChatClient([plan_reply(("a", []))])
        await Manager(client).decompose("find the Eiffel Tower's height")
        assert "find the Eiffel Tower's height" in client.calls[0]["user"]

    @pytest.mark.asyncio
    async def test_a_markdown_fence_is_tolerated(self) -> None:
        manager = manager_with(["```json\n" + plan_reply(("a", [])) + "\n```"])
        dag = await manager.decompose("t")
        assert [s.id for s in dag.subtasks] == ["a"]

    @pytest.mark.asyncio
    async def test_a_cyclic_decomposition_is_rejected_with_the_generation(self) -> None:
        raw = plan_reply(("root", []), ("a", ["b"]), ("b", ["a"]))
        manager = manager_with([raw])
        with pytest.raises(InvalidPlan) as exc:
            await manager.decompose("t")
        assert "cyclic" in str(exc.value)
        assert "Raw generation:" in str(exc.value), "the reply was not attached"

    @pytest.mark.asyncio
    async def test_a_dangling_decomposition_is_rejected(self) -> None:
        manager = manager_with([plan_reply(("a", []), ("b", ["nope"]))])
        with pytest.raises(InvalidPlan, match="nope"):
            await manager.decompose("t")

    @pytest.mark.asyncio
    async def test_prose_instead_of_json_fails_loudly(self) -> None:
        manager = manager_with(["Sure! Here is a plan: first, search for..."])
        with pytest.raises(ManagerProtocolError, match="not valid JSON"):
            await manager.decompose("t")

    @pytest.mark.asyncio
    async def test_an_empty_reply_fails_loudly(self) -> None:
        manager = manager_with(["   "])
        with pytest.raises(ManagerProtocolError, match="empty reply"):
            await manager.decompose("t")

    @pytest.mark.asyncio
    async def test_a_missing_subtasks_key_fails_loudly(self) -> None:
        manager = manager_with(['{"plan": []}'])
        with pytest.raises(ManagerProtocolError) as exc:
            await manager.decompose("t")
        assert "unexpected top-level key" in str(exc.value)

    @pytest.mark.asyncio
    async def test_an_invented_subtask_field_fails_loudly(self) -> None:
        raw = json.dumps(
            {"subtasks": [{"id": "a", "instruction": "x", "depends_on": [],
                           "priority": "high"}]}
        )
        manager = manager_with([raw])
        with pytest.raises(ManagerProtocolError) as exc:
            await manager.decompose("t")
        assert "schema" in str(exc.value) and "Raw generation:" in str(exc.value)

    @pytest.mark.asyncio
    async def test_the_model_is_swappable_and_defaults_to_the_client(self) -> None:
        default = Manager(FakeChatClient([plan_reply(("a", []))]))
        assert default.model == "fake-model"

        client = FakeChatClient([plan_reply(("a", []))])
        stronger = Manager(client, ManagerConfig(model="a-much-better-model"))
        assert stronger.model == "a-much-better-model"
        await stronger.decompose("t")
        assert client.calls[0]["model"] == "a-much-better-model"


class TestPrompts:
    def test_the_example_we_show_the_model_passes_our_own_validator(self) -> None:
        """The worked example in the decompose prompt must be a legal plan.

        Otherwise the prompt teaches the model to emit something the validator
        rejects, and the resulting failures look like model problems.
        """
        rendered = DECOMPOSE_SYSTEM.format(max_subtasks=6)
        assert "{{" not in rendered and "}}" not in rendered

        start = rendered.index('{"subtasks"')
        example = rendered[start : rendered.rindex("]}") + 2]
        dag = DAG(subtasks=json.loads(example)["subtasks"])
        assert len(dag.subtasks) == 3
        assert len(dag.ready()) == 2, "the example should show parallel subtasks"

    def test_the_replan_prompt_states_the_remaining_budget(self) -> None:
        """The model cannot keep to a ceiling it was not told."""
        assert "at most 7 edit(s)" in REPLAN_SYSTEM.format(remaining=7)

    def test_the_replan_example_is_a_reply_we_would_accept(self) -> None:
        """Same discipline as the decompose example: the worked example must
        parse as a Replan AND apply cleanly, or the prompt teaches the model to
        emit something our own validator rejects."""
        rendered = REPLAN_SYSTEM.format(remaining=4)
        assert "{{" not in rendered and "}}" not in rendered

        start = rendered.index('{"reason"')
        example = json.loads(rendered[start : rendered.index('"remove": []}') + len('"remove": []}')])
        replan = Replan.model_validate(example)
        added = replan.add[0]
        assert added.retry_of == "find_price", "the example must show the field it teaches"
        assert added.depends_on == [], "a retry waits for nothing -- lineage is not a dependency"

        dag = dag_of(("find_price", []))
        dag.get("find_price").status = "failed"
        edited = dag.apply(replan)
        assert edited.get("find_price_direct").retry_of == "find_price"

    def test_the_replan_prompt_forbids_superseding_something_it_removes(self) -> None:
        """The one way to get a retry_of edit rejected, so the prompt says it."""
        rendered = REPLAN_SYSTEM.format(remaining=4)
        assert "never point it at an id you are also removing" in rendered

    def test_the_manager_sees_lineage_it_already_declared(self) -> None:
        """Otherwise a third attempt gets declared against the original rather
        than against the second, and the chain reads as a fork."""
        dag = DAG(subtasks=[
            Subtask(id="find", instruction="find it", status="failed"),
            Subtask(id="find_again", instruction="again", retry_of="find", status="failed"),
        ])
        message = _replan_user_message(dag, [], remaining=4)
        assert "(retry of: find)" in message


class TestManagerReplan:
    @pytest.mark.asyncio
    async def test_no_change_is_reported_as_none(self) -> None:
        manager = manager_with([NO_CHANGE])
        assert await manager.replan(dag_of(("a", [])), []) is None
        assert manager.budget.spent == 0

    @pytest.mark.asyncio
    async def test_edits_are_returned_and_counted(self) -> None:
        manager = manager_with([replan_reply(add=[("b", ["a"])], remove=[])])
        replan = await manager.replan(dag_of(("a", [])), [])
        assert replan is not None
        assert replan.edits == 1
        # Not yet charged: the strategy charges only once the edit is accepted.
        assert manager.budget.spent == 0

    @pytest.mark.asyncio
    async def test_results_reach_the_prompt_including_failures(self) -> None:
        client = FakeChatClient([NO_CHANGE])
        manager = Manager(client)
        await manager.replan(
            dag_of(("a", []), ("b", [])),
            [
                SubtaskOutcome(id="a", instruction="do a", status="done",
                               answer="the price is 51.77"),
                SubtaskOutcome(id="b", instruction="do b", status="failed",
                               error="hit a captcha", num_steps=4, num_errors=3),
            ],
        )
        user = client.calls[0]["user"]
        assert "the price is 51.77" in user
        assert "hit a captcha" in user, "the manager cannot plan around unseen failures"

    @pytest.mark.asyncio
    async def test_a_replan_costing_more_than_the_budget_is_refused_whole(self) -> None:
        manager = manager_with(
            [replan_reply(add=[("x", []), ("y", []), ("z", [])])], planning_budget=2
        )
        assert await manager.replan(dag_of(("a", [])), []) is None
        assert manager.budget.spent == 0
        assert manager.budget.refused == 3
        assert "refused" in manager.history[-1]["outcome"]


# ---------------------------------------------------------------------------
# 3. the budget as an ablation
# ---------------------------------------------------------------------------
class TestPlanningBudget:
    @pytest.mark.asyncio
    async def test_budget_zero_still_decomposes_but_never_replans(self) -> None:
        """The B=0 ablation: planning happens, replanning does not."""
        client = FakeChatClient([plan_reply(("a", []), ("b", []))])
        manager = Manager(client, ManagerConfig(planning_budget=0))

        dag = await manager.decompose("t")
        assert len(dag.subtasks) == 2, "decomposition must still run at B=0"

        assert await manager.replan(dag, []) is None
        # And crucially: no call was made, so B=0 costs nothing to replan.
        assert client.num_calls == 1
        assert manager.history[-1]["called_model"] is False

    @pytest.mark.asyncio
    async def test_replanning_stops_once_the_budget_is_spent(self) -> None:
        manager = manager_with(
            [replan_reply(add=[("x", [])]), replan_reply(add=[("y", [])])],
            planning_budget=1,
        )
        dag = dag_of(("a", []))

        first = await manager.replan(dag, [])
        assert first is not None
        manager.record_applied(first, wave=1)
        assert manager.budget.remaining == 0

        # Second time round the budget is gone, so the model is not even asked.
        assert await manager.replan(dag, []) is None
        assert manager.client.num_calls == 1

    def test_the_ledger_reports_what_was_spent(self) -> None:
        manager = manager_with([], planning_budget=10)
        manager.record_applied(
            Replan(add=[Subtask(id="x", instruction="x")], reason="r"), wave=1
        )
        assert manager.budget.as_dict() == {
            "limit": 10, "spent": 1, "remaining": 9, "refused_edits": 0
        }


# ---------------------------------------------------------------------------
# 4. the execution loop -- ordering, failure, budget, reporting
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_dependencies_are_respected_across_waves(tmp_path: Path) -> None:
    """A dependent subtask must not launch before its dependency completes.

    Asserted on launch ORDER, not on completion: a flat batch would also finish
    everything, so "they all ran" says nothing about the graph being honoured.
    """
    order: list[str] = []
    manager = manager_with(
        [plan_reply(("a", []), ("b", []), ("c", ["a", "b"]))],
        planning_budget=0,  # keep the test about ordering, not replanning
    )

    result = await orchestrate(
        task="t",
        strategy=DagStrategy(manager),
        policy_factory=lambda i: RecordingPolicy(i, order),
        run_config=run_config(tmp_path),
        orchestrator_config=OrchestratorConfig(max_concurrent_browsers=3),
    )

    assert order.index("c") > order.index("a")
    assert order.index("c") > order.index("b")
    # a and b are independent, so they belong to one wave and c to the next.
    assert set(order[:2]) == {"a", "b"} and order[2] == "c"

    details = result.details
    assert [w["subtasks"] for w in details["waves"]] == [["a", "b"], ["c"]]
    assert result.answer == "did c", "the answer came from somewhere other than the leaf"
    assert result.num_succeeded == 3


@pytest.mark.asyncio
async def test_a_dependent_receives_its_dependency_answer_as_context(
    tmp_path: Path,
) -> None:
    """The reason dependencies exist at all: passing findings downstream."""
    seen: list[str] = []

    class ContextSpy(RecordingPolicy):
        async def predict(self, task, history, screenshot, page):
            if self.inner is None:
                seen.append(task)
            return await super().predict(task, history, screenshot, page)

    manager = manager_with([plan_reply(("a", []), ("b", ["a"]))], planning_budget=0)
    await orchestrate(
        task="t",
        strategy=DagStrategy(manager),
        policy_factory=lambda i: ContextSpy(i, [], answers={"a": "the answer is 42"}),
        run_config=run_config(tmp_path),
    )

    downstream = next(t for t in seen if t.startswith("do b"))
    assert "the answer is 42" in downstream
    assert "a:" in downstream
    upstream = next(t for t in seen if t.startswith("do a"))
    assert "the answer is 42" not in upstream, "context leaked into the wrong subtask"


@pytest.mark.asyncio
async def test_a_subtask_failure_does_not_sink_the_run(tmp_path: Path) -> None:
    """One subtask crashes: its siblings finish, its dependents block, the run
    still reports."""
    manager = manager_with(
        [plan_reply(("good", []), ("bad", []), ("needs_bad", ["bad"]))],
        planning_budget=0,
    )

    result = await orchestrate(
        task="t",
        strategy=DagStrategy(manager),
        policy_factory=lambda i: RecordingPolicy(i, [], crash_on="bad"),
        run_config=run_config(tmp_path),
    )

    final = {s["id"]: s["status"] for s in result.details["final_dag"]["subtasks"]}
    assert final["good"] == "done"
    assert final["bad"] == "failed"
    # Blocked, not failed: nobody ran it, so calling it a failure would be a lie.
    assert final["needs_bad"] == "blocked"

    # The run still produced an answer, from the part of the graph that worked.
    assert result.answer is not None and "did good" in result.answer
    # And the failure is visible rather than swallowed.
    assert any(s.status != "done" for s in result.sessions)
    assert result.details["final_dag"]["counts"]["failed"] == 1


@pytest.mark.asyncio
async def test_manager_reroutes_around_a_blocked_join(tmp_path: Path) -> None:
    """The pricecheck scenario, end to end: a failed lookup blocks the join,
    the manager removes the blocked join and adds one over the inputs that did
    arrive, and the edit is APPLIED.

    This is the exact edit the old guard refused ("cannot remove: already
    blocked"), which cost a live run its comparison. Here the run must end with
    the re-routed join done and reporting.
    """
    manager = manager_with(
        [
            plan_reply(
                ("price_a", []), ("price_b", []), ("price_c", []),
                ("price_missing", []),
                ("compare_all", ["price_a", "price_b", "price_c", "price_missing"]),
            ),
            replan_reply(
                add=[("compare_found", ["price_a", "price_b", "price_c"])],
                remove=["compare_all"],
                reason="price_missing failed; compare the three prices that were found",
            ),
            NO_CHANGE,
            NO_CHANGE,
        ],
        planning_budget=10,
    )

    result = await orchestrate(
        task="compare four book prices, one of which does not exist",
        strategy=DagStrategy(manager),
        policy_factory=lambda i: RecordingPolicy(i, [], crash_on="price_missing"),
        run_config=run_config(tmp_path),
    )

    final = {s["id"]: s["status"] for s in result.details["final_dag"]["subtasks"]}
    # The blocked join is gone; the re-routed one ran to completion.
    assert "compare_all" not in final, "the blocked join was not removed"
    assert final["compare_found"] == "done"
    assert final["price_missing"] == "failed"
    assert [final[k] for k in ("price_a", "price_b", "price_c")] == ["done"] * 3

    # The re-route was APPLIED, not refused, and charged (1 add + 1 remove).
    applied = [r for r in result.details["replans"] if r.get("applied")]
    assert len(applied) == 1
    assert applied[0]["remove"] == ["compare_all"]
    assert [s["id"] for s in applied[0]["add"]] == ["compare_found"]
    assert result.details["budget"]["spent"] == 2

    # And the run's answer comes from the re-routed join, not three loose prices.
    assert result.answer == "did compare_found"
    assert result.contributing_indices  # the join's agent is credited


@pytest.mark.asyncio
async def test_a_declared_retry_survives_into_run_json(tmp_path: Path) -> None:
    """The whole point of the field, end to end.

    Two attempts at a lookup that fails both times, with the second declared
    as superseding the first. ``run.json`` must carry that declaration, because
    it is the only thing that distinguishes "a retry of the lookup" from "an
    unrelated subtask that also failed" -- which is what a reader of the graph
    could not tell before.
    """
    manager = manager_with(
        [
            plan_reply(("price_a", []), ("price_missing", []),
                       ("compare", ["price_a", "price_missing"])),
            json.dumps({
                "reason": "price_missing ran out of steps; trying it another way",
                "add": [
                    {"id": "price_missing_direct", "instruction": "do price_missing_direct",
                     "depends_on": [], "retry_of": "price_missing"},
                    {"id": "compare_v2", "instruction": "do compare_v2",
                     "depends_on": ["price_a", "price_missing_direct"]},
                ],
                "remove": ["compare"],
            }),
            replan_reply(add=[("compare_found", ["price_a"])], remove=["compare_v2"],
                         reason="both attempts failed; compare what arrived"),
            NO_CHANGE,
            NO_CHANGE,
        ],
        planning_budget=10,
    )

    result = await orchestrate(
        task="two prices, one of which cannot be found",
        strategy=DagStrategy(manager),
        policy_factory=lambda i: RecordingPolicy(i, [], crash_on="price_missing"),
        run_config=run_config(tmp_path),
    )

    final = {s["id"]: s for s in result.details["final_dag"]["subtasks"]}
    assert final["price_missing_direct"]["retry_of"] == "price_missing"
    assert final["price_missing"]["retry_of"] is None
    assert final["price_a"]["retry_of"] is None

    # The attempts ran in different waves -- which is exactly why the graph
    # needs the field: nothing about wave 1 and wave 2 links them.
    assert final["price_missing"]["wave"] == 1
    assert final["price_missing_direct"]["wave"] == 2

    # And the declaration is in the replan record too, so the edit that
    # introduced it can be read back on its own.
    added = [a for r in result.details["replans"] for a in r.get("add", [])]
    assert {a["id"]: a.get("retry_of") for a in added}["price_missing_direct"] == "price_missing"


@pytest.mark.asyncio
async def test_a_retry_naming_a_removed_subtask_is_refused_without_sinking_the_run(
    tmp_path: Path,
) -> None:
    """A manager that supersedes a subtask AND removes it in the same reply is
    refused, recorded, and the run carries on -- the same containment every
    other invalid edit gets, not a new fatal path."""
    manager = manager_with(
        [
            plan_reply(("a", []), ("join", ["a"])),
            json.dumps({
                "reason": "replacing the join",
                "add": [{"id": "join_v2", "instruction": "do join_v2",
                         "depends_on": ["a"], "retry_of": "join"}],
                "remove": ["join"],
            }),
            NO_CHANGE,
            NO_CHANGE,
        ],
        planning_budget=10,
    )
    result = await orchestrate(
        task="t",
        strategy=DagStrategy(manager),
        policy_factory=lambda i: RecordingPolicy(i, []),
        run_config=run_config(tmp_path),
    )
    rejected = [r for r in result.details["replans"]
                if str(r.get("outcome", "")).startswith("rejected")]
    assert len(rejected) == 1
    assert "retries" in rejected[0]["outcome"]
    # Not billed, and the plan it already had still ran to completion.
    assert result.details["budget"]["spent"] == 0
    final = {s["id"]: s["status"] for s in result.details["final_dag"]["subtasks"]}
    assert final == {"a": "done", "join": "done"}

@pytest.mark.asyncio
async def test_replanning_adds_work_and_charges_the_budget(tmp_path: Path) -> None:
    manager = manager_with(
        [
            plan_reply(("a", [])),
            replan_reply(add=[("follow_up", ["a"])], reason="a turned up more to check"),
            NO_CHANGE,
        ],
        planning_budget=10,
    )

    result = await orchestrate(
        task="t",
        strategy=DagStrategy(manager),
        policy_factory=lambda i: RecordingPolicy(i, []),
        run_config=run_config(tmp_path),
    )

    ids = [s["id"] for s in result.details["final_dag"]["subtasks"]]
    assert ids == ["a", "follow_up"]
    assert result.details["budget"] == {
        "limit": 10, "spent": 1, "remaining": 9, "refused_edits": 0
    }
    growth = result.details["growth"]
    assert growth["initial_subtasks"] == 1 and growth["final_subtasks"] == 2
    assert growth["net_growth"] == 1 and growth["replans_applied"] == 1


@pytest.mark.asyncio
async def test_budget_exhaustion_stops_the_graph_growing(tmp_path: Path) -> None:
    """With B=1 the manager gets one edit, then is not consulted again.

    The fake would happily keep adding work forever; the budget is what stops
    it, and that is the property under test.
    """
    manager = manager_with(
        [
            plan_reply(("a", [])),
            replan_reply(add=[("b", ["a"])]),
            replan_reply(add=[("c", ["b"])]),
            replan_reply(add=[("d", ["c"])]),
        ],
        planning_budget=1,
    )

    result = await orchestrate(
        task="t",
        strategy=DagStrategy(manager),
        policy_factory=lambda i: RecordingPolicy(i, []),
        run_config=run_config(tmp_path),
    )

    ids = {s["id"] for s in result.details["final_dag"]["subtasks"]}
    assert ids == {"a", "b"}, "the graph grew past its edit budget"
    assert result.details["budget"]["remaining"] == 0
    # Two waves ran, but the manager was only asked while it could still afford
    # to answer -- the second consultation never reached the model.
    assert manager.client.num_calls == 2
    assert manager.history[-1]["called_model"] is False


@pytest.mark.asyncio
async def test_an_invalid_replan_is_refused_without_sinking_the_run(
    tmp_path: Path,
) -> None:
    """A bad edit is recorded and skipped; completed work is not thrown away."""
    manager = manager_with(
        [
            plan_reply(("a", [])),
            replan_reply(add=[("b", ["ghost"])], reason="depends on nothing real"),
            NO_CHANGE,
        ],
        planning_budget=10,
    )

    result = await orchestrate(
        task="t",
        strategy=DagStrategy(manager),
        policy_factory=lambda i: RecordingPolicy(i, []),
        run_config=run_config(tmp_path),
    )

    assert result.answer == "did a", "a bad replan lost the completed work"
    assert [s["id"] for s in result.details["final_dag"]["subtasks"]] == ["a"]
    rejected = [r for r in result.details["replans"] if "rejected" in str(r.get("outcome"))]
    assert len(rejected) == 1
    assert "ghost" in rejected[0]["outcome"], "the reason was not recorded"
    # Refused edits are not billed: the run did not get what it would have paid for.
    assert result.details["budget"]["spent"] == 0


@pytest.mark.asyncio
async def test_a_broken_replan_reply_does_not_lose_completed_waves(
    tmp_path: Path,
) -> None:
    """A manager that stops emitting JSON mid-run must not sink the run.

    Decompose failing is fatal -- there is no plan and nothing is spent. Replan
    failing at wave 2 is not: waves 1 and 2 are paid for and the remaining plan
    is still executable without further advice.
    """
    manager = manager_with(
        [
            plan_reply(("a", []), ("b", ["a"])),
            "I'm afraid I can't help with that.",  # not JSON, wave 1
            "```\nstill not json\n```",  # and again at wave 2
        ]
    )

    result = await orchestrate(
        task="t",
        strategy=DagStrategy(manager),
        policy_factory=lambda i: RecordingPolicy(i, []),
        run_config=run_config(tmp_path),
    )

    # The graph ran to completion on the plan it already had.
    assert {s["id"] for s in result.details["final_dag"]["subtasks"]} == {"a", "b"}
    assert result.answer == "did b"
    # And every failure is in the report, not swallowed.
    failed = [r for r in result.details["replans"] if "failed" in str(r.get("outcome"))]
    assert len(failed) == 2
    assert "ManagerProtocolError" in failed[0]["outcome"]
    # Nothing was billed against the planning budget for a reply we could not use.
    assert result.details["budget"]["spent"] == 0


@pytest.mark.asyncio
async def test_a_broken_decomposition_is_fatal(tmp_path: Path) -> None:
    """The counterpart: no plan means no run, and it says so."""
    manager = manager_with(["not json at all"])
    with pytest.raises(ManagerProtocolError):
        await orchestrate(
            task="t",
            strategy=DagStrategy(manager),
            policy_factory=lambda i: RecordingPolicy(i, []),
            run_config=run_config(tmp_path),
        )


@pytest.mark.asyncio
async def test_max_waves_stops_a_runaway_plan(tmp_path: Path) -> None:
    """A manager that keeps adding work is bounded even with budget to spare."""
    manager = manager_with(
        [plan_reply(("s0", []))]
        + [replan_reply(add=[(f"s{i}", [f"s{i - 1}"])]) for i in range(1, 8)],
        planning_budget=20,
        max_waves=3,
    )

    result = await orchestrate(
        task="t",
        strategy=DagStrategy(manager),
        policy_factory=lambda i: RecordingPolicy(i, []),
        run_config=run_config(tmp_path),
    )

    assert result.details["growth"]["waves"] == 3
    assert result.details["stopped_early"] is not None
    assert "still runnable" in result.details["stopped_early"]
    # The answer still comes from what did run.
    assert result.answer is not None


@pytest.mark.asyncio
async def test_run_json_carries_the_graph_the_budget_and_every_replan(
    tmp_path: Path,
) -> None:
    """The Phase 3 reporting contract, read back off disk."""
    manager = manager_with(
        [
            plan_reply(("a", []), ("b", [])),
            replan_reply(add=[("c", ["a"])], reason="worth a closer look"),
            NO_CHANGE,
        ]
    )

    result = await orchestrate(
        task="find the thing",
        strategy=DagStrategy(manager),
        policy_factory=lambda i: RecordingPolicy(i, []),
        run_config=run_config(tmp_path),
    )

    payload = json.loads((result.run_dir / "run.json").read_text())
    assert payload["strategy"] == "dag"
    details = payload["details"]

    # The pair of graphs is the measurement: growth is final minus initial.
    assert [s["id"] for s in details["initial_dag"]["subtasks"]] == ["a", "b"]
    assert {s["id"] for s in details["final_dag"]["subtasks"]} == {"a", "b", "c"}

    # Every replan, with its reason -- including the one that changed nothing.
    reasons = [r.get("reason") for r in details["replans"]]
    assert "worth a closer look" in reasons
    assert "the plan still holds" in reasons
    assert details["replans"][0]["applied"] is True

    assert details["budget"]["spent"] == 1
    assert details["growth"]["replan_rate"] == pytest.approx(0.5)  # 1 of 2 waves
    assert details["manager"]["model"] == "fake-model"

    # Per-agent directories keep the Phase 1 shape, and subtasks are labelled.
    assert [a["label"] for a in payload["agents"]][0].startswith("wave 1: ")
    assert (result.run_dir / "agent_0" / "steps.jsonl").exists()


@pytest.mark.asyncio
async def test_the_answer_composes_several_terminal_subtasks(tmp_path: Path) -> None:
    manager = manager_with([plan_reply(("left", []), ("right", []))], planning_budget=0)
    result = await orchestrate(
        task="t",
        strategy=DagStrategy(manager),
        policy_factory=lambda i: RecordingPolicy(
            i, [], answers={"left": "9.99", "right": "12.50"}
        ),
        run_config=run_config(tmp_path),
    )

    assert "left: 9.99" in result.answer
    assert "right: 12.50" in result.answer
    # Both leaves contributed, which is why the field is a list.
    assert sorted(result.contributing_indices) == [0, 1]


@pytest.mark.asyncio
async def test_the_runner_is_shared_across_waves(tmp_path: Path) -> None:
    """Waves reuse one Runner and one browser pool; indices keep incrementing."""
    manager = manager_with([plan_reply(("a", []), ("b", ["a"]))], planning_budget=0)
    async with Runner(
        policy_factory=lambda i: RecordingPolicy(i, []),
        run_config=run_config(tmp_path),
    ) as runner:
        outcome = await DagStrategy(manager).run("t", runner)

    assert [s.index for s in outcome.sessions] == [0, 1]
    assert (runner.run_dir / "agent_1" / "steps.jsonl").exists()


# ---------------------------------------------------------------------------
# 5. the LLM judge
# ---------------------------------------------------------------------------
def candidate(index: int, **kw) -> Candidate:
    base = dict(task="t", answer=f"a{index}", status="done", num_steps=3, num_errors=0)
    base.update(kw)
    return Candidate(index=index, **base)  # type: ignore[arg-type]


class TestLLMJudge:
    @pytest.mark.asyncio
    async def test_it_picks_the_index_the_model_names(self) -> None:
        client = FakeChatClient(['{"index": 2, "reason": "most specific answer"}'])
        verdict = await LLMJudge(client).choose(
            "t", [candidate(0), candidate(1), candidate(2)]
        )
        assert verdict.index == 2
        assert verdict.reason == "most specific answer"

    @pytest.mark.asyncio
    async def test_it_sees_answers_statuses_and_effort(self) -> None:
        client = FakeChatClient(['{"index": 0, "reason": "r"}'])
        await LLMJudge(client).choose(
            "how much is it?",
            [
                candidate(0, answer="51.77"),
                candidate(1, answer=None, status="crashed", error="boom",
                          num_steps=9, num_errors=4),
            ],
        )
        user = client.calls[0]["user"]
        assert "how much is it?" in user
        assert "51.77" in user
        assert "crashed" in user and "boom" in user
        assert "9 (4 errored)" in user

    @pytest.mark.asyncio
    async def test_null_is_a_legitimate_verdict(self) -> None:
        client = FakeChatClient(['{"index": null, "reason": "all of them failed"}'])
        verdict = await LLMJudge(client).choose("t", [candidate(0, status="crashed")])
        assert verdict.index is None
        assert "failed" in verdict.reason

    @pytest.mark.asyncio
    async def test_no_candidates_costs_nothing(self) -> None:
        client = FakeChatClient([])
        verdict = await LLMJudge(client).choose("t", [])
        assert verdict.index is None
        assert client.num_calls == 0, "the judge spent money on an empty run"

    @pytest.mark.asyncio
    async def test_the_model_is_swappable(self) -> None:
        client = FakeChatClient(['{"index": 0, "reason": "r"}'])
        await LLMJudge(client, JudgeConfig(model="stronger")).choose("t", [candidate(0)])
        assert client.calls[0]["model"] == "stronger"

    def test_an_index_that_does_not_exist_is_rejected(self) -> None:
        with pytest.raises(JudgeProtocolError, match="not among the candidates"):
            parse_verdict('{"index": 7, "reason": "r"}', [0, 1, 2])

    def test_a_fenced_reply_is_tolerated(self) -> None:
        verdict = parse_verdict('```json\n{"index": 1, "reason": "r"}\n```', [0, 1])
        assert verdict.index == 1

    def test_prose_fails_loudly(self) -> None:
        with pytest.raises(JudgeProtocolError, match="not valid JSON"):
            parse_verdict("I think agent 1 did best.", [0, 1])

    def test_a_non_integer_index_fails_loudly(self) -> None:
        for bad in ('"1"', "1.5", "true"):
            with pytest.raises(JudgeProtocolError):
                parse_verdict('{"index": %s, "reason": "r"}' % bad, [0, 1])
