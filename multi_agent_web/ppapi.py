"""One HTTP client for the PP API, shared by every component that talks to it.

Phase 2 had exactly one caller -- ``QwenPolicy`` -- so the transport lived
inside it. Phase 3 adds two more (the manager LLM and the LLM judge), and three
copies of "POST, retry on 429/5xx with jittered backoff, count the spend" is
three places for the retry policy to drift apart and three ledgers to reconcile
when someone asks what a run cost. So the transport moved here and the policy
now calls it; nothing about the wire contract changed.

WHAT LIVES HERE AND WHY
=======================
``CallBudget`` is the reason this module is worth having. It is a single ledger
covering **every** call a run makes, agents and manager alike. That matters for
Phase 3 specifically: a DAG run spends on two axes at once -- N browsing agents
plus a manager that decomposes and replans -- and a ceiling that only bounded
the agents would not bound the run. One budget, one number in ``run.json``.

Two ways to call, because there are two kinds of caller:

* ``chat(payload)`` -- full OpenAI-shaped body, returns the parsed response.
  What the vision policy needs: it sends images and reads ``usage``.
* ``complete(system=..., user=...)`` -- text in, text out. What the manager and
  judge need. They reason over text alone; neither has any business assembling
  a messages array.

``ChatClient`` is the abstract half of that second form, and it exists to be
faked. Every Phase 3 test drives the manager, the DAG strategy and the judge
through a stub that returns canned strings -- no key, no network, no spend --
which is only possible because the seam is this narrow.

The API contract itself (the required ``/v1``, the ``Bearer`` header, data-URL
images) is documented on ``QwenConfig`` in ``config.py``, verified against the
live API rather than taken from the vendor docs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import QwenConfig

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


class PPAPIError(RuntimeError):
    """The API call failed, or its reply did not match what we asked for.

    Carries the raw generation whenever there was one, so a mismatch is
    diagnosable from the log line alone rather than needing a re-run.
    """


class CallBudgetExceeded(RuntimeError):
    """The run hit its API call ceiling and was stopped deliberately."""


@dataclass
class CallBudget:
    """Shared spend guard and usage ledger for one run.

    Shared on purpose. Every agent gets its own *policy* (isolation), but they
    all spend from one pot, so the budget object is passed to each policy the
    factory builds -- and, in Phase 3, to the manager and the judge as well. A
    4-agent best-of-N is dozens of image requests against someone else's
    credit; an unbounded retry loop is the failure mode worth engineering
    against.

    Every HTTP attempt counts, retries included -- a retried call is still a
    billed call.

    ``by_agent`` is keyed by agent index. The manager and judge pass no index,
    so their calls land in ``calls`` and in the token totals but not in the
    per-agent breakdown; ``non_agent_calls`` is the difference, and is what
    "the manager's share" means in a DAG run.
    """

    limit: int
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    image_tokens: int = 0
    total_tokens: int = 0
    retries: int = 0
    by_agent: dict[int, int] = field(default_factory=dict)

    def reserve(self, agent_index: int | None = None) -> None:
        if self.calls >= self.limit:
            raise CallBudgetExceeded(
                f"per-run API call budget exhausted: {self.calls}/{self.limit} calls "
                f"already made ({self.total_tokens} tokens). Stopping rather than "
                f"spending further. Raise QwenConfig.max_calls_per_run if this run "
                f"legitimately needs more."
            )
        self.calls += 1
        if agent_index is not None:
            self.by_agent[agent_index] = self.by_agent.get(agent_index, 0) + 1

    def record_usage(self, usage: dict[str, Any] | None) -> None:
        if not isinstance(usage, dict):
            return
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.total_tokens += int(usage.get("total_tokens") or 0)
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict):
            self.reasoning_tokens += int(details.get("reasoning_tokens") or 0)
        prompt_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            self.image_tokens += int(prompt_details.get("image_tokens") or 0)

    @property
    def non_agent_calls(self) -> int:
        """Calls made by something other than a browsing agent (manager, judge)."""
        return self.calls - sum(self.by_agent.values())

    def as_dict(self) -> dict[str, Any]:
        """Safe to serialize into run.json -- contains no credentials."""
        return {
            "calls": self.calls,
            "limit": self.limit,
            "retries": self.retries,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "image_tokens": self.image_tokens,
            "total_tokens": self.total_tokens,
            "calls_by_agent": dict(sorted(self.by_agent.items())),
            # Named for what it is rather than "manager": the judge lands here
            # too, and a field that lies about its contents is worse than a
            # vague one.
            "non_agent_calls": self.non_agent_calls,
        }


# ---------------------------------------------------------------------------
# reply parsing, shared by every JSON-emitting caller
# ---------------------------------------------------------------------------
def strip_code_fence(text: str) -> tuple[str, bool]:
    """Remove a wrapping ```json ... ``` fence. Returns (text, was_fenced).

    General models fence JSON regardless of what the prompt says, so this is
    tolerated everywhere -- and logged by the caller, because a model that
    started fencing may have started doing other things too.
    """
    match = _FENCE_RE.match(text)
    if match:
        return match.group(1).strip(), True
    return text.strip(), False


def parse_json_object(raw: str, what: str, error: type[Exception] = PPAPIError) -> dict:
    """``raw`` -> a JSON object, or raise ``error`` with the generation attached.

    The same three failure modes turn up for every JSON-emitting caller -- empty
    reply, prose instead of JSON, a JSON array or string where an object was
    asked for -- so they are diagnosed once, here. No coercion and no repair:
    ``what`` names the caller so the message says which prompt to go and fix.
    """
    if not raw or not raw.strip():
        raise error(f"{what}: the model returned an empty reply")

    text, fenced = strip_code_fence(raw)
    if fenced:
        logger.info("%s: stripped a markdown code fence from the reply", what)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise error(
            f"{what}: reply is not valid JSON ({exc}).\nRaw generation: {raw}"
        ) from exc

    if not isinstance(payload, dict):
        raise error(
            f"{what}: expected a JSON object, got {type(payload).__name__}."
            f"\nRaw generation: {raw}"
        )
    return payload


# ---------------------------------------------------------------------------
# the client
# ---------------------------------------------------------------------------
class ChatClient(ABC):
    """Text in, text out. The whole surface the manager and judge need.

    Deliberately smaller than the API: no images, no messages array, no usage.
    A component that only reasons over text should not be able to reach the
    parts of the protocol it has no business touching, and a test double for an
    interface this size is a dozen lines with no HTTP anywhere in it.
    """

    name: str = "chat"
    #: Model used when a caller does not name one.
    default_model: str = ""

    @abstractmethod
    async def complete(
        self,
        system: str,
        user: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return the assistant's message content, unparsed."""

    async def close(self) -> None:
        """Release the underlying connection, if any."""


def _request_id(response: httpx.Response, body: dict[str, Any] | None) -> str:
    """Whatever identifier the API gives us, for support requests."""
    for header in ("x-request-id", "x-requestid", "request-id", "cf-ray"):
        if header in response.headers:
            return response.headers[header]
    if isinstance(body, dict) and isinstance(body.get("id"), str):
        return body["id"]
    return "<none>"


class PPAPIClient(ChatClient):
    """The real thing: OpenAI-compatible chat completions over HTTPS.

    Retries only 429 and 5xx. A 4xx other than 429 means our request is wrong,
    and retrying it just spends money repeating the same mistake.
    """

    name = "ppapi"

    def __init__(
        self,
        config: QwenConfig,
        budget: CallBudget | None = None,
        agent_index: int | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        # A private budget when unshared, so even a one-caller run is capped.
        self.budget = budget or CallBudget(limit=config.max_calls_per_run)
        self.agent_index = agent_index
        self.default_model = config.model
        self._client = client
        self._owns_client = client is None

    # --- lifecycle ---------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.config.timeout_s)
        return self._client

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # --- HTTP --------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        # get_secret_value() is the only place the credential is unwrapped.
        return {
            "Authorization": f"Bearer {self.config.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        raw = response.headers.get("retry-after")
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            return None  # HTTP-date form; fall back to computed backoff

    def _backoff(self, attempt: int) -> float:
        """Exponential with jitter, capped. Jitter avoids a thundering herd
        when several agents are throttled by the same 429."""
        delay = min(
            self.config.backoff_base_s * (2 ** (attempt - 1)),
            self.config.backoff_cap_s,
        )
        return delay * (0.5 + random.random() / 2)

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST one chat-completions body. Never logs the key or image data."""
        last_error: str = "no attempt was made"

        for attempt in range(1, self.config.max_attempts + 1):
            self.budget.reserve(self.agent_index)  # raises CallBudgetExceeded
            if attempt > 1:
                self.budget.retries += 1

            try:
                response = await self._get_client().post(
                    self.config.chat_url, headers=self._headers(), json=payload
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "attempt %d/%d: transport error: %s",
                    attempt, self.config.max_attempts, last_error,
                )
                if attempt < self.config.max_attempts:
                    await asyncio.sleep(self._backoff(attempt))
                continue

            if response.status_code == 429 or response.status_code >= 500:
                wait = self._retry_after(response) or self._backoff(attempt)
                last_error = (
                    f"HTTP {response.status_code} "
                    f"(request id {_request_id(response, None)}): {response.text[:300]}"
                )
                logger.warning(
                    "attempt %d/%d: %s -- retrying in %.1fs",
                    attempt, self.config.max_attempts, last_error, wait,
                )
                if attempt < self.config.max_attempts:
                    await asyncio.sleep(wait)
                continue

            if response.status_code != 200:
                # 4xx other than 429 means our request is wrong. Retrying would
                # just spend money repeating the same mistake.
                raise PPAPIError(
                    f"HTTP {response.status_code} from the API "
                    f"(request id {_request_id(response, None)}). "
                    f"Not retried -- a 4xx means the request itself is wrong.\n"
                    f"Body: {response.text[:800]}"
                )

            try:
                body = response.json()
            except ValueError as exc:
                # A 200 carrying HTML is exactly what the host returns when /v1
                # is missing from the URL, so this is a real failure mode.
                raise PPAPIError(
                    f"HTTP 200 but the body is not JSON ({exc}). This usually "
                    f"means the URL is wrong: {self.config.chat_url} should end "
                    f"in /v1/chat/completions.\nBody starts: {response.text[:200]}"
                ) from exc

            self.budget.record_usage(body.get("usage"))
            return body

        raise PPAPIError(
            f"giving up after {self.config.max_attempts} attempts. "
            f"Last error: {last_error}"
        )

    # --- text convenience --------------------------------------------------

    async def complete(
        self,
        system: str,
        user: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        body = await self.chat(
            {
                "model": model or self.default_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": (
                    self.config.temperature if temperature is None else temperature
                ),
                "max_tokens": max_tokens or self.config.max_tokens,
                "stream": False,
            }
        )
        return extract_message_text(body)


def extract_message_text(body: dict[str, Any]) -> str:
    """Pull the assistant's content out of a response, or say why it is missing.

    Split out because the two empty-content cases are worth distinguishing and
    both are easy to misread as "the model had nothing to say". This model
    separates reasoning from content, so an empty ``content`` beside a full
    ``reasoning_content`` means ``max_tokens`` was consumed thinking -- a config
    problem, not a model failure.
    """
    try:
        message = body["choices"][0]["message"]
        raw = message.get("content") or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise PPAPIError(
            f"unexpected response shape ({exc}). Body: {json.dumps(body)[:600]}"
        ) from exc

    if not raw.strip():
        reasoning = (message.get("reasoning_content") or "")[:300]
        raise PPAPIError(
            "model returned empty content. This usually means max_tokens was "
            "consumed by reasoning before any answer was emitted; raise "
            f"max_tokens.\nreasoning_content starts: {reasoning!r}"
        )
    return raw


__all__ = [
    "CallBudget",
    "CallBudgetExceeded",
    "ChatClient",
    "PPAPIClient",
    "PPAPIError",
    "extract_message_text",
    "parse_json_object",
    "strip_code_fence",
]
