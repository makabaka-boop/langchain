"""Unit tests for `RunnableQuorum`."""

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

import pytest
from typing_extensions import override

if TYPE_CHECKING:
    from pydantic import BaseModel

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.exceptions import LangChainException
from langchain_core.load import dumps, loads
from langchain_core.runnables import (
    QuorumNotMetError,
    RunnableConfig,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableQuorum,
    RunnableSerializable,
)

# --------------------------------------------------------------------------- #
# Construction / validation
# --------------------------------------------------------------------------- #


def test_init_requires_at_least_one_branch() -> None:
    with pytest.raises(ValueError, match="at least one branch"):
        RunnableQuorum({}, quorum__=1)

    with pytest.raises(ValueError, match="at least one branch"):
        RunnableQuorum(quorum__=1)


@pytest.mark.parametrize("bad_quorum", [0, -1, 3])
def test_init_validates_quorum_range(bad_quorum: int) -> None:
    with pytest.raises(ValueError, match="quorum"):
        RunnableQuorum(
            {"a": RunnableLambda(lambda x: x), "b": RunnableLambda(lambda x: x)},
            quorum__=bad_quorum,
        )


@pytest.mark.parametrize("bad_quorum", [True, False, 1.0, "1"])
def test_init_validates_quorum_type(bad_quorum: Any) -> None:
    with pytest.raises(TypeError, match="quorum"):
        RunnableQuorum(
            {"a": RunnableLambda(lambda x: x), "b": RunnableLambda(lambda x: x)},
            quorum__=bad_quorum,
        )


def test_init_accepts_kwargs_and_coerces_callables() -> None:
    q: RunnableQuorum[Any] = RunnableQuorum(
        a=lambda x: x + 1,
        b=lambda x: x + 2,
        quorum__=2,
    )
    assert q.quorum__ == 2
    assert set(q.steps__) == {"a", "b"}
    assert q.invoke(10) == {"a": 11, "b": 12}


def test_init_default_quorum_is_one() -> None:
    q = RunnableQuorum({"a": RunnableLambda(lambda x: x)})
    assert q.quorum__ == 1


def test_name_and_repr() -> None:
    q: RunnableQuorum[Any] = RunnableQuorum(
        {"a": RunnablePassthrough(), "b": RunnablePassthrough()}, quorum__=1
    )
    assert q.get_name() == "RunnableQuorum<a,b>"
    assert "RunnableQuorum" in repr(q)
    assert "quorum__=1" in repr(q)


# --------------------------------------------------------------------------- #
# Sync invoke
# --------------------------------------------------------------------------- #


def test_invoke_returns_named_results() -> None:
    q: RunnableQuorum[Any] = RunnableQuorum(
        {"a": RunnableLambda(lambda x: x + 1), "b": RunnableLambda(lambda x: x + 2)},
        quorum__=2,
    )
    assert q.invoke(1) == {"a": 2, "b": 3}


def test_invoke_falsy_values_count_as_success() -> None:
    q: RunnableQuorum[Any] = RunnableQuorum(
        {
            "empty_list": RunnableLambda(lambda _: []),
            "none": RunnableLambda(lambda _: None),
            "zero": RunnableLambda(lambda _: 0),
            "false": RunnableLambda(lambda _: False),
        },
        quorum__=4,
    )
    assert q.invoke("x") == {"empty_list": [], "none": None, "zero": 0, "false": False}


def test_invoke_each_branch_gets_the_same_input() -> None:
    seen: list[object] = []

    def record(x: object) -> str:
        seen.append(x)
        return "ok"

    sentinel = object()
    RunnableQuorum(
        {f"b{i}": RunnableLambda(record) for i in range(3)}, quorum__=3
    ).invoke(sentinel)
    assert seen == [sentinel, sentinel, sentinel]


def test_invoke_collects_in_completion_order() -> None:
    def make(delay: float) -> Callable[[Any], float]:
        def fn(_: Any) -> float:
            time.sleep(delay)
            return delay

        return fn

    q = RunnableQuorum(
        {
            "slow": RunnableLambda(make(0.25)),
            "fast": RunnableLambda(make(0.02)),
            "middle": RunnableLambda(make(0.12)),
        },
        quorum__=3,
    )
    assert list(q.invoke(1)) == ["fast", "middle", "slow"]


def test_invoke_ordinary_exception_does_not_abort() -> None:
    def boom(_: Any) -> str:
        msg = "boom"
        raise ValueError(msg)

    q = RunnableQuorum(
        {"bad": RunnableLambda(boom), "good": RunnableLambda(lambda _: "ok")},
        quorum__=1,
    )
    assert q.invoke(1) == {"good": "ok"}


def test_invoke_quorum_not_met_carries_results_and_errors() -> None:
    def boom_runtime(_: Any) -> str:
        msg = "rt"
        raise RuntimeError(msg)

    def boom_value(_: Any) -> str:
        msg = "ve"
        raise ValueError(msg)

    q = RunnableQuorum(
        {
            "ok": RunnableLambda(lambda _: "result"),
            "rt": RunnableLambda(boom_runtime),
            "ve": RunnableLambda(boom_value),
        },
        quorum__=2,
    )
    with pytest.raises(QuorumNotMetError) as exc_info:
        q.invoke(1)

    err = exc_info.value
    assert isinstance(err, LangChainException)
    assert err.quorum == 2
    assert err.results == {"ok": "result"}
    assert set(err.exceptions) == {"rt", "ve"}
    assert isinstance(err.exceptions["rt"], RuntimeError)
    assert isinstance(err.exceptions["ve"], ValueError)
    message = str(err)
    assert "required 2" in message
    assert "RuntimeError" in message
    assert "ValueError" in message


def test_invoke_failed_branch_frees_slot_for_queued_branch() -> None:
    started: list[str] = []
    lock = threading.Lock()

    def make(
        name: str, *, fail: bool = False, delay: float = 0.0
    ) -> Callable[[Any], str]:
        def fn(_: Any) -> str:
            with lock:
                started.append(name)
            if delay:
                time.sleep(delay)
            if fail:
                raise RuntimeError(name)
            return name

        return fn

    # Concurrency 1: b0 fails first, b1 must be launched into the freed slot;
    # b2 must stay queued forever once the quorum is met by b1.
    q = RunnableQuorum(
        {
            "b0": RunnableLambda(make("b0", fail=True)),
            "b1": RunnableLambda(make("b1")),
            "b2": RunnableLambda(make("b2", delay=5.0)),
        },
        quorum__=1,
    )
    assert q.invoke(1, config=RunnableConfig(max_concurrency=1)) == {"b1": "b1"}
    assert started == ["b0", "b1"]


def test_invoke_returns_without_waiting_for_running_or_queued_branches() -> None:
    started: list[str] = []
    finished: list[str] = []
    lock = threading.Lock()
    release = threading.Event()

    def make(name: str, delay: float | None = None) -> Callable[[Any], str]:
        def fn(_: Any) -> str:
            with lock:
                started.append(name)
            if delay is None:
                release.wait(timeout=5.0)
            else:
                time.sleep(delay)
            with lock:
                finished.append(name)
            return name

        return fn

    # b0 finishes quickly; b1 is already running (slow); b2..b4 are queued.
    q = RunnableQuorum(
        {
            "b0": RunnableLambda(make("b0", delay=0.05)),
            "b1": RunnableLambda(make("b1")),
            "b2": RunnableLambda(make("b2")),
            "b3": RunnableLambda(make("b3")),
            "b4": RunnableLambda(make("b4")),
        },
        quorum__=1,
    )
    try:
        start = time.monotonic()
        output = q.invoke(1, config=RunnableConfig(max_concurrency=2))
        elapsed = time.monotonic() - start

        assert output == {"b0": "b0"}
        # invoke must not wait for the in-flight slow branch
        assert elapsed < 2.0, elapsed
        assert set(started) == {"b0", "b1"}, started
        assert finished == ["b0"], finished

        # Give the abandoned thread a moment; queued branches still must not
        # start even once a worker frees up.
        release.set()
        time.sleep(0.3)
        assert set(started) == {"b0", "b1"}, started
    finally:
        release.set()


def test_invoke_respects_max_concurrency() -> None:
    active = 0
    max_active = 0
    started: list[str] = []
    lock = threading.Lock()

    def make(name: str) -> Callable[[Any], str]:
        def fn(_: Any) -> str:
            nonlocal active, max_active
            with lock:
                active += 1
                started.append(name)
                max_active = max(max_active, active)
            time.sleep(0.1)
            with lock:
                active -= 1
            return name

        return fn

    q = RunnableQuorum(
        {f"b{i}": RunnableLambda(make(f"b{i}")) for i in range(5)},
        quorum__=2,
    )
    q.invoke(1, config=RunnableConfig(max_concurrency=2))
    assert max_active <= 2
    assert set(started) <= {"b0", "b1", "b2"}
    assert "b4" not in started


# --------------------------------------------------------------------------- #
# Async ainvoke
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_ainvoke_returns_named_results() -> None:
    def make(n: int) -> Callable[[int], Awaitable[int]]:
        async def add(x: int) -> int:
            return x + n

        return add

    q = RunnableQuorum(
        {"a": RunnableLambda(make(1)), "b": RunnableLambda(make(2))},
        quorum__=2,
    )
    assert await q.ainvoke(1) == {"a": 2, "b": 3}


@pytest.mark.asyncio
async def test_ainvoke_falsy_values_count_as_success() -> None:
    async def empty(_: Any) -> list[Any]:
        return []

    async def none(_: Any) -> None:
        return None

    q = RunnableQuorum(
        {"empty": RunnableLambda(empty), "none": RunnableLambda(none)},
        quorum__=2,
    )
    assert await q.ainvoke(1) == {"empty": [], "none": None}


@pytest.mark.asyncio
async def test_ainvoke_collects_in_completion_order() -> None:
    def make(name: str, delay: float) -> Callable[[Any], Awaitable[str]]:
        async def fn(_: Any) -> str:
            await asyncio.sleep(delay)
            return name

        return fn

    q = RunnableQuorum(
        {
            "slow": RunnableLambda(make("slow", 0.2)),
            "fast": RunnableLambda(make("fast", 0.02)),
            "middle": RunnableLambda(make("middle", 0.1)),
        },
        quorum__=2,
    )
    assert list(await q.ainvoke(1)) == ["fast", "middle"]


@pytest.mark.asyncio
async def test_ainvoke_ordinary_exception_does_not_abort() -> None:
    async def boom(_: Any) -> str:
        msg = "boom"
        raise ValueError(msg)

    q = RunnableQuorum(
        {"bad": RunnableLambda(boom), "good": RunnableLambda(lambda _: "ok")},
        quorum__=1,
    )
    assert await q.ainvoke(1) == {"good": "ok"}


@pytest.mark.asyncio
async def test_ainvoke_quorum_not_met_carries_results_and_errors() -> None:
    async def boom(_: Any) -> str:
        msg = "rt"
        raise RuntimeError(msg)

    q = RunnableQuorum(
        {"ok": RunnableLambda(lambda _: "result"), "bad": RunnableLambda(boom)},
        quorum__=2,
    )
    with pytest.raises(QuorumNotMetError) as exc_info:
        await q.ainvoke(1)

    err = exc_info.value
    assert err.results == {"ok": "result"}
    assert set(err.exceptions) == {"bad"}
    assert isinstance(err.exceptions["bad"], RuntimeError)


@pytest.mark.asyncio
async def test_ainvoke_each_branch_gets_the_same_input() -> None:
    seen: list[object] = []

    async def record(x: object) -> str:
        seen.append(x)
        return "ok"

    sentinel = object()
    await RunnableQuorum(
        {f"b{i}": RunnableLambda(record) for i in range(3)}, quorum__=3
    ).ainvoke(sentinel)
    assert seen == [sentinel, sentinel, sentinel]


@pytest.mark.asyncio
async def test_ainvoke_early_return_cancels_remaining_tasks() -> None:
    started: list[str] = []
    cancelled: list[str] = []

    def slow(name: str) -> Callable[[Any], Awaitable[str]]:
        async def fn(_: Any) -> str:
            started.append(name)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(name)
                raise
            return name

        return fn

    q = RunnableQuorum(
        {
            "fast1": RunnableLambda(lambda _: "f1"),
            "fast2": RunnableLambda(lambda _: "f2"),
            "slow1": RunnableLambda(slow("slow1")),
            "slow2": RunnableLambda(slow("slow2")),
        },
        quorum__=2,
    )
    start = time.monotonic()
    output = await q.ainvoke(1)
    elapsed = time.monotonic() - start

    assert output == {"fast1": "f1", "fast2": "f2"}
    assert elapsed < 1.0, elapsed
    assert set(started) == {"slow1", "slow2"}
    # Remaining tasks must be cancelled rather than abandoned pending.
    assert set(cancelled) == {"slow1", "slow2"}, cancelled
    pending = asyncio.all_tasks() - {asyncio.current_task()}
    assert not pending, pending


@pytest.mark.asyncio
async def test_ainvoke_respects_max_concurrency_and_never_starts_queued() -> None:
    active = 0
    max_active = 0
    started: list[str] = []

    def make(name: str) -> Callable[[Any], Awaitable[str]]:
        async def fn(_: Any) -> str:
            nonlocal active, max_active
            active += 1
            started.append(name)
            max_active = max(max_active, active)
            await asyncio.sleep(0.1)
            active -= 1
            return name

        return fn

    q = RunnableQuorum(
        {f"b{i}": RunnableLambda(make(f"b{i}")) for i in range(5)},
        quorum__=2,
    )
    output = await q.ainvoke(1, config=RunnableConfig(max_concurrency=2))
    assert len(output) == 2
    assert max_active <= 2
    assert set(started) <= {"b0", "b1", "b2"}
    assert {"b3", "b4"}.isdisjoint(started)


@pytest.mark.asyncio
async def test_ainvoke_failed_branch_frees_slot_for_queued_branch() -> None:
    started: list[str] = []

    def make(name: str, *, fail: bool = False) -> Callable[[Any], Awaitable[str]]:
        async def fn(_: Any) -> str:
            started.append(name)
            await asyncio.sleep(0.01)
            if fail:
                raise RuntimeError(name)
            return name

        return fn

    # Concurrency 1: b0 fails, b1 is launched next and meets the quorum;
    # b2 is never started.
    q = RunnableQuorum(
        {
            "b0": RunnableLambda(make("b0", fail=True)),
            "b1": RunnableLambda(make("b1")),
            "b2": RunnableLambda(make("b2")),
        },
        quorum__=1,
    )
    assert await q.ainvoke(1, config=RunnableConfig(max_concurrency=1)) == {"b1": "b1"}
    assert started == ["b0", "b1"]


@pytest.mark.asyncio
async def test_ainvoke_external_cancellation_propagates_and_cancels_children() -> None:
    cancelled: list[str] = []
    entered: list[str] = []

    def hang(name: str) -> Callable[[Any], Awaitable[str]]:
        async def fn(_: Any) -> str:
            entered.append(name)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(name)
                raise
            return name

        return fn

    q = RunnableQuorum(
        {"b0": RunnableLambda(hang("b0")), "b1": RunnableLambda(hang("b1"))},
        quorum__=2,
    )
    task = asyncio.create_task(q.ainvoke(1))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    assert set(cancelled) == {"b0", "b1"}, cancelled


@pytest.mark.asyncio
async def test_ainvoke_external_cancellation_with_queued_branches() -> None:
    entered: list[str] = []
    cancelled: list[str] = []

    def hang(name: str) -> Callable[[Any], Awaitable[str]]:
        async def fn(_: Any) -> str:
            entered.append(name)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(name)
                raise
            return name

        return fn

    # Only b0 can run; b1/b2 stay queued and must never be started.
    q = RunnableQuorum(
        {
            "b0": RunnableLambda(hang("b0")),
            "b1": RunnableLambda(hang("b1")),
            "b2": RunnableLambda(hang("b2")),
        },
        quorum__=2,
    )
    task = asyncio.create_task(q.ainvoke(1, config=RunnableConfig(max_concurrency=1)))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert entered == ["b0"], entered
    assert cancelled == ["b0"], cancelled


# --------------------------------------------------------------------------- #
# Callbacks / config propagation
# --------------------------------------------------------------------------- #


class _EventCollector(BaseCallbackHandler):
    def __init__(self) -> None:
        # (branch name tag or "", parent run id, sorted tags)
        self.starts: list[tuple[str, UUID | None, list[str]]] = []
        # (run id, parent run id)
        self.ends: list[tuple[UUID, UUID | None]] = []
        # (error type, parent run id)
        self.errors: list[tuple[str, UUID | None]] = []

    @override
    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        sorted_tags = sorted(tags or [])
        branch = next(
            (
                t.removeprefix("map:key:")
                for t in sorted_tags
                if t.startswith("map:key:")
            ),
            "",
        )
        self.starts.append((branch, parent_run_id, sorted_tags))

    @override
    def on_chain_end(
        self,
        outputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.ends.append((run_id, parent_run_id))

    @override
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.errors.append((type(error).__name__, parent_run_id))


def _failing_runnable() -> RunnableLambda[Any, str]:
    def boom(_: Any) -> str:
        msg = "boom"
        raise ValueError(msg)

    return RunnableLambda(boom)


def test_invoke_callbacks_link_children_to_root() -> None:
    collector = _EventCollector()
    q = RunnableQuorum(
        {
            "good1": RunnableLambda(lambda _: "a"),
            "bad": _failing_runnable(),
            "good2": RunnableLambda(lambda _: "b"),
        },
        quorum__=2,
    )
    config = RunnableConfig(callbacks=[collector], tags=["inherited-tag"])
    output = q.invoke(1, config=config)

    assert output == {"good1": "a", "good2": "b"}
    root_starts = [s for s in collector.starts if s[1] is None]
    assert len(root_starts) == 1
    # Root run ends successfully even though one branch failed.
    assert sum(1 for _, parent in collector.ends if parent is None) == 1
    assert not [e for e in collector.errors if e[1] is None]

    child_starts = [s for s in collector.starts if s[1] is not None]
    child_tags = {branch: tags for branch, _, tags in child_starts}
    assert set(child_tags) == {"good1", "bad", "good2"}
    for tags in child_tags.values():
        assert "inherited-tag" in tags
    assert any(e[0] == "ValueError" for e in collector.errors)


@pytest.mark.asyncio
async def test_ainvoke_callbacks_link_children_to_root() -> None:
    async def boom(_: Any) -> str:
        msg = "boom"
        raise ValueError(msg)

    collector = _EventCollector()
    q = RunnableQuorum(
        {
            "good1": RunnableLambda(lambda _: "a"),
            "bad": RunnableLambda(boom),
            "good2": RunnableLambda(lambda _: "b"),
        },
        quorum__=2,
    )
    config = RunnableConfig(callbacks=[collector], tags=["inherited-tag"])
    output = await q.ainvoke(1, config=config)

    assert output == {"good1": "a", "good2": "b"}
    root_starts = [s for s in collector.starts if s[1] is None]
    assert len(root_starts) == 1
    assert sum(1 for _, parent in collector.ends if parent is None) == 1
    child_starts = [s for s in collector.starts if s[1] is not None]
    child_tags = {branch: tags for branch, _, tags in child_starts}
    assert set(child_tags) == {"good1", "bad", "good2"}
    for tags in child_tags.values():
        assert "inherited-tag" in tags
    assert any(e[0] == "ValueError" for e in collector.errors)


# --------------------------------------------------------------------------- #
# Schemas, graph, serialization, composition
# --------------------------------------------------------------------------- #


def test_output_schema_fields_are_optional() -> None:
    q: RunnableQuorum[Any] = RunnableQuorum(
        {"a": RunnableLambda(lambda _: 1), "b": RunnableLambda(lambda _: "x")},
        quorum__=1,
    )
    schema = cast("type[BaseModel]", q.output_schema)
    assert set(schema.model_fields) == {"a", "b"}
    # Any subset of branches may be returned; no required fields.
    assert schema(a=None, b=None).model_dump() == {"a": None, "b": None}


def test_get_graph_connects_every_branch() -> None:
    q: RunnableQuorum[Any] = RunnableQuorum(
        {"a": RunnablePassthrough(), "b": RunnablePassthrough()}, quorum__=1
    )
    graph = q.get_graph()
    assert len(graph.nodes) >= 3


def test_serialization_roundtrip() -> None:
    q: RunnableQuorum[Any] = RunnableQuorum(
        {"a": RunnablePassthrough(), "b": RunnablePassthrough()}, quorum__=2
    )
    restored = loads(dumps(q))
    assert isinstance(restored, RunnableQuorum)
    assert restored.quorum__ == 2
    assert set(restored.steps__) == {"a", "b"}
    assert restored.invoke("v") == {"a": "v", "b": "v"}


def test_composes_in_a_sequence() -> None:
    chain: RunnableSerializable[Any, Any] = (
        RunnableLambda(lambda x: x + 1)
        | RunnableQuorum({"doubled": RunnableLambda(lambda x: x * 2)}, quorum__=1)
        | RunnableLambda(lambda d: d["doubled"] + 1)
    )
    assert chain.invoke(1) == 5


def test_runnable_parallel_behavior_unchanged() -> None:
    """A failing branch still fails the whole parallel run."""

    def boom(_: Any) -> str:
        msg = "boom"
        raise ZeroDivisionError(msg)

    parallel = RunnableParallel(
        good=RunnableLambda(lambda _: "ok"), bad=RunnableLambda(boom)
    )
    with pytest.raises(ZeroDivisionError):
        parallel.invoke(1)
