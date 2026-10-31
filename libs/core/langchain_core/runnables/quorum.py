"""`Runnable` that returns as soon as a quorum of named branches succeeds."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, wait
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict
from typing_extensions import override

from langchain_core.exceptions import LangChainException
from langchain_core.runnables.base import (
    Runnable,
    RunnableSerializable,
    _get_schema_field_definition,
    coerce_to_runnable,
)
from langchain_core.runnables.config import (
    ContextThreadPoolExecutor,
    RunnableConfig,
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    patch_config,
    set_config_context,
)
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    Input,
    coro_with_context,
    get_unique_config_specs,
    indent_lines_after_first,
)
from langchain_core.utils.pydantic import (
    TypeBaseModel,
    create_model_v2,
    get_fields,
)

if TYPE_CHECKING:
    from langchain_core.callbacks.manager import AsyncCallbackManagerForChainRun
    from langchain_core.runnables.graph import Graph


class QuorumNotMetError(LangChainException):
    """Raised when a `RunnableQuorum` ends without enough successful branches.

    Every branch has finished (either by returning or by raising) and fewer than
    the required number of branches succeeded.

    Attributes:
        quorum: The number of successful branches that was required.
        results: Mapping of successful branch names to their return values,
            ordered by completion.
        exceptions: Mapping of failed branch names to the exception they raised.
    """

    def __init__(
        self,
        *,
        quorum: int,
        results: Mapping[str, Any],
        exceptions: Mapping[str, BaseException],
    ) -> None:
        """Initialize the error.

        Args:
            quorum: The number of successful branches that was required.
            results: Mapping of successful branch names to their return values.
            exceptions: Mapping of failed branch names to the exception they
                raised.
        """
        self.quorum = quorum
        self.results = dict(results)
        self.exceptions = dict(exceptions)
        branch_errors = ", ".join(
            f"{name}: {type(exc).__name__}: {exc}"
            for name, exc in self.exceptions.items()
        )
        total = len(self.results) + len(self.exceptions)
        msg = (
            f"`RunnableQuorum` required {quorum} successful branch(es) but only "
            f"{len(self.results)} of {total} succeeded. Branch errors: "
            f"{branch_errors or '(none)'}"
        )
        super().__init__(msg)


class RunnableQuorum(RunnableSerializable[Input, dict[str, Any]]):
    """Runnable that returns as soon as a required number of branches succeeds.

    `RunnableQuorum` runs a mapping of named `Runnable`s concurrently, providing
    the same input to each, and collects their results in the order in which
    they complete successfully. As soon as `quorum` branches have succeeded it
    returns a mapping of their names to their outputs; it does not wait for the
    remaining branches.

    Unlike `RunnableParallel`, which waits for every branch and fails if any
    branch fails, `RunnableQuorum` treats ordinary branch exceptions as
    non-fatal. A branch counts as successful whenever it returns without
    raising, even if the return value is falsy (e.g. `None` or an empty list).
    If every branch has finished and fewer than `quorum` succeeded, a
    `QuorumNotMetError` carrying the successful results and the branch
    exceptions is raised.

    The `max_concurrency` key of the config is respected: no more than that
    many branches run at the same time. After the quorum is reached, queued
    branches are never started, pending asynchronous branches are cancelled,
    and in-flight synchronous branches are abandoned rather than awaited.

    Example:
        ```python
        from langchain_core.runnables import RunnableLambda, RunnableQuorum

        quorum = RunnableQuorum(
            {
                "fast": RunnableLambda(lambda x: x + 1),
                "flaky": RunnableLambda(lambda x: 1 / 0),
                "slow": RunnableLambda(lambda x: x + 3),
            },
            quorum__=2,
        )
        # Returns as soon as two of the three branches have succeeded:
        quorum.invoke(1)
        await quorum.ainvoke(1)
        ```
    """

    steps__: Mapping[str, Runnable[Input, Any]]
    """The named branches run by the quorum."""

    quorum__: int
    """The number of branches that must succeed for the quorum to return."""

    def __init__(
        self,
        steps__: Mapping[
            str,
            Runnable[Input, Any]
            | Callable[[Input], Any]
            | Mapping[str, Runnable[Input, Any] | Callable[[Input], Any]],
        ]
        | None = None,
        quorum__: int = 1,
        **kwargs: (
            Runnable[Input, Any]
            | Callable[[Input], Any]
            | Mapping[str, Runnable[Input, Any] | Callable[[Input], Any]]
        ),
    ) -> None:
        """Create a `RunnableQuorum`.

        Args:
            steps__: The named branches to include. Each branch receives the
                same input.
            quorum__: Number of branches that must succeed before the quorum
                returns. Must be at least `1` and no larger than the number of
                branches.
            **kwargs: Additional named branches to include.

        Raises:
            TypeError: If `quorum__` is not an integer.
            ValueError: If no branches are provided or `quorum__` is not a
                positive integer no larger than the number of branches.
        """
        merged = {**steps__} if steps__ is not None else {}
        merged.update(kwargs)
        if not merged:
            msg = "`RunnableQuorum` requires at least one branch."
            raise ValueError(msg)
        if not isinstance(quorum__, int) or isinstance(quorum__, bool):
            msg = (
                "`quorum__` must be a positive integer no larger than the "
                f"number of branches ({len(merged)}); got {quorum__!r}."
            )
            raise TypeError(msg)
        if quorum__ < 1 or quorum__ > len(merged):
            msg = (
                f"`quorum__` must be between 1 and the number of branches "
                f"({len(merged)}); got {quorum__}."
            )
            raise ValueError(msg)
        super().__init__(
            steps__={key: coerce_to_runnable(r) for key, r in merged.items()},
            quorum__=quorum__,
        )

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        """Return `True` as this class is serializable."""
        return True

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @override
    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str:
        """Get the name of the `Runnable`.

        Args:
            suffix: The suffix to use.
            name: The name to use.

        Returns:
            The name of the `Runnable`.
        """
        name = name or self.name or f"RunnableQuorum<{','.join(self.steps__.keys())}>"
        return super().get_name(suffix, name=name)

    @property
    @override
    def InputType(self) -> Any:
        """The type of the input to the `Runnable`."""
        for step in self.steps__.values():
            if step.InputType:
                return step.InputType

        return Any

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> TypeBaseModel:
        """Get the input schema of the `Runnable`.

        Args:
            config: The config to use.

        Returns:
            The input schema of the `Runnable`.
        """
        if all(
            s.get_input_jsonschema(config).get("type", "object") == "object"
            for s in self.steps__.values()
        ):
            for step in self.steps__.values():
                step_input_schema = step.get_input_schema(config)
                fields = get_fields(step_input_schema)
                root_field = fields.get("root")
                if root_field is not None and root_field.annotation != Any:
                    return super().get_input_schema(config)
                root_field = fields.get("__root__")
                if root_field is not None and root_field.annotation != Any:
                    return step_input_schema

            # This is correct, but pydantic typings/mypy don't think so.
            return create_model_v2(
                self.get_name("Input"),
                field_definitions={
                    k: _get_schema_field_definition(v)
                    for step in self.steps__.values()
                    for k, v in get_fields(step.get_input_schema(config)).items()
                    if k != "__root__"
                },
            )

        return super().get_input_schema(config)

    @override
    def get_output_schema(
        self, config: RunnableConfig | None = None
    ) -> type[BaseModel]:
        """Get the output schema of the `Runnable`.

        Only a subset of the branches may appear in the output, so every field
        is optional.

        Args:
            config: The config to use.

        Returns:
            The output schema of the `Runnable`.
        """
        fields = {k: (v.OutputType, None) for k, v in self.steps__.items()}
        # The return type is narrowed to `type[BaseModel]` (rather than the base
        # class's `TypeBaseModel`) because this override always builds the schema
        # with `create_model_v2`, so it is guaranteed to be a Pydantic v2 model.
        return create_model_v2(self.get_name("Output"), field_definitions=fields)

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        """Get the config specs of the `Runnable`.

        Returns:
            The config specs of the `Runnable`.
        """
        return get_unique_config_specs(
            spec for step in self.steps__.values() for spec in step.config_specs
        )

    @override
    def get_graph(self, config: RunnableConfig | None = None) -> Graph:
        """Get the graph representation of the `Runnable`.

        Args:
            config: The config to use.

        Returns:
            The graph representation of the `Runnable`.

        Raises:
            ValueError: If a `Runnable` has no first or last node.
        """
        # Import locally to prevent circular import
        from langchain_core.runnables.graph import Graph  # noqa: PLC0415

        graph = Graph()
        input_node = graph.add_node(self.get_input_schema(config))
        output_node = graph.add_node(self.get_output_schema(config))
        for step in self.steps__.values():
            step_graph = step.get_graph()
            step_graph.trim_first_node()
            step_graph.trim_last_node()
            if not step_graph:
                graph.add_edge(input_node, output_node)
            else:
                step_first_node, step_last_node = graph.extend(step_graph)
                if not step_first_node:
                    msg = f"Runnable {step} has no first node"
                    raise ValueError(msg)
                if not step_last_node:
                    msg = f"Runnable {step} has no last node"
                    raise ValueError(msg)
                graph.add_edge(input_node, step_first_node)
                graph.add_edge(step_last_node, output_node)

        return graph

    @override
    def __repr__(self) -> str:
        map_for_repr = ",\n  ".join(
            f"{k}: {indent_lines_after_first(repr(v), '  ' + k + ': ')}"
            for k, v in self.steps__.items()
        )
        return (
            f"RunnableQuorum(quorum__={self.quorum__}, "
            "steps__={\n  " + map_for_repr + "\n})"
        )

    @override
    def invoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        # setup callbacks
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        # start the root run
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        def _invoke_step(
            step: Runnable[Input, Any], input_: Input, config: RunnableConfig, key: str
        ) -> Any:
            child_config = patch_config(
                config,
                # mark each step as a child run
                callbacks=run_manager.get_child(f"map:key:{key}"),
            )
            with set_config_context(child_config) as context:
                return context.run(
                    step.invoke,
                    input_,
                    child_config,
                )

        # gather results until the quorum is reached (or every branch finished)
        successful: dict[str, Any] = {}
        branch_errors: dict[str, BaseException] = {}
        fatal_error: BaseException | None = None
        try:
            # copy to avoid issues from the caller mutating the steps during invoke()
            steps = dict(self.steps__)
            names = list(steps)
            max_concurrency = config.get("max_concurrency")
            # Branches are launched lazily, at most `in_flight_limit` at a time;
            # this also bounds the thread pool so no work queues up ahead of the
            # in-flight branches.
            in_flight_limit = (
                max_concurrency
                if isinstance(max_concurrency, int)
                and not isinstance(max_concurrency, bool)
                and max_concurrency > 0
                else len(names)
            )

            # The executor is managed manually (instead of via
            # `get_executor_for_config`) so that, once the quorum is reached,
            # queued branches are never submitted and in-flight branches are
            # not awaited.
            executor = ContextThreadPoolExecutor(max_workers=in_flight_limit)
            try:
                future_to_name: dict[Future[Any], str] = {}
                next_index = 0

                def _submit_next() -> Future[Any]:
                    nonlocal next_index
                    name = names[next_index]
                    next_index += 1
                    future = executor.submit(
                        _invoke_step, steps[name], input, config, name
                    )
                    future_to_name[future] = name
                    return future

                pending: set[Future[Any]] = {
                    _submit_next() for _ in range(min(in_flight_limit, len(names)))
                }

                while pending and len(successful) < self.quorum__:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        name = future_to_name[future]
                        if future.cancelled():
                            continue
                        exc = future.exception()
                        if exc is None:
                            # Any non-raising return value counts as success,
                            # including falsy values such as `[]` or `None`.
                            successful[name] = future.result()
                        elif isinstance(exc, Exception):
                            # Ordinary branch errors do not abort the quorum.
                            branch_errors[name] = exc
                        else:
                            # Non-Exception `BaseException`s (e.g.
                            # `KeyboardInterrupt`) surface immediately.
                            fatal_error = exc
                            break

                    if fatal_error is not None:
                        break

                    # Replace finished branches with queued ones only while the
                    # quorum is still unmet; once it is met the remaining
                    # branches are never submitted.
                    while (
                        len(successful) < self.quorum__
                        and next_index < len(names)
                        and len(pending) < in_flight_limit
                    ):
                        pending.add(_submit_next())

                if fatal_error is None and len(successful) < self.quorum__:
                    fatal_error = QuorumNotMetError(
                        quorum=self.quorum__,
                        results=successful,
                        exceptions=branch_errors,
                    )
            finally:
                # Do not start queued branches and do not wait for branches that
                # are already running when returning (or unwinding an error).
                executor.shutdown(wait=False, cancel_futures=True)
        # finish the root run
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise

        if fatal_error is not None:
            run_manager.on_chain_error(fatal_error)
            raise fatal_error
        run_manager.on_chain_end(successful)
        return successful

    @override
    async def ainvoke(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> dict[str, Any]:
        # setup callbacks
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        # start the root run
        run_manager: AsyncCallbackManagerForChainRun = (
            await callback_manager.on_chain_start(
                None,
                input,
                name=config.get("run_name") or self.get_name(),
                run_id=config.pop("run_id", None),
            )
        )

        # copy to avoid issues from the caller mutating the steps during ainvoke()
        steps = dict(self.steps__)
        names = list(steps)
        max_concurrency = config.get("max_concurrency")
        # Branches are launched lazily, at most `in_flight_limit` at a time.
        # Queued branches have no task until a slot frees up, so once the quorum
        # is reached they are simply never started.
        in_flight_limit = (
            max_concurrency
            if isinstance(max_concurrency, int)
            and not isinstance(max_concurrency, bool)
            and max_concurrency > 0
            else len(names)
        )

        async def _cancel_remaining(tasks_: set[asyncio.Task[Any]]) -> None:
            for task in tasks_:
                task.cancel()
            if tasks_:
                # Let cancelled tasks run their cleanup; discard their
                # `CancelledError`/branch results.
                await asyncio.gather(*tasks_, return_exceptions=True)

        successful: dict[str, Any] = {}
        branch_errors: dict[str, BaseException] = {}
        fatal_error: BaseException | None = None
        try:
            task_to_name: dict[asyncio.Task[Any], str] = {}
            next_index = 0

            def _launch_next() -> asyncio.Task[Any]:
                nonlocal next_index
                name = names[next_index]
                next_index += 1
                step = steps[name]
                child_config = patch_config(
                    config,
                    # mark each step as a child run
                    callbacks=run_manager.get_child(f"map:key:{name}"),
                )
                with set_config_context(child_config) as context:
                    task = cast(
                        "asyncio.Task[Any]",
                        coro_with_context(
                            step.ainvoke(input, child_config),
                            context,
                            create_task=True,
                        ),
                    )
                task_to_name[task] = name
                return task

            pending: set[asyncio.Task[Any]] = {
                _launch_next() for _ in range(min(in_flight_limit, len(names)))
            }

            while pending and len(successful) < self.quorum__:
                try:
                    done, pending = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED
                    )
                except asyncio.CancelledError:
                    # External cancellation: clean up child tasks and propagate.
                    await _cancel_remaining(pending)
                    raise

                for task in done:
                    name = task_to_name[task]
                    if task.cancelled():
                        # A sibling was cancelled without this coroutine being
                        # cancelled; propagate cancellation to the caller.
                        fatal_error = asyncio.CancelledError()
                        break
                    exc = task.exception()
                    if exc is None:
                        # Any non-raising return value counts as success,
                        # including falsy values such as `[]` or `None`.
                        successful[name] = task.result()
                    elif isinstance(exc, Exception):
                        # Ordinary branch errors do not abort the quorum.
                        branch_errors[name] = exc
                    else:
                        # `CancelledError` and other non-Exception
                        # `BaseException`s surface immediately.
                        fatal_error = exc
                        break

                if fatal_error is not None:
                    break

                # Launch queued branches into freed slots only while the quorum
                # is still unmet; once it is met the remaining branches are
                # never started.
                while (
                    len(successful) < self.quorum__
                    and next_index < len(names)
                    and len(pending) < in_flight_limit
                ):
                    pending.add(_launch_next())

            if fatal_error is None and len(successful) < self.quorum__:
                fatal_error = QuorumNotMetError(
                    quorum=self.quorum__,
                    results=successful,
                    exceptions=branch_errors,
                )
            else:
                # Quorum reached (or fatal error): remaining in-flight tasks
                # are cancelled; branches beyond `next_index` never started.
                await _cancel_remaining(pending)
        # finish the root run
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise

        if fatal_error is not None:
            await run_manager.on_chain_error(fatal_error)
            raise fatal_error
        await run_manager.on_chain_end(successful)
        return successful
