"""Requirement-aware orchestration primitives for Fitzy Phase 2.

Concrete HTTP/tool implementations remain outside this module. The coordinator
is the safety boundary between planning and tool execution: it checks required
parameters, waits when customer input is required, and runs independent ready
operations concurrently.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("fitzy.execution")

from .contracts import RequirementCheckResult, TOOL_CONTRACTS, ToolName
from .requirements import ToolRequirementChecker
from .state import ActionStatus, ConversationState, PlannedAction

ToolExecutor = Callable[[ToolName, Mapping[str, Any]], Awaitable[Any]]


@dataclass(frozen=True)
class ActionExecutionResult:
    """Outcome of one coordinator pass over the current action graph."""

    executed_action_ids: tuple[str, ...]
    waiting_action_id: str | None
    missing_parameters: tuple[str, ...]
    completed_action_ids: tuple[str, ...]
    failed_action_ids: tuple[str, ...]


class ActionExecutionCoordinator:
    """Validate and execute currently eligible actions without business logic."""

    def __init__(self, requirement_checker: ToolRequirementChecker | None = None) -> None:
        self._requirements = requirement_checker or ToolRequirementChecker()

    async def run_ready_actions(
        self,
        state: ConversationState,
        *,
        tool_executor: ToolExecutor,
        state_values: Mapping[str, Any] | None = None,
        derived_values: Mapping[str, Any] | None = None,
    ) -> ActionExecutionResult:
        """Execute all actions that are dependency-ready and contract-ready.

        Missing requirements are converted to ``WAITING_FOR_INPUT`` rather than
        causing an API call. Unrelated ready actions can still run concurrently.
        """

        ready_actions = state.action_plan.ready_actions()
        if not ready_actions:
            return ActionExecutionResult((), None, (), (), ())

        runnable: list[tuple[PlannedAction, RequirementCheckResult]] = []
        first_waiting: PlannedAction | None = None
        waiting_requirements: tuple[str, ...] = ()

        for action in ready_actions:
            contract = TOOL_CONTRACTS[action.tool_name]
            check = self._requirements.check(
                contract,
                provided_parameters=action.parameters,
                state_values=state_values,
                derived_values=derived_values,
            )
            if not check.ready:
                action.status = ActionStatus.WAITING_FOR_INPUT
                action.missing_parameters = [*check.missing_parameters, *check.invalid_parameters]
                if first_waiting is None:
                    first_waiting = action
                    waiting_requirements = tuple(action.missing_parameters)
                continue

            action.status = ActionStatus.READY
            action.parameters = check.resolved_parameters
            if action.tool_name in {ToolName.ADD_TO_CART, ToolName.UPDATE_CART, ToolName.REMOVE_FROM_CART, ToolName.CLEAR_CART, ToolName.PLACE_ORDER}:
                action.parameters.setdefault("idempotency_key", action.action_id)
                if action.tool_name == ToolName.PLACE_ORDER:
                    action.parameters.setdefault("checkout_request_id", action.action_id)
            runnable.append((action, check))

        async def execute(item: tuple[PlannedAction, RequirementCheckResult]) -> tuple[PlannedAction, Any, Exception | None]:
            action = item[0]
            action.status = ActionStatus.RUNNING
            try:
                result = await tool_executor(action.tool_name, action.parameters)
            except Exception as exc:  # noqa: BLE001 - action state carries failure
                logger.exception("action_execution_failed tool=%s error=%s", action.tool_name.value, exc)
                action.status = ActionStatus.FAILED
                return action, None, exc
            action.status = ActionStatus.COMPLETED
            return action, result, None

        outcomes = await asyncio.gather(*(execute(item) for item in runnable)) if runnable else []
        completed: list[str] = []
        failed: list[str] = []
        executed: list[str] = []

        for action, result, error in outcomes:
            executed.append(action.action_id)
            if error is None:
                completed.append(action.action_id)
                state.record_tool_result(action.tool_name, result)
            else:
                failed.append(action.action_id)

        state.pending_action_id = first_waiting.action_id if first_waiting else None
        return ActionExecutionResult(
            executed_action_ids=tuple(executed),
            waiting_action_id=first_waiting.action_id if first_waiting else None,
            missing_parameters=waiting_requirements,
            completed_action_ids=tuple(completed),
            failed_action_ids=tuple(failed),
        )
