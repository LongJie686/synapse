"""Human-in-the-loop node - pauses execution for human approval."""

from __future__ import annotations

from synapse_core.graph.state import RunState


async def human_approval_node(state: RunState, *, approval_message: str = "") -> dict:
    """
    Graph node that signals a human approval step is needed.
    In a real system, this would integrate with the frontend to show
    an approval dialog and wait for the human's response.

    For now, it marks the state as needing approval.
    """
    return {
        "context": {
            **state.context,
            "human_approval_required": True,
            "approval_message": approval_message or "Please review and approve before continuing.",
        },
        "metadata": {**state.metadata, "paused_for_human": True},
    }


async def process_human_input_node(state: RunState, *, approved: bool, feedback: str = "") -> dict:
    """
    Process the human's approval decision and continue execution.
    """
    return {
        "context": {
            **state.context,
            "human_approval_required": False,
            "human_approved": approved,
            "human_feedback": feedback,
        },
        "metadata": {**state.metadata, "paused_for_human": False},
    }
