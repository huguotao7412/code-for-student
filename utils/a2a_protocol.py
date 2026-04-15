from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, MutableMapping, TypedDict


class A2AMessage(TypedDict, total=False):
    version: str
    sender: str
    receiver: str
    kind: str
    payload: Dict[str, Any]
    meta: Dict[str, Any]


class A2AChannels:
    TASK_CONTEXT = "task_context"
    PLAN = "plan"
    ACTION_PROPOSAL = "action_proposal"
    REVIEW = "review"
    FINAL_OUTPUT = "final_output"


A2A_VERSION = "1.0"


def ensure_mailbox(state: Any) -> Dict[str, A2AMessage]:
    if not isinstance(state, dict):
        return {}
    mailbox = state.get("mailbox")
    if isinstance(mailbox, dict):
        return mailbox
    return {}


def read_payload(state: Any, channel: str, default: Any = None) -> Any:
    mailbox = ensure_mailbox(state)
    packet = mailbox.get(channel)
    if not isinstance(packet, dict):
        return default
    payload = packet.get("payload")
    if payload is None:
        return default
    return deepcopy(payload)


def write_packet(
    mailbox: MutableMapping[str, A2AMessage],
    *,
    channel: str,
    sender: str,
    receiver: str,
    kind: str,
    payload: Dict[str, Any],
    meta: Dict[str, Any] | None = None,
) -> MutableMapping[str, A2AMessage]:
    mailbox[channel] = {
        "version": A2A_VERSION,
        "sender": sender,
        "receiver": receiver,
        "kind": kind,
        "payload": deepcopy(payload),
        "meta": deepcopy(meta or {}),
    }
    return mailbox
