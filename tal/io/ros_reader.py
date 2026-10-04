from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from .adapter_cleanup import suppress_cleanup_during_active_error
from .options import RosIngestOptions
from .ros_metadata import normalize_ros_text
from .ros_payload import require_message_family


@dataclass(frozen=True)
class RosMessage:
    topic: str
    msgtype: str
    family: str
    msg: object
    receive_ns: object | None


@dataclass(frozen=True)
class _TopicConnection:
    connection: object
    topic: str


@dataclass(frozen=True)
class _RosConnectionContext:
    connection: object
    topic: str
    msgtype: str
    family: str


def _connection_text(
    connection: object,
    *,
    field: str,
    owner: str,
    path: str,
) -> str:
    try:
        value = getattr(connection, field)
    except Exception as exc:
        raise ValueError(
            f"{owner}: malformed ROS connection {field} in {path!r}."
        ) from exc
    return normalize_ros_text(
        value,
        field=f"connection {field}",
        owner=owner,
        path=path,
    )


def _connection_has_messages(connection: object, *, owner: str, path: str) -> bool:
    try:
        return int(getattr(connection, "msgcount", 1)) > 0
    except Exception as exc:
        raise ValueError(
            f"{owner}: malformed ROS connection msgcount in {path!r}."
        ) from exc


def _select_topic(
    connections: Sequence[object],
    *,
    opts: RosIngestOptions,
    owner: str,
    path: str,
) -> tuple[str, tuple[_TopicConnection, ...]]:
    resolved = tuple(
        _TopicConnection(
            conn,
            _connection_text(conn, field="topic", owner=owner, path=path),
        )
        for conn in connections
    )
    topics = sorted({item.topic for item in resolved})
    if opts.topic is not None:
        selected = tuple(item for item in resolved if item.topic == opts.topic)
        if not selected:
            raise ValueError(f"{owner}: topic {opts.topic!r} was not found in {path!r}.")
        return opts.topic, selected
    if len(topics) != 1:
        raise ValueError(
            f"{owner}: topic ambiguity in {path!r}; candidates={tuple(topics)!r}. "
            "Provide topic explicitly."
        )
    topic = topics[0]
    return topic, tuple(item for item in resolved if item.topic == topic)


def _select_msgtype(
    connections: Sequence[_TopicConnection],
    *,
    opts: RosIngestOptions,
    owner: str,
    path: str,
) -> tuple[str, tuple[_RosConnectionContext, ...]]:
    resolved = tuple(
        (
            item,
            _connection_text(
                item.connection,
                field="msgtype",
                owner=owner,
                path=path,
            ),
        )
        for item in connections
    )
    msgtypes = sorted({msgtype for _item, msgtype in resolved})
    if opts.message_type is not None:
        if opts.message_type not in msgtypes:
            raise ValueError(
                f"{owner}: message_type {opts.message_type!r} was not found in {path!r}; "
                f"available={tuple(msgtypes)!r}."
            )
        msgtype = opts.message_type
    elif len(msgtypes) != 1:
        raise ValueError(
            f"{owner}: message-type ambiguity in {path!r}; candidates={tuple(msgtypes)!r}. "
            "Provide message_type explicitly."
        )
    else:
        msgtype = msgtypes[0]
    family = require_message_family(msgtype, owner=owner)
    return msgtype, tuple(
        _RosConnectionContext(item.connection, item.topic, value, family)
        for item, value in resolved
        if value == msgtype
    )


def _select_ros_connections(
    connections: Sequence[object],
    *,
    opts: RosIngestOptions,
    owner: str,
    path: str,
) -> tuple[_RosConnectionContext, ...]:
    active = tuple(
        conn
        for conn in connections
        if _connection_has_messages(conn, owner=owner, path=path)
    )
    if not active:
        raise ValueError(f"{owner}: no message-bearing ROS connections were found in {path!r}.")
    _topic, topic_connections = _select_topic(active, opts=opts, owner=owner, path=path)
    _msgtype, selected = _select_msgtype(
        topic_connections,
        opts=opts,
        owner=owner,
        path=path,
    )
    return selected


def _deserialize_ros_payload(
    reader: object,
    rawdata: object,
    *,
    context: _RosConnectionContext,
    owner: str,
    path: str,
) -> object:
    try:
        return reader.deserialize(rawdata, context.msgtype)  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - backend decode failure envelope.
        raise ValueError(
            f"{owner}: failed deserializing ROS message type {context.msgtype!r} "
            f"on topic {context.topic!r}."
        ) from exc


def _enter_ros_reader(any_reader: type, path: str, *, owner: str) -> tuple[ExitStack, object]:
    stack = ExitStack()
    try:
        reader = stack.enter_context(any_reader([Path(path)]))
    except Exception as exc:  # pragma: no cover - backend open failure envelope.
        suppress_cleanup_during_active_error(stack.close)
        raise ValueError(f"{owner}: failed opening ROS recording {path!r}.") from exc
    return stack, reader


def _iter_selected_ros_messages(
    reader: object,
    connections: Sequence[object],
    *,
    owner: str,
    path: str,
) -> Iterable[tuple[object, object, object]]:
    try:
        yield from reader.messages(connections=connections)  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - backend iteration failure envelope.
        raise ValueError(f"{owner}: failed reading ROS messages from {path!r}.") from exc


def _close_ros_reader(stack: ExitStack, *, owner: str, path: str) -> None:
    try:
        stack.close()
    except Exception as exc:  # pragma: no cover - backend close failure envelope.
        raise ValueError(f"{owner}: failed closing ROS recording {path!r}.") from exc


def _close_message_iterator(iterator: Iterator[RosMessage]) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


@contextmanager
def owned_ros_message_stream(
    messages: Iterable[RosMessage],
) -> Iterator[Iterator[RosMessage]]:
    """Close a reader-backed message stream at its consumer boundary."""
    iterator = iter(messages)
    try:
        yield iterator
    except BaseException:
        suppress_cleanup_during_active_error(partial(_close_message_iterator, iterator))
        raise
    else:
        _close_message_iterator(iterator)


def _read_ros_connections(reader: object, *, owner: str, path: str) -> tuple[object, ...]:
    try:
        return tuple(reader.connections)  # type: ignore[attr-defined]
    except Exception as exc:
        raise ValueError(f"{owner}: failed reading ROS connections from {path!r}.") from exc


def _decode_ros_message(
    reader: object,
    raw_message: object,
    *,
    contexts: dict[int, _RosConnectionContext],
    owner: str,
    path: str,
) -> RosMessage:
    try:
        connection, timestamp_ns, rawdata = raw_message  # type: ignore[misc]
    except Exception as exc:
        raise ValueError(f"{owner}: malformed ROS message record in {path!r}.") from exc
    context = contexts.get(id(connection))
    if context is None or context.connection is not connection:
        raise ValueError(f"{owner}: ROS backend yielded an unselected connection in {path!r}.")
    return RosMessage(
        topic=context.topic,
        msgtype=context.msgtype,
        family=context.family,
        msg=_deserialize_ros_payload(
            reader,
            rawdata,
            context=context,
            owner=owner,
            path=path,
        ),
        receive_ns=timestamp_ns,
    )


def iter_ros_messages(
    path: str,
    *,
    opts: RosIngestOptions,
    owner: str,
    reader_loader: Callable[..., type],
) -> Iterable[RosMessage]:
    any_reader = reader_loader(owner=owner)
    stack, reader = _enter_ros_reader(any_reader, path, owner=owner)
    try:
        connections = _select_ros_connections(
            _read_ros_connections(reader, owner=owner, path=path),
            opts=opts,
            owner=owner,
            path=path,
        )
        backend_connections = tuple(context.connection for context in connections)
        contexts = {id(context.connection): context for context in connections}
        raw_messages = _iter_selected_ros_messages(
            reader,
            backend_connections,
            owner=owner,
            path=path,
        )
        for raw_message in raw_messages:
            yield _decode_ros_message(
                reader,
                raw_message,
                contexts=contexts,
                owner=owner,
                path=path,
            )
    except BaseException:
        suppress_cleanup_during_active_error(
            partial(_close_ros_reader, stack, owner=owner, path=path)
        )
        raise
    else:
        _close_ros_reader(stack, owner=owner, path=path)


__all__ = ["RosMessage", "iter_ros_messages", "owned_ros_message_stream"]
