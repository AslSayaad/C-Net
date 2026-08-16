"""Blackboard state and reference resolution."""

import pytest

from agentflow.blackboard import Blackboard, RefError, ref, resolve


def test_ref_walks_dicts_and_lists():
    state = {"analysis": {"phases": [{"name": "discovery"}, {"name": "build"}]}}
    assert ref("analysis.phases.1.name").resolve(state) == "build"


def test_ref_without_default_raises_on_missing_path():
    with pytest.raises(RefError):
        ref("analysis.missing").resolve({"analysis": {}})


def test_ref_with_default_returns_it():
    assert ref("nothing.here", "fallback").resolve({}) == "fallback"
    assert ref("nothing.here", None).resolve({}) is None


def test_ref_default_none_is_distinguishable_from_no_default():
    assert ref("x", None).has_default is True
    assert ref("x").has_default is False


def test_resolve_walks_nested_structures():
    state = {"a": 1, "b": [2, 3]}
    shape = {"one": ref("a"), "many": [ref("b.0"), {"deep": ref("b.1")}], "literal": "x"}
    assert resolve(shape, state) == {"one": 1, "many": [2, {"deep": 3}], "literal": "x"}


async def test_set_and_snapshot_are_isolated():
    board = Blackboard({"input": {"k": "v"}})
    await board.set("step", {"list": [1]})
    snapshot = board.snapshot()
    snapshot["step"]["list"].append(2)
    assert board.get("step") == {"list": [1]}


async def test_require_raises_for_unknown_key():
    board = Blackboard()
    with pytest.raises(RefError):
        board.require("nope")


async def test_write_log_records_order():
    board = Blackboard()
    await board.set("a", 1)
    await board.update({"b": 2, "c": 3})
    assert board.write_log[0] == "a"
    assert set(board.write_log) == {"a", "b", "c"}
