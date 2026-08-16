"""Graph construction and static validation."""

import pytest

from agentflow import Step, Workflow, WorkflowError, agent, ref


@agent(name="noop")
def noop():
    """Does nothing."""
    return None


def build(steps):
    return Workflow(name="test", steps=steps)


def test_dependencies_are_inferred_from_refs():
    workflow = build(
        [
            Step(name="a", agent=noop),
            Step(name="b", agent=noop, inputs={"x": ref("a.value")}),
        ]
    )
    assert workflow.graph() == {"a": set(), "b": {"a"}}


def test_explicit_depends_on_is_merged_with_refs():
    workflow = build(
        [
            Step(name="a", agent=noop),
            Step(name="b", agent=noop),
            Step(name="c", agent=noop, depends_on=["a"], inputs={"x": ref("b")}),
        ]
    )
    assert workflow.graph()["c"] == {"a", "b"}


def test_defaulted_ref_is_a_soft_dependency():
    workflow = build(
        [
            Step(name="maybe", agent=noop),
            Step(name="always", agent=noop, inputs={"x": ref("maybe", None)}),
        ]
    )
    assert workflow.graph()["always"] == {"maybe"}
    assert workflow.hard_graph()["always"] == set()


def test_step_referenced_both_ways_is_hard():
    workflow = build(
        [
            Step(name="a", agent=noop),
            Step(name="b", agent=noop, inputs={"x": ref("a", None), "y": ref("a.field")}),
        ]
    )
    assert workflow.hard_graph()["b"] == {"a"}


def test_foreach_ref_creates_a_dependency():
    workflow = build(
        [
            Step(name="a", agent=noop),
            Step(name="b", agent=noop, foreach=ref("a.items")),
        ]
    )
    assert workflow.graph()["b"] == {"a"}


def test_cycles_are_rejected():
    with pytest.raises(WorkflowError, match="cycle"):
        build(
            [
                Step(name="a", agent=noop, inputs={"x": ref("b")}),
                Step(name="b", agent=noop, inputs={"x": ref("a")}),
            ]
        )


def test_self_reference_is_not_a_cycle_with_itself():
    workflow = build([Step(name="a", agent=noop, inputs={"x": ref("a", None)})])
    assert workflow.graph()["a"] == set()


def test_unknown_explicit_dependency_is_rejected():
    with pytest.raises(WorkflowError, match="unknown step"):
        build([Step(name="a", agent=noop, depends_on=["ghost"])])


def test_duplicate_step_names_are_rejected():
    with pytest.raises(WorkflowError, match="duplicate"):
        build([Step(name="a", agent=noop), Step(name="a", agent=noop)])


def test_empty_workflow_is_rejected():
    with pytest.raises(WorkflowError, match="no steps"):
        build([])


def test_layers_group_independent_steps():
    workflow = build(
        [
            Step(name="root", agent=noop),
            Step(name="left", agent=noop, inputs={"x": ref("root")}),
            Step(name="right", agent=noop, inputs={"x": ref("root")}),
            Step(name="join", agent=noop, inputs={"a": ref("left"), "b": ref("right")}),
        ]
    )
    assert workflow.layers() == [["root"], ["left", "right"], ["join"]]


def test_describe_mentions_step_features():
    workflow = build(
        [
            Step(name="a", agent=noop),
            Step(name="b", agent=noop, foreach=ref("a.items"), retries=2, optional=True),
        ]
    )
    described = workflow.describe()
    assert "foreach a.items" in described
    assert "retries=2" in described
    assert "optional" in described
