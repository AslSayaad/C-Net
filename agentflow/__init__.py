"""agentflow — a multi-agent workflow system.

A workflow is a DAG of steps; each step runs an agent. Agents come in two
kinds and mix freely in one graph:

  * deterministic agents — plain Python functions (validation, arithmetic,
    I/O, existing model code), wrapped with `@agent`;
  * LLM agents — a Claude call with a role prompt and, optionally, a JSON
    Schema constraining its output.

The engine handles ordering, parallelism, retries, timeouts, conditional
branches, fan-out, and skip propagation, and reports token usage per step.

    from agentflow import Step, Workflow, agent, ref, run_workflow

See `agentflow/workflows/ai_ml_ops.py` for a complete example.
"""

from .agent import Agent, AgentContext, FunctionAgent, agent
from .blackboard import Blackboard, Ref, ref
from .engine import Engine, run_workflow
from .events import ConsoleReporter, Event, EventBus
from .llm import ClaudeClient, LLMError, LLMResponse, RefusalError, default_client
from .llm_agent import LLMAgent, array_of, schema
from .registry import AgentRegistry
from .types import RunResult, Status, StepResult, Usage
from .workflow import Step, Workflow, WorkflowError

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentContext",
    "AgentRegistry",
    "Blackboard",
    "ClaudeClient",
    "ConsoleReporter",
    "Engine",
    "Event",
    "EventBus",
    "FunctionAgent",
    "LLMAgent",
    "LLMError",
    "LLMResponse",
    "Ref",
    "RefusalError",
    "RunResult",
    "Status",
    "Step",
    "StepResult",
    "Usage",
    "Workflow",
    "WorkflowError",
    "agent",
    "array_of",
    "default_client",
    "ref",
    "run_workflow",
    "schema",
]
