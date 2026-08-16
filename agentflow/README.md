# agentflow

A multi-agent workflow system: define a graph of steps, let each step run an
agent, and have the engine handle ordering, parallelism, retries, and failure.

Agents come in two kinds and mix freely in the same graph:

- **Deterministic agents** — plain Python functions. Validation, arithmetic,
  file and database I/O, existing model code. Cheap, reproducible, testable.
- **LLM agents** — one Claude call with a role prompt and, when the output
  feeds something downstream, a JSON Schema constraining its shape.

The split matters: the model decides *what the phases of a project are*, and
Python decides *what they cost*. Anything a computer can get exactly right
should not be delegated to a language model.

## Install

```bash
pip install -r agentflow/requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...     # only needed for real model calls
```

## Try it

```bash
python -m agentflow plan                       # print the execution graph
python -m agentflow agents                     # list the agents in it
python -m agentflow run --example --dry-run    # full run, every model call stubbed
python -m agentflow run --example              # for real (needs an API key)
python -m agentflow run --example --show proposal -v
```

The shipped example (`agentflow/workflows/ai_ml_ops.py`) turns a client brief
into a researched, priced, reviewed proposal:

```
brief          normalize + validate, derive research questions   (deterministic)
  findings     one model call per question, in parallel          (LLM, fan-out)
    analysis   phased plan, sized in engineer-days, with risks   (LLM, structured)
      estimate day rates, contingency, schedule                  (deterministic)
        readiness  weighted go/no-go score                       (deterministic)
        proposal   Markdown draft                                (LLM)
          critique structured review                             (LLM, structured)
            revision  runs only if the critic flags something     (LLM, conditional)
              final     revision if present, else the draft
                publish write the document to disk               (deterministic)
```

## Writing a workflow

```python
from agentflow import LLMAgent, Step, Workflow, agent, ref, run_workflow, schema

@agent(name="load")
def load(path: str) -> dict:
    """Deterministic work: read and validate the input."""
    return {"rows": [...], "path": path}

summarize = LLMAgent(
    name="summarize",
    system="You summarise datasets for a technical audience.",
    prompt=lambda ctx: f"Summarise these rows:\n{ctx.require('data')['rows']}",
    output_schema=schema({"headline": {"type": "string"}}),
    effort="medium",
)

workflow = Workflow(
    name="quicklook",
    steps=[
        Step(name="data", agent=load, inputs={"path": ref("input.path")}),
        Step(name="summary", agent=summarize, inputs={"data": ref("data")}, retries=1),
    ],
)

result = await run_workflow(workflow, {"path": "sales.csv"})
print(result.output("summary")["headline"])
```

### Wiring: refs and the blackboard

Every step's output lands on a per-run blackboard under the step's name.
`ref("analysis.phases.0.name")` reads into it with dotted paths and list
indices. Run inputs live under `input`.

Referencing a step is what creates the dependency — you rarely need
`depends_on`. A ref **with a default** (`ref("revision", None)`) is a *soft*
dependency: the step still runs after that one, but a skipped or failed
producer just means the default is used. That is how a conditional branch feeds
a step that must run either way.

### Step options

| Option | Effect |
| --- | --- |
| `inputs` | Values passed to the agent; `ref(...)` entries resolve at run time |
| `depends_on` | Extra ordering when there is no data dependency |
| `foreach=ref(...)` | Fan out: run the agent once per list item, in parallel; output is a list |
| `max_parallel` | Cap on concurrent fan-out items |
| `when` | Predicate over run state; false means skip this step |
| `retries`, `retry_backoff` | Retry the agent with exponential backoff |
| `timeout` | Seconds before the step is failed |
| `optional` | This step failing does not fail the run |

Failure semantics: a step whose hard dependencies did not all succeed is
**skipped**, not run with missing inputs. By default the first failure of a
non-optional step aborts the run (`fail_fast=False` keeps independent branches
going). Every step, whatever its outcome, appears in the result with a reason.

### Running it

```python
from agentflow import ConsoleReporter, Engine

engine = Engine(
    workflow,
    registry=registry,          # agents referenced by name
    client=client,              # shared ClaudeClient
    max_concurrency=4,
    listeners=[ConsoleReporter()],
)
result = await engine.run(inputs)

result.status                   # Status.OK | FAILED
result.output("summary")        # a step's output
result.usage.output_tokens      # tokens across the whole run
result.steps["summary"].usage   # ... and per step
```

## Model configuration

All model calls go through `ClaudeClient` (`agentflow/llm.py`), so the defaults
live in one place:

- `claude-opus-5` with **adaptive thinking**. No `budget_tokens`, no
  `temperature`/`top_p`/`top_k` — the current Opus family rejects them.
- `output_config.effort` as the cost/quality dial, per agent
  (`effort="low"` for cheap steps, `"high"` for the hard ones).
- **Server-side refusal fallbacks** (`fallbacks="default"`), so a request
  declined by safety classifiers is retried on a fallback model inside the same
  call. Set `AGENTFLOW_DISABLE_FALLBACKS=1` to turn this off.
- The system prompt is sent as a cached block, and agent system prompts are
  static, so repeated runs read from cache rather than re-processing the prefix.
- Requests above 16k `max_tokens` stream, to avoid HTTP timeouts.
- A refusal that survives the fallback raises `RefusalError` rather than
  returning empty content.

Environment overrides: `AGENTFLOW_MODEL`, `AGENTFLOW_EFFORT`,
`AGENTFLOW_MAX_TOKENS`, `AGENTFLOW_DISABLE_FALLBACKS`.

## Testing

`agentflow.testing.StubClient` stands in for the real client: it returns canned
values per agent, or synthesises one from the agent's output schema, and records
every call. Whole workflows run in milliseconds with no API key — that is what
`--dry-run` uses.

```python
client = StubClient(responses={"critic": {"needs_revision": True, ...}})
result = await Engine(workflow, registry=registry, client=client).run(inputs)
assert len(client.calls_for("researcher")) == 4
```

```bash
python -m pytest tests/ -q
```

## Layout

```
agentflow/
  agent.py         Agent base class, @agent decorator, AgentContext
  blackboard.py    Run state and ref resolution
  workflow.py      Step/Workflow definitions and graph validation
  engine.py        Scheduling, retries, skips, cancellation
  llm.py           Claude client wrapper (the only API-facing code)
  llm_agent.py     LLMAgent and JSON Schema helpers
  registry.py      Name -> agent lookup
  events.py        Event bus and console reporter
  testing.py       StubClient for tests and dry runs
  cli.py           python -m agentflow
  agents/          Example agents (deterministic + LLM)
  workflows/       Example workflow
```
