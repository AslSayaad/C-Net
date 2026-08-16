# C-Net — your own AI agent

C-Net is a Jarvis-style assistant that runs on your laptop, in Python. It talks
to you in the terminal, and it can actually *do* things: check the time, look at
your machine, read and write files, run commands, and remember facts about you
between sessions.

This document is both the manual and the learning path. Phase 1 is already
built and working — the rest is what we add next, in order.

---

## 1. Get it running (5 minutes)

```bash
# 1. install the one dependency
pip install -r cnet/requirements.txt

# 2. add your API key  (get one at https://console.anthropic.com)
cp cnet/.env.example .env
#    then open .env and paste your key into ANTHROPIC_API_KEY

# 3. check everything is wired up (this makes no API calls, costs nothing)
python -m cnet.selftest

# 4. talk to it
python -m cnet
```

One-off questions without the chat session:

```bash
python -m cnet "what time is it?"
python -m cnet --verbose "how much disk space do I have left?"
python -m cnet --resume            # continue your last conversation
python -m cnet --model claude-sonnet-5   # cheaper model while practising
```

Inside the chat, `/help` lists the commands (`/tools`, `/memory`, `/new`,
`/effort`, `/config`, `/quit`).

### Try these first

```
what time is it?
what kind of machine am I on?
remember that I'm learning AI engineering
write a file called notes.md with 3 things to learn this week
what's in my workspace?
```

Watch the `⚙ tool_name(...)` lines appear. That's C-Net deciding on its own to
use a tool — that is the whole game.

---

## 2. How it works

Five files, each with one job:

```
cnet/
├── config.py        settings — model, workspace, safety switches
├── personality.py   the system prompt: who C-Net is and how it behaves
├── memory.py        conversation history + long-term facts
├── brain.py         ← THE AGENT LOOP. Read this one first.
├── cli.py           the terminal you type into
└── tools/           the things C-Net can actually do
    ├── base.py          plumbing: turns a Python function into a tool
    ├── system_tools.py  clock, machine info
    ├── file_tools.py    list / read / write, sandboxed
    ├── memory_tools.py  remember / recall / forget
    └── shell_tools.py   run commands (off by default)
```

**The loop, in `brain.py`, is the entire idea:**

```
1. Send the conversation + the list of tools to Claude.
2. Claude replies with either text, or a request to use a tool.
3. If it's text  → done, show it.
4. If it's a tool → run the Python function, append the result, GO TO 1.
```

That's it. An "AI agent" is a while-loop around a model that can ask for
function calls. Everything else — memory, personality, safety — is scaffolding
you build around those ~40 lines.

### Four concepts worth understanding properly

**1. The model is stateless.** The API remembers *nothing* between calls. Every
single turn, we resend the whole conversation. "Memory" is not a model feature —
it is a list in `memory.py` that we maintain and resend. Once this clicks, most
of agent engineering makes sense.

**2. Tool results go back as a `user` message.** It feels wrong the first time.
The assistant asks for `get_datetime`, and the answer is delivered as if the
user said it. Also: all results from one turn must go in a *single* message.

**3. The tool description is the prompt.** The model decides whether to use a
tool based only on its name, description, and schema. A vague description means
a tool that never gets used. That text is engineering, not documentation.

**4. Model output is untrusted input.** When the model hands you a file path,
treat it exactly like a path typed by a stranger on the internet. That is why
`file_tools.py` resolves every path and refuses anything outside the workspace,
and why `run_command` asks you before every single command.

---

## 3. Add your own tool (the exercise to do next)

This is the highest-value thing you can practise. Adding a capability is one
function. Open `cnet/tools/system_tools.py` and add:

```python
    @registry.add(
        "get_battery",
        "Check the laptop's battery percentage and whether it is charging. "
        "Use this when the user asks about battery or power.",
    )
    def get_battery() -> str:
        import psutil                       # pip install psutil
        b = psutil.sensors_battery()
        if b is None:
            return "No battery found — this machine is probably a desktop."
        state = "charging" if b.power_plugged else "on battery"
        return f"Battery at {b.percent:.0f}% ({state})."
```

Restart C-Net and ask "how's my battery?". Nothing else to change — the
registry picks it up, the schema is generated, the model sees it.

**Tools take arguments** by declaring a JSON schema:

```python
    @registry.add(
        "flip_coin",
        "Flip a coin a number of times and report the results.",
        {
            "type": "object",
            "properties": {
                "times": {"type": "integer", "description": "How many flips, 1-100."}
            },
            "required": ["times"],
        },
    )
    def flip_coin(times: int) -> str:
        import random
        flips = [random.choice("HT") for _ in range(min(times, 100))]
        return f"{' '.join(flips)}  ({flips.count('H')} heads)"
```

**Tools that change things** should ask permission first — add a
`confirm_prompt` and the CLI will stop and ask you:

```python
        confirm_prompt=lambda args: f"Delete {args['path']}?",
```

Ideas to build, roughly in order of difficulty: a weather tool (hits a public
API), a timer/reminder, a note-taker, a "summarise this file" tool, a music
controller, a git status tool.

---

## 4. The roadmap

Phase 1 is done. Each phase after it is a real, self-contained project — build
them in order, because each one depends on the ideas in the last.

### ✅ Phase 1 — The foundation (built)
A model that talks, tools it can call on its own, memory that survives
restarts, a sandbox, and a personality.
*You learned:* the agent loop, tool calling, statelessness, prompt design.

### Phase 2 — More senses and hands
Add 5–10 tools of your own: weather, calendar, notes, open apps, git, music.
No new concepts, just repetition until tool-writing is automatic.
*Learn:* API integration, schema design, when a tool is worth its context cost.
*Start here.* Do not skip to Phase 3 — this is the phase that builds fluency.

### Phase 3 — Web search
Give C-Net live knowledge. The easy path is Anthropic's built-in server-side
web search tool — one entry in the `tools` list and the search runs on
Anthropic's side, no scraping code:

```python
tools=[{"type": "web_search_20260209", "name": "web_search"}, *registry.specs()]
```

The harder, more educational path is your own tool using an API like Brave or
Tavily, plus fetching and trimming pages before they hit the context window.
*Learn:* server tools vs client tools, citations, context budgeting.

### Phase 4 — Deep thinking
Right now C-Net answers in one pass. Make it able to work on a problem.
- Turn `effort` up (`/effort xhigh`) and see how the behaviour changes.
- Add a planning step: ask it to write a plan to a file, then execute it.
- Add sub-agents: a second `Brain` with a narrow prompt and fewer tools, that
  the main C-Net can delegate a self-contained job to.
- Add self-verification: after finishing, check the work against the request.
*Learn:* adaptive thinking, effort, multi-agent orchestration, task decomposition.

### Phase 5 — Real memory (RAG)
The current `facts.json` does not scale past a few dozen facts. Replace
keyword matching with semantic search:
1. Split documents into chunks.
2. Turn each chunk into an embedding (a vector of numbers).
3. Store them in a vector DB (`chromadb` is the easiest to start with).
4. On each question, retrieve the most similar chunks and add them to the prompt.
Then point it at your own notes, code, or PDFs.
*Learn:* embeddings, vector search, chunking, retrieval-augmented generation.
This is the single most employable skill in the list.

### Phase 6 — Machine learning of your own
Now the ML you asked about, on top of everything above:
- **Start small:** a classifier that routes each message ("chat" vs "code" vs
  "search") so C-Net picks the cheapest model for the job. scikit-learn,
  a hundred labelled examples, one afternoon.
- **Then:** learn from your own history — which answers you accepted, which
  tools actually got used, what you ask at what time of day.
- **Then:** run a small local model (Ollama + Llama) for offline/cheap tasks,
  with Claude as the escalation path.
*Learn:* the difference between using models and training them, evaluation,
the cost/quality trade-off.

### Phase 7 — Always on
Voice in (`whisper`), voice out (`piper` / system TTS), a hotkey to summon it,
a background service that watches your calendar and inbox and speaks up
unprompted. This is where it stops feeling like a chatbot.

---

## 5. Safety, in plain terms

You are giving a language model access to your computer. The defaults here are
deliberately careful:

- **Sandbox.** File tools cannot touch anything outside `~/cnet-workspace`.
  Paths are resolved and checked, so `../../.ssh/id_rsa` is refused.
- **Shell is off.** `CNET_ALLOW_SHELL=false` by default. When you turn it on,
  every command still has to be approved by you, one at a time. Read them.
- **Writes are confirmed.** `CNET_CONFIRM_WRITES=true` by default.
- **Your key stays local.** `.env` is gitignored. Never commit it, never paste
  it into a chat.

Before you widen any of these, ask: if the model got something badly wrong
here — or if a web page it read told it to do something — what is the worst
outcome? Keep the blast radius small while you are learning.

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `No ANTHROPIC_API_KEY found` | Key not set. `cp cnet/.env.example .env`, paste your key. |
| `No module named cnet` | Run from the project root, not from inside `cnet/`. |
| `No module named anthropic` | `pip install -r cnet/requirements.txt` |
| It never uses a tool | The description is too vague. Say *when* to use it, not just what it does. |
| Answers are too long | Edit `personality.py` — it is just text. |
| Replies feel slow | `/effort low`, or `--model claude-sonnet-5`. |
| Costs adding up | Sonnet is cheaper than Opus; low effort is cheaper than high; `/new` clears history, and history is resent every turn. |

Run `python -m cnet.selftest` any time something feels broken — it checks the
whole setup without spending a token.
