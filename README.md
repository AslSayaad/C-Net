# C-Net

A Jarvis-style AI agent that runs on your own machine, in Python. It talks to
you in the terminal and can actually do things: inspect your system, read and
write files, run commands, and remember facts about you between sessions.

Deliberately small — one dependency, everything else is the standard library,
so the whole project is readable end to end.

## Quickstart

```bash
pip install -r cnet/requirements.txt

cp cnet/.env.example .env          # then paste your key into ANTHROPIC_API_KEY
python -m cnet.selftest            # checks the wiring, makes no API calls
python -m cnet                     # start talking
```

One-off questions, without opening a chat session:

```bash
python -m cnet "what time is it?"
python -m cnet --resume            # continue the last conversation
```

Inside the chat, `/help` lists the commands.

## Documentation

[`docs/CNET.md`](docs/CNET.md) is the manual and the build plan — how each part
works, and what gets added next.
