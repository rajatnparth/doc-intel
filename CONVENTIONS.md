# Code conventions for this repo

These are deliberate, and they exist to make the code *readable while learning*.
Keep them as the repo grows.

## 1. Every import is commented with its origin

At a glance you should know, for any import, whether it's free (stdlib), a
dependency you must install (3rd-party), or our own code (local).

```python
import json                          # stdlib — no install needed
from pydantic import BaseModel       # 3rd-party: pydantic — from requirements.txt
from app.llm.base import LLMClient   # local — app/llm/base.py
```

The three tags, exactly:

| Tag | Meaning | How to recognise it |
|---|---|---|
| `# stdlib` | ships with Python | you never `pip install` it (`json`, `asyncio`, `typing`, `re`, `pathlib`, `dataclasses`, `enum`, `logging`, `uuid`, `functools`, `contextlib`) |
| `# 3rd-party: <pkg>` | installed via pip | it's in `requirements.txt`; **name the package** so the reader knows what to install |
| `# local — <path>` | our own code | give the file it lives in |

Extra notes worth adding inline when true:

- **Submodule vs package:** `from fastapi.responses import ...` is the same
  package as `fastapi` but a deeper module — say so.
- **Lazy imports:** an import inside a function (not at top of file) is a
  design decision — comment *why* it's lazy (e.g. "so stub mode never needs
  `openai` installed").
- **`from __future__ import annotations`:** stdlib but special — must be the
  first statement in the file. Note it.
- **A group of exception-only imports** (see `app/llm/azure.py`) is a hint that
  the file translates them — say what into.

## 2. Docstrings explain WHY, not just what

Every module opens with a docstring that states the decision the file embodies,
not a paraphrase of its code. Comments answer "why is it this way", because the
code already shows "what it does".

## 3. Tests are executable proof of a claim

Each test's name and comments name the concept it proves (e.g.
`test_strict_mode_refuses_to_coerce_money`). Before trusting a test, delete the
code it covers and confirm it fails — a test that passes with the code removed
is testing nothing.

## 4. The seam rule

Nothing under `app/llm/` imports FastAPI. Nothing outside `app/llm/` imports
`openai`. Provider details stay behind the `LLMClient` Protocol.
