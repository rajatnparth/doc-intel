"""Session-wide test setup.

AUTH_JWT_SECRET is generated FRESH for every test run, before the app is ever
imported. Two claims, both load-bearing:

  1. The suite depends on no fixed secret — the strongest possible proof that
     nothing in the codebase has quietly memorised one.
  2. `git clone && pytest` needs zero setup, while the SERVING path still
     fails closed: validate_for_serving() would refuse to boot uvicorn without
     a real secret. Tests and servers get their secrets the same way
     production does — from the environment — just from different suppliers.

setdefault, not assignment: an explicitly exported AUTH_JWT_SECRET (e.g. in a
debugging session) still wins, matching pydantic-settings precedence.
"""

import os                               # stdlib — the environment IS the config channel
import secrets                          # stdlib — cryptographically strong randomness

os.environ.setdefault("AUTH_JWT_SECRET", secrets.token_hex(32))

# The suite must be hermetic with respect to STORAGE too: a developer whose
# .env says VECTOR_STORE=qdrant must not have tests silently reading (or
# racing a running server for the folder lock on) their local database.
# Tests that exercise the qdrant store do so EXPLICITLY, in tmp_path
# (tests/test_store.py); everything app-level runs on memory. Same precedence
# trick as the secret: setdefault beats .env, an exported var beats both.
os.environ.setdefault("VECTOR_STORE", "memory")

# Audit records land in a per-run temp file, not in the developer's
# var/audit.jsonl — the suite must neither read nor grow local state. Tests
# that assert on records fetch them BY REQUEST ID (the x-request-id response
# header), so sharing one file across the run is safe by construction.
import tempfile                         # stdlib — per-run isolation for the audit sink

os.environ.setdefault(
    "AUDIT_PATH",
    os.path.join(tempfile.mkdtemp(prefix="doc-intel-test-audit-"), "audit.jsonl"),
)
