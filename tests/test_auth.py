"""Phase 2 — every auth assertion names the attack it blocks.

These tests never assert "the library works". They assert OUR wiring: that the
route actually consults the verifier, that identity cannot arrive in the body
any more, and that the claims — not the client — decide what retrieval sees.
"""

import time                             # stdlib — build expired/rigged tokens

import jwt                              # 3rd-party: PyJWT — forge tokens the attacker's way

from fastapi.testclient import TestClient  # 3rd-party: fastapi (submodule) — drives the app

from app.auth import AUDIENCE, mint     # local — app/auth.py
from app.config import get_settings     # local — app/config.py
from app.main import app                # local — app/main.py

SECRET = get_settings().auth_jwt_secret
Q = {"question": "how quickly must I report an accident?"}


def _post(client, token: str | None, body=Q):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post("/v1/ask", json=body, headers=headers)


# =============================================================================
# 401s — one per attack.
# =============================================================================
def test_no_token_is_401_with_www_authenticate() -> None:
    with TestClient(app) as client:
        r = _post(client, None)
    assert r.status_code == 401
    # RFC 7235: a 401 MUST say which scheme would have worked.
    assert r.headers["WWW-Authenticate"] == "Bearer"


def test_forged_signature_is_401() -> None:
    """The attack: write your own claims, sign with your own key. The claims
    are perfectly shaped — only the signature betrays them."""
    forged = jwt.encode(
        {"tid": "asha", "groups": ["agent"], "aud": AUDIENCE, "exp": int(time.time()) + 600},
        "attacker-key",
        algorithm="HS256",
    )
    with TestClient(app) as client:
        r = _post(client, forged)
    assert r.status_code == 401


def test_expired_token_is_401() -> None:
    """The attack: a leaked token from last month. `exp` is why leaks age out."""
    with TestClient(app) as client:
        r = _post(client, mint("asha", ["customer"], secret=SECRET, ttl_seconds=-10))
    assert r.status_code == 401


def test_token_without_exp_is_401() -> None:
    """The subtle one: omit exp entirely. `options={"require": ["exp"]}` is the
    difference between "check expiry if present" and "expiry is mandatory" —
    without it, the immortal token VALIDATES."""
    immortal = jwt.encode({"tid": "asha", "groups": ["customer"], "aud": AUDIENCE}, SECRET, algorithm="HS256")
    with TestClient(app) as client:
        r = _post(client, immortal)
    assert r.status_code == 401


def test_wrong_audience_is_401() -> None:
    """The attack: a genuine, in-date token minted for a DIFFERENT service,
    signed with a key we happen to share. Audience scoping is what stops
    cross-service replay."""
    other = jwt.encode(
        {"tid": "asha", "groups": ["customer"], "aud": "other-service", "exp": int(time.time()) + 600},
        SECRET,
        algorithm="HS256",
    )
    with TestClient(app) as client:
        r = _post(client, other)
    assert r.status_code == 401


# =============================================================================
# The old, unsafe shape is REJECTED, not ignored.
# =============================================================================
def test_tenant_id_in_body_is_now_a_422() -> None:
    """Phase 1's scaffolding didn't just stop working — it became a validation
    error. A client silently 'ignored' keeps believing it controls its tenant;
    a client 422'd knows the contract changed."""
    with TestClient(app) as client:
        r = _post(
            client,
            mint("asha", ["customer"], secret=SECRET),
            body={**Q, "tenant_id": "vikram"},
        )
    assert r.status_code == 422
    assert "tenant_id" in r.text


# =============================================================================
# Claims are load-bearing: groups in the TOKEN decide what retrieval sees.
# =============================================================================
def test_groups_claim_gates_the_claims_file() -> None:
    q = {"question": "what is the status of claim CLM-2026-0891?"}

    with TestClient(app) as client:
        agent = _post(client, mint("asha", ["agent"], secret=SECRET), body=q)
        customer = _post(client, mint("asha", ["customer"], secret=SECRET), body=q)

    # The call-centre view cites the claims file...
    assert "Claims File" in agent.text
    # ...the customer's self-service view cannot retrieve it AT ALL — it is not
    # refused, it is invisible (gated.py), so it can never appear in any frame:
    # not as a source, not as a near-miss, not in a snippet.
    assert "Claims File" not in customer.text
