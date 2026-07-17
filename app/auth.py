"""Phase 2 — identity is a property of the TRANSPORT, not the payload.

Until now the principal arrived in the request body, which meant every client
chose its own tenant. This module replaces that with the only thing a client
cannot write for itself: a claim SIGNED by a key it does not hold.

WHAT EACH VERIFICATION STEP BLOCKS (none of them is decoration)
---------------------------------------------------------------
  signature   a forged token. Anyone can WRITE {"tid": "asha"}; only the key
              holder can SIGN it. Without this check a JWT is a suggestion.
  exp         a stolen token that works forever. `require` makes exp mandatory
              — a token WITHOUT an expiry must fail, not skip the check.
  aud         a perfectly valid token minted for a DIFFERENT service, replayed
              against this one. Tokens are scoped to an audience for the same
              reason cheques are made out to a name.

401 vs 403, because they are different sentences:
  401 "who are you?"      — no/bad credentials. MUST carry WWW-Authenticate
                            (RFC 7235) so the client knows which scheme to use.
  403 "I know who you are, and no" — authenticated, not authorised.
The tenant gate never 403s here: a foreign tenant's documents are not
forbidden, they are INVISIBLE (gated.py). You cannot leak the existence of
what you never admit exists.

SMALL APP vs ENTERPRISE
-----------------------
HS256 = one shared secret; fine while minter and verifier are the same
process (dev tokens below). An insurer deployment uses RS256/JWKS: their IdP
(e.g. Entra ID) signs with a private key we NEVER hold, and jwt.decode gets
the rotating public key from a JWKS URL. The swap lands entirely inside
`get_principal` — the route and the Principal never change. Same seam idea
as the LLM factory: the dependency is the interface, the scheme is the detail.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

import logging                          # stdlib — auth failures are logged, not leaked
import time                             # stdlib — iat/exp for the dev minter
from typing import Annotated            # stdlib — the DI annotation

import jwt                              # 3rd-party: PyJWT — sign/verify + claim validation

from fastapi import Depends, HTTPException  # 3rd-party: fastapi — DI + the 401
from fastapi.security import (          # 3rd-party: fastapi (submodule) — parses the
    HTTPAuthorizationCredentials,       #   Authorization header and documents the scheme
    HTTPBearer,                         #   in OpenAPI
)

from app.config import Settings, get_settings   # local — app/config.py
from app.retrieval.gated import Principal       # local — app/retrieval/gated.py

log = logging.getLogger("doc_intel.auth")

ALGORITHM = "HS256"
AUDIENCE = "doc-intel"

# The default in config.py. Its NAME is its documentation; lifespan logs a
# warning whenever the process boots with it. Visible scaffolding, like the
# body-principal was — except now the unsafe part is a VALUE you rotate, not a
# DESIGN you rebuild.
DEV_SECRET = "dev-secret-do-not-deploy"

# auto_error=False: HTTPBearer's own error is a 403 with no WWW-Authenticate,
# which gets both halves of RFC 7235 wrong for a missing credential. We take
# `None` and shape the 401 ourselves.
_bearer = HTTPBearer(auto_error=False)


def _unauthorized(reason: str) -> HTTPException:
    """One 401 for every failure mode, deliberately vague.

    The SPECIFIC reason (expired vs bad signature vs wrong audience) goes to
    the LOG. Telling the caller which check failed is a debugging gift to
    whoever is fabricating tokens; the legitimate client's fix is the same
    either way — get a fresh token.
    """
    log.info("auth rejected: %s", reason)
    return HTTPException(
        status_code=401,
        detail="invalid or missing credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_principal(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> Principal:
    """The ONLY place a Principal is constructed on a request path.

    gated.py insists the principal is an argument, never ambient state. This
    dependency is where that argument is born — from verified claims, nowhere
    else. Grep for `Principal(` under app/: one HTTP site, by design.
    """
    if credentials is None:
        raise _unauthorized("no bearer token presented")

    try:
        claims = jwt.decode(
            credentials.credentials,
            settings.auth_jwt_secret,
            algorithms=[ALGORITHM],     # pin the algorithm — never trust the header's
                                        #   `alg` field; that is the classic JWT attack
                                        #   (alg=none, or HS256 verified with a public key)
            audience=AUDIENCE,
            options={"require": ["exp", "aud"]},
        )
    except jwt.InvalidTokenError as exc:
        # PyJWT's exception tree (expired/audience/signature) all descend from
        # InvalidTokenError — one except, specific reason preserved in the log.
        raise _unauthorized(f"{type(exc).__name__}: {exc}") from exc

    tid = claims.get("tid")
    groups = claims.get("groups")
    if not isinstance(tid, str) or not tid:
        raise _unauthorized("token verified but has no usable `tid` claim")
    if not isinstance(groups, list) or not all(isinstance(g, str) for g in groups) or not groups:
        raise _unauthorized("token verified but has no usable `groups` claim")

    return Principal(tenant_id=tid, groups=frozenset(groups))


PrincipalDep = Annotated[Principal, Depends(get_principal)]


# =============================================================================
# The dev minter. In production this function DOES NOT EXIST — tokens come
# from the IdP, and the verifier holding a signing key would itself be a
# finding. It lives here so the quickstart stays keyless:
#
#     TOKEN=$(python -m app.auth --tenant asha --groups customer)
# =============================================================================
def mint(
    tenant: str,
    groups: list[str],
    *,
    secret: str,
    ttl_seconds: int = 3600,
) -> str:
    now = int(time.time())
    return jwt.encode(
        {"tid": tenant, "groups": groups, "aud": AUDIENCE, "iat": now, "exp": now + ttl_seconds},
        secret,
        algorithm=ALGORITHM,
    )


if __name__ == "__main__":
    import argparse                     # stdlib — CLI for the dev minter only

    parser = argparse.ArgumentParser(description="Mint a DEV token (never in production).")
    parser.add_argument("--tenant", default="asha")
    parser.add_argument("--groups", default="customer", help="comma-separated, e.g. customer or agent")
    parser.add_argument("--ttl", type=int, default=3600)
    args = parser.parse_args()

    print(
        mint(
            args.tenant,
            [g.strip() for g in args.groups.split(",") if g.strip()],
            secret=get_settings().auth_jwt_secret,
            ttl_seconds=args.ttl,
        )
    )
