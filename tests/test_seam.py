"""The seam rule, as a test instead of a README sentence.

Three mechanical claims:
  1. no module outside app/llm imports a provider SDK (openai, fastembed)
  2. nothing inside app/llm imports FastAPI
  3. outside app/llm, only the seam's public face is imported —
     app.llm / app.llm.base / app.llm.factory, never a concrete client

AST-based rather than grep-based, so imports inside FUNCTION BODIES are caught
too — moving `from fastembed import TextEmbedding` into a lazy helper is
exactly how the violation happened the first time. It compiled, it ran, every
test passed, and "swap the embedding provider by config" was quietly false.
"""

import ast                              # stdlib — parse modules without importing them
from pathlib import Path                # stdlib — walk the app/ tree

APP = Path(__file__).resolve().parent.parent / "app"
SEAM = APP / "llm"
STORE_SEAM = APP / "store"

PROVIDER_SDKS = {"openai", "fastembed"}
STORE_SDKS = {"qdrant_client"}
WEB_FRAMEWORK = {"fastapi", "starlette"}
SEAM_PUBLIC_FACE = {"app.llm", "app.llm.base", "app.llm.factory"}


def _imports(path: Path) -> set[str]:
    """Every dotted module name imported anywhere in the file — including
    imports nested inside functions, which is where lazy ones hide."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def _outside_seam() -> list[Path]:
    return [f for f in sorted(APP.rglob("*.py")) if SEAM not in f.parents]


def _inside_seam() -> list[Path]:
    return sorted(SEAM.rglob("*.py"))


def test_no_provider_sdk_outside_the_seam() -> None:
    """Rule 1. `openai` lives in azure.py; `fastembed` lives in local.py."""
    offenders = {
        str(f.relative_to(APP.parent)): sdks
        for f in _outside_seam()
        if (sdks := {i for i in _imports(f) if i.split(".")[0] in PROVIDER_SDKS})
    }
    assert not offenders, f"provider SDK imported outside app/llm: {offenders}"


def test_no_web_framework_inside_the_seam() -> None:
    """Rule 2 — the original seam rule, now also executable."""
    offenders = {
        str(f.relative_to(APP.parent)): fw
        for f in _inside_seam()
        if (fw := {i for i in _imports(f) if i.split(".")[0] in WEB_FRAMEWORK})
    }
    assert not offenders, f"web framework imported inside app/llm: {offenders}"


def test_no_store_sdk_outside_the_storage_seam() -> None:
    """Phase 5's fence, same shape as the others: `qdrant_client` lives in
    app/store/qdrant.py and nowhere else. A direct import in retrieval code
    would weld the gate to one backend — the exact mistake the embedding
    seam already caught once."""
    offenders = {
        str(f.relative_to(APP.parent)): sdks
        for f in sorted(APP.rglob("*.py"))
        if f != STORE_SEAM / "qdrant.py"
        and (sdks := {i for i in _imports(f) if i.split(".")[0] in STORE_SDKS})
    }
    assert not offenders, f"store SDK imported outside app/store/qdrant.py: {offenders}"


def test_no_web_framework_inside_the_storage_seam() -> None:
    offenders = {
        str(f.relative_to(APP.parent)): fw
        for f in sorted(STORE_SEAM.rglob("*.py"))
        if (fw := {i for i in _imports(f) if i.split(".")[0] in WEB_FRAMEWORK})
    }
    assert not offenders, f"web framework imported inside app/store: {offenders}"


def test_only_the_seams_public_face_is_imported() -> None:
    """Rule 3. `from app.llm.azure import ...` in retrieval code would bind it
    to one provider — the factory exists so nobody else has to choose."""
    offenders = {
        str(f.relative_to(APP.parent)): private
        for f in _outside_seam()
        if (
            private := {
                i
                for i in _imports(f)
                if (i == "app.llm" or i.startswith("app.llm."))
                and i not in SEAM_PUBLIC_FACE
            }
        )
    }
    assert not offenders, f"concrete client imported past the factory: {offenders}"
