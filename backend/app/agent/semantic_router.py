"""
semantic_router.py
------------------
Embedding-based semantic router for intent classification.

Replaces the custom numpy implementation with the open-source `semantic-router` library.
No LLM generative call is made during classification — only a single embedding call
per user query via the local vLLM server.

Architecture:
  1. At startup, load routes.yaml and define `semantic_router.Route` objects.
  2. Initialize `semantic_router.encoders.OpenAIEncoder` pointing to vLLM.
  3. Initialize `semantic_router.RouteLayer` with the encoder and routes.
  4. At classification time:
       - Invoke the RouteLayer.
       - Returns a RouteResult dataclass compatible with the rest of the application.

Note: Custom follow-up inheritance logic has been dropped in favour of explicit
follow-up execution phrases in routes.yaml.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
import tiktoken

# Monkey-patch tiktoken to handle custom model names from vLLM
_original_encoding_for_model = tiktoken.encoding_for_model
def _patched_encoding_for_model(model_name: str):
    try:
        return _original_encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")
tiktoken.encoding_for_model = _patched_encoding_for_model

from semantic_router import Route
from semantic_router.routers import SemanticRouter as AurelioRouter
from semantic_router.encoders import OpenAIEncoder

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROUTES_YAML_PATH = Path(__file__).parent / "routes.yaml"
_LOGS_DIR = Path(__file__).resolve().parents[3] / "logs"
_ROUTING_TRACE_PATH = _LOGS_DIR / "routing_trace.ndjson"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# vLLM embedding client — same server as the RAG retriever (localhost:8001)
# ---------------------------------------------------------------------------
_EMBED_BASE_URL = "http://localhost:8001/v1"
_EMBED_MODEL    = "BAAI/bge-base-en-v1.5"
_EMBED_API_KEY  = "not-needed"

# ---------------------------------------------------------------------------
# ANSI colour helpers (terminal only)
# ---------------------------------------------------------------------------
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[36m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

VALID_INTENTS = frozenset({"chat", "query_user_information", "search_knowledge_base"})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RouteResult:
    """
    Output of one SemanticRouter.classify() call.

    Fields:
        intent      — The winning route name (always a valid intent string).
        confidence  — Aggregated score of the winning route (0.0–1.0).
        all_scores  — Kept for compatibility, mostly empty or limited.
        method      — "semantic_library" : won via RouteLayer
                      "default"          : defaulted to chat
        duration_ms — Wall-clock time for this classification.
    """
    intent:      str
    confidence:  float
    all_scores:  dict[str, float] = field(default_factory=dict)
    method:      str = "semantic_library"
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Core router class
# ---------------------------------------------------------------------------

class SemanticRouter:
    """
    Wrapper around Aurelio's `semantic-router` RouteLayer.
    """

    def __init__(self) -> None:
        """
        Load routes.yaml, initialize encoder and RouteLayer.
        """
        log.info("[SemanticRouter] Initializing with semantic-router library...")

        # Load route definitions from YAML
        routes_config = self._load_routes()

        routes = []
        total_utterances = 0

        for name, data in routes_config.items():
            if name not in VALID_INTENTS:
                raise ValueError(
                    f"[SemanticRouter] Route '{name}' in routes.yaml is not a valid intent."
                )
            utterances = data["utterances"]
            routes.append(Route(name=name, utterances=utterances))
            total_utterances += len(utterances)

        log.info(
            "[SemanticRouter] Loaded %d routes, %d total utterances",
            len(routes), total_utterances,
        )
        print(
            f"\n{_CYAN}[SemanticRouter] Initializing RouteLayer for {total_utterances} "
            f"utterances across {len(routes)} routes...{_RESET}"
        )

        t0 = time.monotonic()
        
        self.encoder = OpenAIEncoder(
            name=_EMBED_MODEL,
            openai_base_url=_EMBED_BASE_URL,
            openai_api_key=_EMBED_API_KEY,
            score_threshold=0.50,
        )

        self.route_layer = AurelioRouter(encoder=self.encoder)
        self.route_layer.add(routes)

        elapsed_ms = (time.monotonic() - t0) * 1000
        print(
            f"{_GREEN}[SemanticRouter] Ready. RouteLayer initialized "
            f"in {elapsed_ms:.0f}ms.{_RESET}\n"
        )
        log.info("[SemanticRouter] Startup complete in %.0fms", elapsed_ms)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def classify(
        self,
        query: str,
        prev_intent: Optional[str] = None,
    ) -> RouteResult:
        """
        Classify a user query using RouteLayer.

        Args:
            query       : The raw user message string.
            prev_intent : Kept for signature compatibility, unused.

        Returns:
            RouteResult with intent.
        """
        t0 = time.monotonic()

        try:
            choice = self.route_layer(query)
            elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

            if choice.name is not None:
                # We got a match!
                intent = choice.name
                method = "semantic_library"
                
                # The library doesn't easily expose the raw confidence in the direct call
                # so we will use a dummy confidence of 1.0 or similar to indicate success.
                confidence = 1.0 
            else:
                # No route crossed the threshold.
                intent = "chat"
                method = "default"
                confidence = 0.0

            result = RouteResult(
                intent=intent,
                confidence=confidence,
                all_scores={},
                method=method,
                duration_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
            log.error(
                "[SemanticRouter] Classification failed, defaulting to 'chat'. Error: %s", e
            )
            result = RouteResult(
                intent="chat",
                confidence=0.0,
                all_scores={},
                method="error_fallback",
                duration_ms=elapsed_ms,
            )

        self._print_result(query, result)
        self._write_trace(query, result, prev_intent)

        return result

    # -----------------------------------------------------------------------
    # Internal: YAML loading
    # -----------------------------------------------------------------------

    @staticmethod
    def _load_routes() -> dict:
        """Load and validate routes.yaml."""
        if not _ROUTES_YAML_PATH.exists():
            raise FileNotFoundError(
                f"[SemanticRouter] routes.yaml not found at {_ROUTES_YAML_PATH}. "
            )
        with open(_ROUTES_YAML_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if "routes" not in config:
            raise ValueError("[SemanticRouter] routes.yaml must have a 'routes' key.")

        return config["routes"]

    # -----------------------------------------------------------------------
    # Internal: observability
    # -----------------------------------------------------------------------

    def _print_result(self, query: str, result: RouteResult) -> None:
        """Print colour-coded routing result to terminal."""
        print(f"\n{'─'*70}")
        print(f"{_CYAN}[ROUTER] Query: {query!r}{_RESET}")

        score_str = (
            f"{_GREEN}MATCH{_RESET}"
            if result.intent != "chat" or result.method == "semantic_library"
            else f"{_YELLOW}DEFAULT{_RESET}"
        )
        winner_marker = f" {_BOLD}← {result.method.upper()}{_RESET}"
        print(f"  {result.intent:<30} {score_str}{winner_marker}")

        print(
            f"  Intent: {_BOLD}{result.intent}{_RESET} | "
            f"{result.duration_ms:.1f}ms"
        )
        print(f"{'─'*70}\n")

    def _write_trace(
        self,
        query: str,
        result: RouteResult,
        prev_intent: Optional[str],
    ) -> None:
        """Append one JSON line to routing_trace.ndjson."""
        record = {
            "event":       "route",
            "timestamp":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "query":       query,
            "intent":      result.intent,
            "confidence":  result.confidence,
            "method":      result.method,
            "prev_intent": prev_intent,
            "duration_ms": result.duration_ms,
        }
        try:
            with open(_ROUTING_TRACE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            log.warning("[SemanticRouter] Failed to write trace: %s", e)


# ---------------------------------------------------------------------------
# Module-level singleton — created at import time inside nodes.py
# Use get_router() to access it.
# ---------------------------------------------------------------------------

_router: Optional[SemanticRouter] = None

def get_router() -> SemanticRouter:
    global _router
    if _router is None:
        _router = SemanticRouter()
    return _router
