"""
test_router.py
--------------
Smoke test for the SemanticRouter.

Tests all three routes with representative queries including:
  - Canonical clear-intent queries
  - Edge cases: informal + finance words, statements vs queries
  - Follow-up inheritance: short ambiguous queries with prev_intent context

Run with:
  uv run --project backend python backend/scripts/test_router.py

Prerequisites:
  - BAAI/bge-base-en-v1.5 must be running on localhost:8001
    (same vLLM Docker container used by the main app)
"""

import sys
import time
from pathlib import Path

# Add backend/ to sys.path so 'app.' imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent.semantic_router import SemanticRouter

# ─────────────────────────────────────────────────────────────────────────────
# ANSI colours
# ─────────────────────────────────────────────────────────────────────────────
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[36m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


# ─────────────────────────────────────────────────────────────────────────────
# Test cases: (query, prev_intent, expected_intent, description)
# ─────────────────────────────────────────────────────────────────────────────

TEST_CASES: list[tuple[str, str | None, str, str]] = [

    # ── CHAT — Canonical ──────────────────────────────────────────────────────
    ("hello",                       None,  "chat",                   "greeting: hello"),
    ("hi",                          None,  "chat",                   "greeting: hi"),
    ("hey",                         None,  "chat",                   "greeting: hey"),
    ("yo",                          None,  "chat",                   "greeting: yo alone"),
    ("good morning",                None,  "chat",                   "greeting: good morning"),
    ("my name is Neelavro",         None,  "chat",                   "introduction"),
    ("who are you?",                None,  "chat",                   "identity question"),
    ("thanks",                      None,  "chat",                   "acknowledgement"),
    ("thank you so much",           None,  "chat",                   "acknowledgement: long"),
    ("that was helpful",            None,  "chat",                   "acknowledgement: reaction"),
    ("bye",                         None,  "chat",                   "farewell"),
    ("what is the weather today?",  None,  "chat",                   "general knowledge"),
    ("tell me a joke",              None,  "chat",                   "entertainment request"),
    ("I spent a lot this month",    None,  "chat",                   "CRITICAL: statement, not query"),
    ("I bought something expensive yesterday", None, "chat",         "CRITICAL: past statement"),
    ("I think I overspent this week", None, "chat",                  "CRITICAL: opinion/statement"),
    ("I am having a bad day",       None,  "chat",                   "emotional statement"),
    ("what is compound interest?",  None,  "chat",                   "general finance knowledge"),
    ("explain inflation",           None,  "chat",                   "general economic concept"),

    # ── QUERY_USER_INFORMATION — Canonical ────────────────────────────────────
    ("show me my transactions",             None, "query_user_information", "core: list transactions"),
    ("what was my last transaction?",       None, "query_user_information", "core: last transaction"),
    ("how much did I spend this month?",    None, "query_user_information", "core: monthly spend"),
    ("how much did I spend on food?",       None, "query_user_information", "core: category spend"),
    ("list my subscription charges",        None, "query_user_information", "core: subscriptions"),
    ("show me my most expensive purchase",  None, "query_user_information", "core: max transaction"),
    ("what is my total income this month?", None, "query_user_information", "core: income"),
    ("find transactions from Netflix",      None, "query_user_information", "core: merchant search"),
    ("how many transactions do I have?",    None, "query_user_information", "core: count"),
    ("show me debit transactions",          None, "query_user_information", "core: direction filter"),
    ("make a tool call to check my spending", None, "query_user_information", "explicit tool request"),
    ("query the database for my expenses",  None, "query_user_information", "explicit DB request"),

    # ── QUERY_USER_INFORMATION — Informal / Edge Cases ────────────────────────
    ("yo list my subscriptions",            None, "query_user_information", "EDGE: informal + query"),
    ("yo from my transaction what is the most expensive item in my transaction history",
                                            None, "query_user_information", "EDGE: yo + complex query"),
    ("hey can you check my food expenses?", None, "query_user_information", "EDGE: informal greeting + query"),
    ("bro show me what I spent on electronics", None, "query_user_information", "EDGE: bro + query"),
    ("what did I blow on subscriptions?",   None, "query_user_information", "EDGE: slang spend verb"),
    ("how much did I drop on food last week?", None, "query_user_information", "EDGE: colloquial"),

    # ── QUERY_USER_INFORMATION — Follow-up inheritance ────────────────────────
    ("what about food?",      "query_user_information", "query_user_information", "FOLLOW-UP: short category"),
    ("and electronics?",      "query_user_information", "query_user_information", "FOLLOW-UP: even shorter"),
    ("what about last month?","query_user_information", "query_user_information", "FOLLOW-UP: time period"),
    ("how many were there?",  "query_user_information", "query_user_information", "FOLLOW-UP: count follow-up"),
    ("and the DPS plans?",    "search_knowledge_base",  "search_knowledge_base",  "FOLLOW-UP: KB follow-up"),

    # ── SEARCH_KNOWLEDGE_BASE — Canonical ─────────────────────────────────────
    ("what is the ATM withdrawal limit?",       None, "search_knowledge_base", "core: ATM limit"),
    ("how do I activate my debit card?",        None, "search_knowledge_base", "core: card activation"),
    ("what are the fees for RTGS transfer?",    None, "search_knowledge_base", "core: fees"),
    ("how does fixed deposit work?",            None, "search_knowledge_base", "core: FD explanation"),
    ("what is the interest rate on FD?",        None, "search_knowledge_base", "core: interest rate"),
    ("how do I block my card?",                 None, "search_knowledge_base", "core: card management"),
    ("what documents do I need to open an account?", None, "search_knowledge_base", "core: eligibility"),
    ("tell me about the DPS plans",             None, "search_knowledge_base", "core: product info"),
    ("what is the annual fee for the debit card?", None, "search_knowledge_base", "core: annual fee"),
    ("how do I contact customer support?",      None, "search_knowledge_base", "core: support"),
    ("what happens if my account is dormant?",  None, "search_knowledge_base", "core: account policy"),
    ("how does Scan and Pay work?",             None, "search_knowledge_base", "core: feature how-to"),
    ("will Prime NOW ever ask for my PIN?",     None, "search_knowledge_base", "core: security policy"),

]


def run_tests(router: SemanticRouter) -> None:
    print(f"\n{'═'*80}")
    print(f"  {_BOLD}Qurio SemanticRouter — Smoke Test{_RESET}")
    print(f"{'═'*80}\n")

    passed = 0
    failed = 0
    failures: list[tuple[str, str, str]] = []

    t_total = time.monotonic()

    for query, prev_intent, expected, description in TEST_CASES:
        t0 = time.monotonic()
        result = router.classify(query, prev_intent=prev_intent)
        elapsed = (time.monotonic() - t0) * 1000

        ok = result.intent == expected
        if ok:
            passed += 1
            status = f"{_GREEN}✓ PASS{_RESET}"
        else:
            failed += 1
            status = f"{_RED}✗ FAIL{_RESET}"
            failures.append((description, expected, result.intent))

        # Print compact result line
        conf_str = (
            f"{_GREEN}{result.confidence:.3f}{_RESET}"
            if result.confidence >= 0.50
            else f"{_YELLOW}{result.confidence:.3f}{_RESET}"
        )
        prev_str = f" [prev={prev_intent}]" if prev_intent else ""
        print(
            f"  {status}  {description:<55} "
            f"→ {result.intent:<25} conf={conf_str} "
            f"[{result.method}] {elapsed:.0f}ms{prev_str}"
        )

    total_elapsed = (time.monotonic() - t_total) * 1000

    # Summary
    print(f"\n{'═'*80}")
    total = passed + failed
    pass_rate = (passed / total * 100) if total else 0
    color = _GREEN if failed == 0 else _YELLOW if failed <= 3 else _RED
    print(
        f"  {_BOLD}Results: {color}{passed}/{total} passed ({pass_rate:.0f}%){_RESET}  "
        f"| Total: {total_elapsed:.0f}ms | Avg: {total_elapsed/total:.0f}ms/query"
    )

    if failures:
        print(f"\n  {_RED}{_BOLD}FAILURES:{_RESET}")
        for desc, exp, got in failures:
            print(f"    {_RED}✗{_RESET}  {desc}")
            print(f"       Expected: {_BOLD}{exp}{_RESET}  Got: {_RED}{got}{_RESET}")

    print(f"{'═'*80}\n")


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.WARNING,
        format="%(name)s | %(levelname)s | %(message)s",
    )
    # Silence httpx noise
    logging.getLogger("httpx").setLevel(logging.WARNING)

    print(f"\n{_CYAN}Initializing SemanticRouter (embedding utterances)...{_RESET}")
    router = SemanticRouter()
    run_tests(router)
