"""
config.py — Controlled vocabulary and metadata constants for the Prime NOW
knowledge base ingestion pipeline.

These values are referenced by the chunker and the future retrieval node.
Changing a product_area key here must be reflected in the headings_map.json
and any ChromaDB where-filter logic in the agent.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parents[3] / "knowledge_database"
HEADINGS_MAP_PATH = KNOWLEDGE_BASE_DIR / "headings_map.json"
CHUNKS_OUTPUT_PATH = Path(__file__).resolve().parent / "chunks.json"

# ── Product area vocabulary (used as ChromaDB `where` filter values) ─────────
# Every chunk is tagged with exactly one of these strings.
PRODUCT_AREAS = {
    "debit_card",
    "transactional_features",
    "transaction_account",
    "fd",
    "goal_dps",
    "flexi_dps",
    "support",
}

# ── File → product_area mapping ───────────────────────────────────────────────
# Maps each source .txt filename stem to its controlled product_area tag.
FILE_TO_PRODUCT_AREA: dict[str, str] = {
    "debit_card":                       "debit_card",
    "transactional_features":           "transactional_features",
    "prime_now_transaction_account":    "transaction_account",
    "prime_now_fixed_deposit_account":  "fd",
    "prime_now_goal_based_fixed_dps":   "goal_dps",
    "prime_now_flexi_installment_dps":  "flexi_dps",
    "prime_now_customer_service":       "support",
}

# ── Hand-curated topics per parent_id ─────────────────────────────────────────
# 2–4 keyword tags per section. Used for debugging / future BM25 pass.
# Do NOT use in ChromaDB where-filters in v1.
PARENT_TOPICS: dict[str, list[str]] = {
    # debit_card
    "debit_card__onboarding_customization": ["card_design", "name_personalization", "onboarding", "issuance"],
    "debit_card__delivery":                  ["delivery", "address", "otp", "courier"],
    "debit_card__activation_security":       ["qr_code", "activation", "pin", "security"],
    "debit_card__details":                   ["card_number", "cvv", "expiry", "view_card"],
    "debit_card__atm_limits_fees":           ["atm", "withdrawal", "cash_limit", "npsb_fee"],
    "debit_card__management":                ["reissue", "block", "unblock", "lost_card"],
    "debit_card__fees_transactions":         ["annual_fee", "contactless", "pos", "dual_currency"],

    # transactional_features
    "transactional_features__payments_payees":    ["scan_pay", "qr", "payee", "recurring", "favorite"],
    "transactional_features__recharge_add_money": ["top_up", "recharge", "add_money", "visa", "mastercard"],
    "transactional_features__transfers_bills":    ["npsb", "rtgs", "beftn", "bill_payment", "biller"],
    "transactional_features__customization":      ["cheque", "home_page", "widget", "customization"],
    "transactional_features__fees":               ["rtgs_fee", "npsb_free", "charges"],
    "transactional_features__security_actions":   ["mpin", "biometric", "transaction_history", "statement", "receipt"],

    # transaction_account
    "transaction_account__eligibility_opening":   ["eligibility", "nid", "onboarding", "kyc"],
    "transaction_account__features_benefits":     ["interest", "mastercard", "qr", "nominee", "digital_banking"],
    "transaction_account__security_regulations":  ["dormant", "inactive", "2fa", "sms_fee", "bangladesh_bank"],

    # fd
    "fixed_deposit__overview_eligibility":  ["fd", "term_deposit", "tenure", "minimum_deposit"],
    "fixed_deposit__interest_maturity":     ["interest", "premature_closure", "cumulative", "renewal", "maturity"],
    "fixed_deposit__nominees_regulations":  ["nominee", "tax", "psr", "digital_certificate"],

    # goal_dps
    "goal_based_dps__overview_setup":           ["goal_dps", "recurring_deposit", "installment", "minimum_goal"],
    "goal_based_dps__interest_maturity":        ["interest", "maturity", "missed_installment"],
    "goal_based_dps__early_closure_regulations":["premature_closure", "tax", "psr", "bangladesh_bank"],

    # flexi_dps
    "flexi_dps__overview_eligibility":           ["flexi_dps", "goal_savings", "tenure", "bdt"],
    "flexi_dps__contributions_nominees":         ["flexible_deposit", "no_fixed_date", "nominee", "multiple_accounts"],
    "flexi_dps__interest_maturity":              ["interest", "slab_rate", "maturity", "linked_account"],
    "flexi_dps__early_withdrawal_regulations":   ["premature_closure", "partial_closure", "tax", "psr"],

    # support
    "customer_service__outbound_calls":      ["outbound_call", "phone_number", "09644213030", "09610961000"],
    "customer_service__security_verification":["otp", "pin", "fraud", "verification", "security"],
    "customer_service__support_services":    ["app_support", "guidance", "card_delivery", "navigation"],
}

# ── Splitter configuration ─────────────────────────────────────────────────────
CHILD_CHUNK_SIZE    = 500   # characters (~100-120 tokens); prevents Q/A cleavage
CHILD_CHUNK_OVERLAP = 80    # characters (~1 sentence overlap)
# Separator priority: prefer breaking between Q&A pairs, then paragraphs,
# then lines, then sentence boundaries. "Question: " ensures the splitter
# breaks BEFORE a new question rather than mid-answer.
CHILD_SEPARATORS    = ["\n\nQuestion: ", "\n\n", "\n", ". "]

# ── ChromaDB configuration ────────────────────────────────────────────────────
CHROMA_PERSIST_PATH = Path(__file__).resolve().parents[3] / "chroma_db"
CHROMA_COLLECTION_NAME = "qurio_knowledge_v2"

# ── Embedding model ───────────────────────────────────────────────────────────
# Using the local vLLM hosted model on port 8001
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# ── Cosine similarity thresholds ─────────────────────────────────────────────
# ChromaDB returns L2 distances in cosine space; convert: similarity = 1 - distance
#
# SIMILARITY_THRESHOLD_REJECT (0.35):
#   Below this: the query and chunk embed into completely different regions of
#   vector space. These are definitively off-topic or noise. Hard discard.
#   Off-topic queries (e.g. weather, sports) typically score 0.10-0.28.
#
# SIMILARITY_THRESHOLD_WARN (0.55):
#   0.35-0.55: Some lexical or semantic overlap, but not confidently relevant.
#   Retrieved and included in context, but flagged as confidence="low" in logs.
#   Terminal output is yellow-coded; log files record confidence="low".
#   Frontend users never see confidence labels.
#
# Above 0.55: High-confidence retrieval. confidence="high". Terminal is green.
#
# Calibration: at 71 chunks with bge-small-en-v1.5, paraphrased on-topic
# queries typically score 0.62-0.85. Tune once production query logs are available.
SIMILARITY_THRESHOLD_REJECT = 0.35
SIMILARITY_THRESHOLD_WARN   = 0.55

# Number of candidate chunks to retrieve before threshold filtering
RETRIEVAL_TOP_K = 3
