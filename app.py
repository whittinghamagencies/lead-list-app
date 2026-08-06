"""
Lead List Prep - internal prospecting pipeline
==============================================

Four stages, run in order. Each one narrows the list:

    1. Size      - drop companies below a headcount threshold
    2. Address   - verify mailing addresses, keep commercial ones
    3. Contacts  - confirm / correct / flag the decision maker
                     green  = same person, still there
                     yellow = corrected with new data from Apollo
                     red    = nobody found, set aside
    4. Score     - rank the remaining leads, red ones excluded

Run with:  streamlit run app.py
Requires:  streamlit, pandas, requests, smartystreets_python_sdk
"""

import json

import pandas as pd
import streamlit as st

try:
    from smartystreets_python_sdk import ClientBuilder, StaticCredentials, Batch
    from smartystreets_python_sdk import us_street
    SMARTY_AVAILABLE = True
except ImportError:
    SMARTY_AVAILABLE = False

import requests


# ===========================================================================
# Configuration
# ===========================================================================

APP_TITLE = "Lead List Prep"

KEEP_COLUMNS = [
    "Company Name",
    "Mailing Address",
    "Mailing City",
    "Mailing State",
    "Mailing Zip Code",
    "Location Employee Size Range",
    "Executive First Name",
    "Executive Last Name",
    "Executive Title",
    "Phone Number Combined",
    "Primary SIC Code",
    "Primary SIC Code Description",
    "Infogroup ID",
]

SIZE_COLUMN = "Location Employee Size Range"
STREET_COLUMN = "Mailing Address"
CITY_COLUMN = "Mailing City"
STATE_COLUMN = "Mailing State"
ZIP_COLUMN = "Mailing Zip Code"

DEFAULT_MIN_EMPLOYEES = 5
MAX_PREVIEW_ROWS = 500
SMARTY_BATCH_SIZE = 100

# --- Industry tiers, by SIC major group (the first two digits) ---------------
#
# Blue collar: hourly W-2 crews, the workforce worksite enrollment is built
# for. White collar professional practices are excluded - a law office or a
# medical practice is a handful of salaried professionals plus admin staff,
# which doesn't support an on-site enrollment.

TIER_PRIORITY = "Priority"
TIER_NEUTRAL = "Neutral"
TIER_EXCLUDED = "Excluded"

TIER_ORDER = {TIER_PRIORITY: 0, TIER_NEUTRAL: 1, TIER_EXCLUDED: 2}

TIER_COLORS = {
    TIER_PRIORITY: "#d4f4dd",
    TIER_NEUTRAL: "#e6e8eb",
    TIER_EXCLUDED: "#fbd5d5",
}

# Major group -> label. Anything not listed lands in Neutral.
SIC_PRIORITY_GROUPS = {
    "07": "Agricultural services", "08": "Forestry",
    "10": "Metal mining", "12": "Coal mining", "13": "Oil and gas", "14": "Quarrying",
    "15": "Building construction", "16": "Heavy construction", "17": "Special trade contractors",
    "20": "Food manufacturing", "21": "Tobacco", "22": "Textile mills", "23": "Apparel",
    "24": "Lumber and wood", "25": "Furniture", "26": "Paper", "27": "Printing",
    "28": "Chemicals", "29": "Petroleum refining", "30": "Rubber and plastics",
    "31": "Leather", "32": "Stone, clay, glass", "33": "Primary metals",
    "34": "Fabricated metals", "35": "Industrial machinery", "36": "Electronics",
    "37": "Transportation equipment", "38": "Instruments", "39": "Misc manufacturing",
    "41": "Transit and ground transport", "42": "Trucking and warehousing",
    "44": "Water transport", "45": "Air transport", "46": "Pipelines",
    "47": "Transportation services", "48": "Communications", "49": "Utilities",
    "50": "Wholesale - durable goods", "51": "Wholesale - nondurable goods",
    "75": "Auto repair and services", "76": "Misc repair services",
}

# Full major-group labels, so the exclude picker can show readable names for
# whatever happens to be in a given file.
SIC_GROUP_LABELS = {
    "01": "Agricultural production - crops", "02": "Agricultural production - livestock",
    "07": "Agricultural services", "08": "Forestry", "09": "Fishing and hunting",
    "10": "Metal mining", "12": "Coal mining", "13": "Oil and gas", "14": "Quarrying",
    "15": "Building construction", "16": "Heavy construction", "17": "Special trade contractors",
    "20": "Food manufacturing", "21": "Tobacco", "22": "Textile mills", "23": "Apparel",
    "24": "Lumber and wood", "25": "Furniture", "26": "Paper", "27": "Printing",
    "28": "Chemicals", "29": "Petroleum refining", "30": "Rubber and plastics",
    "31": "Leather", "32": "Stone, clay, glass", "33": "Primary metals",
    "34": "Fabricated metals", "35": "Industrial machinery", "36": "Electronics",
    "37": "Transportation equipment", "38": "Instruments", "39": "Misc manufacturing",
    "40": "Railroads", "41": "Transit and ground transport", "42": "Trucking and warehousing",
    "44": "Water transport", "45": "Air transport", "46": "Pipelines",
    "47": "Transportation services", "48": "Communications", "49": "Utilities",
    "50": "Wholesale - durable goods", "51": "Wholesale - nondurable goods",
    "52": "Building materials and garden retail", "53": "General merchandise stores",
    "54": "Food stores", "55": "Auto dealers and gas stations", "56": "Apparel stores",
    "57": "Furniture and home furnishings stores", "58": "Restaurants and bars",
    "59": "Misc retail",
    "60": "Banks", "61": "Nondepository credit", "62": "Security and commodity brokers",
    "63": "Insurance carriers", "64": "Insurance agents", "65": "Real estate",
    "67": "Holding and investment offices",
    "70": "Hotels and lodging", "72": "Personal services", "73": "Business services",
    "75": "Auto repair and services", "76": "Misc repair services",
    "78": "Motion pictures", "79": "Amusement and recreation",
    "80": "Health services - doctors, dentists, clinics", "81": "Legal services",
    "82": "Educational services", "83": "Social services", "84": "Museums",
    "86": "Membership organizations",
    "87": "Engineering, accounting, research, consulting", "88": "Private households",
    "89": "Misc services",
    "91": "Executive and legislative government", "92": "Justice and public safety",
    "93": "Public finance", "94": "Human resources administration",
    "95": "Environmental programs", "96": "Economic programs", "97": "National security",
}

# Excluded by default: only the two you named. Everything else is kept so the
# net stays wide - add groups from the picker as you spot them in real files.
DEFAULT_EXCLUDED_GROUPS = ("80", "81")

# Keyword fallback for rows with no usable SIC code. Deliberately short.
PRIORITY_KEYWORDS = (
    "construction", "contractor", "manufactur", "warehouse", "trucking",
    "freight", "logistics", "fabricat", "welding", "machine shop", "roofing",
    "plumbing", "electrical", "hvac", "excavat", "concrete", "drywall",
    "landscap", "distribut", "wholesale", "repair", "maintenance", "plant",
    "mill", "foundry", "assembly", "printing", "packaging", "auto body",
)

DEFAULT_EXCLUDED_KEYWORDS = (
    "physician", "doctor", "dentist", "attorney", "lawyer", "law office",
)


# Titles that can approve a benefits decision at a small or mid-size employer.
DM_TITLES = (
    "owner, president, CEO, general manager, HR director, HR manager, "
    "office manager, controller, CFO, vice president"
)

# Stage 3 statuses and their colors.
STATUS_CONFIRMED = "Verified"      # our listed contact was found, still there
STATUS_CORRECTED = "Updated"       # a different decision maker was found
STATUS_UNVERIFIED = "Unverified"   # company found, no personnel info published
STATUS_MISSING = "Missing"         # no company presence and no name to work with

STATUS_COLORS = {
    STATUS_CONFIRMED: "#d4f4dd",   # green
    STATUS_CORRECTED: "#fdf3c8",   # yellow
    STATUS_UNVERIFIED: "#e6e8eb",  # grey
    STATUS_MISSING: "#fbd5d5",     # red
}

# Statuses that carry a usable name into scoring.
ACTIONABLE_STATUSES = (STATUS_CONFIRMED, STATUS_CORRECTED)

XAI_URL = "https://api.x.ai/v1/chat/completions"
XAI_RESPONSES_URL = "https://api.x.ai/v1/responses"   # agentic endpoint, has web_search
XAI_MODEL = "grok-4.5"

# Contact lookups are the expensive stage, so they are batched: several
# companies per agentic call rather than one call each.
# --- Cost model -------------------------------------------------------------
# Published xAI rates for grok-4.5 as of August 2026. These change, so they're
# editable in the UI rather than buried here.
DEFAULT_RATES = {
    "input_per_million": 2.00,
    "cached_input_per_million": 0.30,
    "output_per_million": 6.00,
    "per_thousand_searches": 5.00,
}

CONTACT_BATCH_SIZE = 4
CONTACT_MAX_TOKENS = 600      # replies are compact JSON, so cap them hard
SCORING_BATCH_SIZE = 15

DEFAULT_SCORING_CRITERIA = """We sell voluntary worksite benefits to employers,
enrolling their employees on site. A strong lead has enough employees to make an
on-site enrollment worth the trip, a stable W-2 workforce rather than seasonal or
contract labor, and a contact senior enough to approve a benefits decision.

Weaker leads: very small headcount, branches that cannot decide locally,
industries with mostly part-time or transient staff."""


# ===========================================================================
# Secrets - never hardcode keys
# ===========================================================================

def get_secret(name: str, default: str = "") -> str:
    """Read st.secrets without crashing when the key isn't set."""
    try:
        return str(st.secrets[name]).strip()  # a pasted newline causes a 401
    except Exception:
        return default


SMARTY_AUTH_ID = get_secret("SMARTY_AUTH_ID")
SMARTY_AUTH_TOKEN = get_secret("SMARTY_AUTH_TOKEN")
XAI_API_KEY = get_secret("XAI_API_KEY")


# ===========================================================================
# Loading
# ===========================================================================

def load_csv(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV, or raise a readable error."""
    if not uploaded_file.name.lower().endswith(".csv"):
        raise ValueError("That file isn't a CSV. Export the list as .csv and upload again.")

    try:
        df = pd.read_csv(uploaded_file, dtype=str)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, dtype=str, encoding="latin-1")
    except pd.errors.EmptyDataError:
        raise ValueError("That CSV is empty - no columns or rows to read.")
    except pd.errors.ParserError:
        raise ValueError("That CSV couldn't be parsed. Check for stray commas or quotes.")

    if df.empty:
        raise ValueError("That CSV has headers but no rows.")

    return df


def filter_columns(df: pd.DataFrame):
    """Keep only KEEP_COLUMNS, matching headers case-insensitively."""
    normalized = {str(col).strip().lower(): col for col in df.columns}

    found, missing, actual = [], [], []
    for wanted in KEEP_COLUMNS:
        match = normalized.get(wanted.lower())
        if match is not None:
            found.append(wanted)
            actual.append(match)
        else:
            missing.append(wanted)

    filtered = df[actual].copy()
    filtered.columns = found
    return filtered, found, missing


# ===========================================================================
# Stage 1 - Employee size
# ===========================================================================

def parse_min_employees(size_range):
    """Lower bound of a size band: "1 to 4" -> 1, "1,000 to 4,999" -> 1000."""
    if not isinstance(size_range, str):
        return None

    digits = ""
    for char in size_range.replace(",", ""):
        if char.isdigit():
            digits += char
        elif digits:
            break

    return int(digits) if digits else None


def filter_by_size(df: pd.DataFrame, minimum: int, include_unknown: bool):
    """
    Drop companies below the threshold and sort largest first.

    Sorting by size means the later stages - which are capped - spend their
    budget on the biggest employers rather than whatever happened to be at
    the top of the export.
    """
    if SIZE_COLUMN not in df.columns:
        return df, 0

    bounds = df[SIZE_COLUMN].map(parse_min_employees)

    keep = bounds.notna() & (bounds >= minimum)
    if include_unknown:
        keep = keep | bounds.isna()

    result = df[keep].copy()
    result["_size_sort"] = bounds[keep]
    result = result.sort_values("_size_sort", ascending=False, na_position="last")
    result = result.drop(columns=["_size_sort"])

    return result, int((~keep).sum())


# ===========================================================================
# Stage 2 - Address verification (Smarty)
# ===========================================================================

RECORD_TYPES = {
    "F": "Firm", "G": "General Delivery", "H": "Highrise",
    "P": "PO Box", "R": "Rural Route", "S": "Street",
}


def blank_address() -> dict:
    return {
        "Address Status": "Not found",
        "Verified Address": "",
        "Verified City": "",
        "Verified State": "",
        "Verified ZIP+4": "",
        "Property Type": "",
        "Record Type": "",
        "County": "",
    }


def candidate_to_address(candidate, original_street: str) -> dict:
    """Turn one Smarty candidate into our address columns."""
    components, metadata = candidate.components, candidate.metadata

    zip_plus_four = components.zipcode or ""
    if components.plus4_code:
        zip_plus_four = f"{zip_plus_four}-{components.plus4_code}"

    delivered = candidate.delivery_line_1 or ""
    status = "Verified"
    if delivered.strip().lower() != str(original_street or "").strip().lower():
        status = "Corrected"

    return {
        "Address Status": status,
        "Verified Address": delivered,
        "Verified City": components.city_name or "",
        "Verified State": components.state_abbreviation or "",
        "Verified ZIP+4": zip_plus_four,
        # rdi is USPS delivery classification: Residential or Commercial.
        "Property Type": metadata.rdi or "Unknown",
        "Record Type": RECORD_TYPES.get(metadata.record_type, metadata.record_type or ""),
        "County": metadata.county_name or "",
    }


def verify_addresses(df: pd.DataFrame, auth_id: str, auth_token: str, progress=None) -> pd.DataFrame:
    """Send mailing addresses to Smarty in batches of 100 and append results."""
    client = ClientBuilder(StaticCredentials(auth_id, auth_token)).build_us_street_api_client()

    results = {index: blank_address() for index in df.index}

    sendable = [
        index for index in df.index
        if str(df.at[index, STREET_COLUMN] or "").strip() not in ("", "None", "nan")
    ]

    for start in range(0, len(sendable), SMARTY_BATCH_SIZE):
        chunk = sendable[start:start + SMARTY_BATCH_SIZE]

        batch = Batch()
        for index in chunk:
            lookup = us_street.Lookup()
            lookup.input_id = str(index)
            lookup.street = str(df.at[index, STREET_COLUMN] or "")
            lookup.city = str(df.at[index, CITY_COLUMN] or "")
            lookup.state = str(df.at[index, STATE_COLUMN] or "")
            lookup.zipcode = str(df.at[index, ZIP_COLUMN] or "")
            lookup.candidates = 1
            batch.add(lookup)

        client.send_batch(batch)

        for lookup in batch.all_lookups:
            index = int(lookup.input_id)
            if lookup.result:
                results[index] = candidate_to_address(
                    lookup.result[0], df.at[index, STREET_COLUMN]
                )

        if progress is not None:
            progress.progress(min((start + SMARTY_BATCH_SIZE) / len(sendable), 1.0))

    return df.join(pd.DataFrame.from_dict(results, orient="index"))


def keep_commercial(df: pd.DataFrame, drop_po_boxes: bool = True) -> pd.DataFrame:
    """
    Narrow to addresses a setter could actually visit.

    Keeps anything not explicitly flagged Residential - an unmatched address
    is unknown, not residential, and dropping it would lose real prospects.
    """
    result = df[df["Property Type"] != "Residential"].copy()
    if drop_po_boxes:
        result = result[~result["Record Type"].isin(["PO Box", "General Delivery"])].copy()
    return result


# ===========================================================================
# Stage 3 - Industry fit (SIC code)
# ===========================================================================

def sic_major_group(sic_code) -> str:
    """
    First two digits of a SIC code, which is the major industry group.

    Handles the ways these arrive in exports: "1731", "1731.0", " 1731",
    and short codes like "17" that need no padding.
    """
    if sic_code is None:
        return ""

    digits = "".join(char for char in str(sic_code) if char.isdigit())
    if not digits:
        return ""

    return digits[:2]


def classify_industry(sic_code, description, excluded_groups, excluded_keywords) -> tuple:
    """
    Sort a lead into Priority, Neutral, or Excluded.

    Nothing is excluded unless it matches a group or keyword you chose, so the
    default behaviour is to keep the row. Priority is only a sort hint - a
    Neutral row still carries forward.

    Returns (tier, reason).
    """
    group = sic_major_group(sic_code)
    text = str(description or "").lower()

    # Your exclusions come first, so a group you've added always wins.
    if group and group in excluded_groups:
        return TIER_EXCLUDED, SIC_GROUP_LABELS.get(group, f"SIC group {group}")

    for keyword in excluded_keywords:
        if keyword and keyword in text:
            return TIER_EXCLUDED, f"matched '{keyword}'"

    if group in SIC_PRIORITY_GROUPS:
        return TIER_PRIORITY, SIC_PRIORITY_GROUPS[group]

    for keyword in PRIORITY_KEYWORDS:
        if keyword in text:
            return TIER_PRIORITY, f"matched '{keyword}'"

    if group:
        return TIER_NEUTRAL, SIC_GROUP_LABELS.get(group, f"SIC group {group}")
    return TIER_NEUTRAL, "no SIC code"


def group_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Every SIC major group present in this file, with a label and a count.

    This is what the exclude picker is built from - you curate against what
    you actually have rather than a generic list.
    """
    if "Primary SIC Code" not in df.columns:
        return pd.DataFrame(columns=["Group", "Industry", "Companies"])

    groups = df["Primary SIC Code"].map(sic_major_group)
    counts = groups[groups != ""].value_counts()

    return pd.DataFrame([
        {"Group": group, "Industry": SIC_GROUP_LABELS.get(group, "Unlabelled"),
         "Companies": int(count)}
        for group, count in counts.items()
    ])


def apply_industry_tiers(df: pd.DataFrame, excluded_groups=(),
                         excluded_keywords=()) -> pd.DataFrame:
    """
    Add Industry Tier and Industry Basis columns, then sort priority first
    and largest employer first inside each tier.

    Sorting here matters because the decision maker stage that follows is
    capped - this puts the big blue collar employers at the top of the queue.
    """
    if "Primary SIC Code" not in df.columns and "Primary SIC Code Description" not in df.columns:
        return df

    tiers, reasons = [], []
    for _, row in df.iterrows():
        tier, reason = classify_industry(
            row.get("Primary SIC Code"), row.get("Primary SIC Code Description"),
            excluded_groups, excluded_keywords,
        )
        tiers.append(tier)
        reasons.append(reason)

    result = df.copy()
    result["Industry Tier"] = tiers
    result["Industry Basis"] = reasons

    result["_tier_sort"] = result["Industry Tier"].map(TIER_ORDER)
    result["_size_sort"] = result[SIZE_COLUMN].map(parse_min_employees) \
        if SIZE_COLUMN in result.columns else 0
    result = result.sort_values(
        ["_tier_sort", "_size_sort"], ascending=[True, False], na_position="last"
    ).drop(columns=["_tier_sort", "_size_sort"])

    return result


# ===========================================================================
# Stage 4 - Decision maker resolution (Grok with web search)
# ===========================================================================
#
# Token strategy:
#   - Several companies per call, so one system prompt covers a batch
#     instead of being repaid for every company.
#   - Compact single-letter status codes and short field names in the reply.
#   - A hard max_tokens ceiling, since the reply is small JSON either way.
#   - Results cached per company name, so duplicates and reruns cost nothing.
#   - A search order in the prompt, so the model finds the answer on the first
#     or second search rather than wandering.

CONTACT_SYSTEM_PROMPT = (
    "You verify business decision makers using web search. "
    "Search each company separately. Prefer, in order: (1) the company's own "
    "website - about, team, staff, contact, or leadership pages; "
    "(2) state business filings and Secretary of State registries, which list "
    "owners, officers, and registered agents for small businesses; "
    "(3) local chamber of commerce and licensing board listings; "
    "(4) LinkedIn or business directories. "
    "Use the phone number and address given to confirm you have the right "
    "business, not a same-named company elsewhere. "
    "Never guess a name from the company name. Never invent a person. "
    "Reply with a JSON array only - no prose, no markdown fences. "
    'Each element: {"i":<id>,"s":"<V|C|U|M>","n":"<First Last or empty>",'
    '"t":"<title or empty>","u":"<source URL or empty>"}. '
    "Status codes: "
    "V = the person named in the record was found currently at this company. "
    "C = that person was not found, but a different current decision maker was, "
    "so you are correcting the record - put the new name in n. "
    "U = the business exists online but publishes no personnel information. "
    "M = no evidence the business exists online at all. "
    "Use U, never C or M, when you simply could not find personnel details. "
    "Absence of information is never proof someone left."
)

# Maps the model's single letters back to our status names.
STATUS_CODES = {
    "V": STATUS_CONFIRMED,
    "C": STATUS_CORRECTED,
    "U": STATUS_UNVERIFIED,
    "M": STATUS_MISSING,
}


def contact_prompt_line(index, row) -> str:
    """
    One compact line describing a company to check.

    Includes phone and address because they disambiguate a local business far
    better than the name alone, and cost only a few tokens.
    """
    first = str(row.get("Executive First Name", "") or "").strip()
    last = str(row.get("Executive Last Name", "") or "").strip()
    title = str(row.get("Executive Title", "") or "").strip()
    listed = f"{first} {last}".strip()

    parts = [
        f"id={index}",
        str(row.get("Company Name", "") or ""),
        f"{row.get('Verified Address') or row.get(STREET_COLUMN) or ''}",
        f"{row.get(CITY_COLUMN, '')} {row.get(STATE_COLUMN, '')}".strip(),
        str(row.get("Phone Number Combined", "") or ""),
    ]

    if listed:
        parts.append(f"record says: {listed}" + (f", {title}" if title else ""))
    else:
        parts.append(f"record has no name - find a {DM_TITLES.split(',')[0]} or similar")

    return " | ".join(part for part in parts if part)


def extract_usage(payload: dict) -> dict:
    """
    Pull token and tool counts out of a response.

    Field names differ between endpoints and have changed over time, so we
    check the plausible shapes rather than assuming one. Anything missing
    comes back as zero, which understates rather than overstates the bill -
    so treat the total as a floor, not a guarantee.
    """
    usage = payload.get("usage") or {}

    def first_of(*keys):
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                return int(value)
        return 0

    cached = first_of("cached_input_tokens", "cache_read_input_tokens")
    total_input = first_of("input_tokens", "prompt_tokens")

    # Count server-side tool calls wherever they're reported.
    searches = 0
    for container in (payload.get("server_side_tool_usage"), usage.get("server_side_tool_usage")):
        if isinstance(container, dict):
            for key, value in container.items():
                if "search" in key.lower() and isinstance(value, (int, float)):
                    searches += int(value)
    if not searches:
        searches = first_of("num_sources_used", "web_search_calls")

    return {
        "input_tokens": max(total_input - cached, 0),
        "cached_tokens": cached,
        "output_tokens": first_of("output_tokens", "completion_tokens"),
        "searches": searches,
        "searches_reported": bool(searches),
    }


def usage_cost(usage: dict, rates: dict) -> float:
    """Dollar cost of one call's usage at the given rates."""
    return (
        usage["input_tokens"] / 1_000_000 * rates["input_per_million"]
        + usage["cached_tokens"] / 1_000_000 * rates["cached_input_per_million"]
        + usage["output_tokens"] / 1_000_000 * rates["output_per_million"]
        + usage["searches"] / 1_000 * rates["per_thousand_searches"]
    )


def call_grok_search(prompt: str, api_key: str):
    """
    One agentic call with web search enabled.

    Uses the /v1/responses endpoint because the older search_parameters API
    was retired in January 2026 and now returns 410.

    Returns (text, usage) so the caller can price the run.
    """
    response = requests.post(
        XAI_RESPONSES_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": XAI_MODEL,
            "input": [
                {"role": "system", "content": CONTACT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "tools": [{"type": "web_search"}],
            "max_output_tokens": CONTACT_MAX_TOKENS,
        },
        timeout=240,
    )
    response.raise_for_status()
    payload = response.json()

    # output is a list of items - tool calls, reasoning, messages - so collect
    # text from whichever items carry it rather than assuming a position.
    chunks = []
    for item in payload.get("output", []):
        for part in item.get("content", []) or []:
            if part.get("text"):
                chunks.append(part["text"])

    return "\n".join(chunks).strip(), extract_usage(payload)


def parse_contact_reply(raw: str) -> dict:
    """Turn the compact JSON reply into {row_id: fields}."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]

    # Tolerate a stray sentence before or after the array.
    if "[" in cleaned and "]" in cleaned:
        cleaned = cleaned[cleaned.index("["):cleaned.rindex("]") + 1]

    parsed = {}
    for item in json.loads(cleaned):
        try:
            parsed[int(item["i"])] = {
                "status": STATUS_CODES.get(str(item.get("s", "")).upper()[:1], STATUS_UNVERIFIED),
                "name": str(item.get("n", "")).strip(),
                "title": str(item.get("t", "")).strip(),
                "url": str(item.get("u", "")).strip(),
            }
        except (KeyError, ValueError, TypeError):
            continue
    return parsed


def blank_contact() -> dict:
    return {
        "Contact Status": "",
        "Current First Name": "",
        "Current Last Name": "",
        "Current Title": "",
        "Contact Source": "",
        "Contact Note": "",
    }


def split_name(full_name: str):
    """Split a returned full name into first and last."""
    parts = [part for part in full_name.replace(",", " ").split() if part]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[-1]


def apply_contact_result(row, result: dict) -> dict:
    """
    Turn one parsed result into our contact columns.

    Guards against the model returning status C with no name, which would
    otherwise produce a yellow row with nothing to call.
    """
    contact = blank_contact()
    status = result.get("status", STATUS_UNVERIFIED)
    first, last = split_name(result.get("name", ""))

    our_first = str(row.get("Executive First Name", "") or "").strip()
    our_last = str(row.get("Executive Last Name", "") or "").strip()
    had_name = bool(our_first or our_last)

    # A correction with no name isn't a correction.
    if status == STATUS_CORRECTED and not (first or last):
        status = STATUS_UNVERIFIED

    # Verified means our person - keep ours if the model echoed nothing.
    if status == STATUS_CONFIRMED and not (first or last):
        first, last = our_first, our_last

    # No name anywhere, and nothing found: that's a set-aside, not unverified.
    if status == STATUS_UNVERIFIED and not had_name:
        status = STATUS_MISSING

    contact.update({
        "Contact Status": status,
        "Current First Name": first,
        "Current Last Name": last,
        "Current Title": result.get("title", ""),
        "Contact Source": result.get("url", ""),
    })

    if status == STATUS_CONFIRMED:
        contact["Contact Note"] = "Listed contact confirmed"
    elif status == STATUS_CORRECTED:
        contact["Contact Note"] = (
            f"Replaces {our_first} {our_last}".strip() if had_name
            else "New contact - record had no name"
        )
    elif status == STATUS_UNVERIFIED:
        contact["Contact Note"] = "Business found, no personnel published"
    else:
        contact["Contact Note"] = "No web presence found"

    return contact


def resolve_contacts(df: pd.DataFrame, api_key: str, limit: int, rates: dict, progress=None):
    """
    Resolve decision makers for the first `limit` rows, cheapest path first.

    Companies are batched CONTACT_BATCH_SIZE per call, and every result is
    cached by company name so a rerun or a duplicate company costs nothing.
    Returns (dataframe, diagnostics).
    """
    results = {index: blank_contact() for index in df.index}
    diagnostics = {
        "calls": 0, "cached": 0, "checked": 0, "errors": [],
        "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0,
        "searches": 0, "searches_reported": True, "cost": 0.0,
    }

    cache = st.session_state.setdefault("grok_contact_cache", {})
    targets = list(df.index)[:limit]

    # Reuse anything already resolved for the same company name.
    pending = []
    for index in targets:
        company = str(df.at[index, "Company Name"] or "").strip()
        if not company:
            results[index]["Contact Status"] = STATUS_MISSING
            results[index]["Contact Note"] = "No company name"
            continue
        if company in cache:
            results[index] = apply_contact_result(df.loc[index], cache[company])
            diagnostics["cached"] += 1
        else:
            pending.append(index)

    for start in range(0, len(pending), CONTACT_BATCH_SIZE):
        chunk = pending[start:start + CONTACT_BATCH_SIZE]
        prompt = "Check each business:\n" + "\n".join(
            contact_prompt_line(index, df.loc[index]) for index in chunk
        )

        try:
            reply, usage = call_grok_search(prompt, api_key)
            parsed = parse_contact_reply(reply)
            diagnostics["calls"] += 1

            # Track spend as we go, so a cancelled run still reports its cost.
            for field in ("input_tokens", "cached_tokens", "output_tokens", "searches"):
                diagnostics[field] += usage[field]
            if not usage["searches_reported"]:
                diagnostics["searches_reported"] = False
            diagnostics["cost"] += usage_cost(usage, rates)
        except Exception as err:
            parsed = {}
            if len(diagnostics["errors"]) < 3:
                diagnostics["errors"].append(str(err)[:120])

        for index in chunk:
            result = parsed.get(index)
            if result is None:
                results[index]["Contact Status"] = STATUS_UNVERIFIED
                results[index]["Contact Note"] = "No result returned - rerun to retry"
                continue

            results[index] = apply_contact_result(df.loc[index], result)
            cache[str(df.at[index, "Company Name"] or "").strip()] = result
            diagnostics["checked"] += 1

        if progress is not None:
            progress.progress(min((start + CONTACT_BATCH_SIZE) / max(len(pending), 1), 1.0))

    # Running total for the session, so the cost of several passes is visible.
    st.session_state["xai_spend"] = st.session_state.get("xai_spend", 0.0) + diagnostics["cost"]
    st.session_state["xai_calls"] = st.session_state.get("xai_calls", 0) + diagnostics["calls"]

    return df.join(pd.DataFrame.from_dict(results, orient="index")), diagnostics



def color_status(df: pd.DataFrame):
    """Shade the Contact Status and Industry Tier columns."""
    def shade(column, palette):
        return [
            f"background-color: {palette.get(value, '')}" if value in palette else ""
            for value in column
        ]

    styler = df.style
    if "Contact Status" in df.columns:
        styler = styler.apply(shade, palette=STATUS_COLORS, subset=["Contact Status"])
    if "Industry Tier" in df.columns:
        styler = styler.apply(shade, palette=TIER_COLORS, subset=["Industry Tier"])
    return styler


# ===========================================================================
# Stage 4 - Scoring (xAI)
# ===========================================================================

def lead_to_summary(row) -> str:
    """Condense one lead into the fields that matter for judging fit."""
    contact = f"{row.get('Current First Name', '')} {row.get('Current Last Name', '')}".strip()
    title = row.get("Current Title") or row.get("Executive Title") or "no title"

    return (
        f"{row.get('Company Name', '')} | {row.get(SIZE_COLUMN, 'unknown')} employees | "
        f"{row.get('Primary SIC Code Description', 'unknown industry')} | "
        f"contact: {contact or 'unnamed'}, {title} | "
        f"{row.get('Mailing City', '')}, {row.get('Mailing State', '')}"
    )


def parse_scores(raw: str) -> dict:
    """Turn the model's JSON reply into {row_id: (score, reason)}."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]

    parsed = {}
    for item in json.loads(cleaned):
        try:
            parsed[int(item["id"])] = (int(item["score"]), str(item.get("reason", "")))
        except (KeyError, ValueError, TypeError):
            continue
    return parsed


def score_leads(df: pd.DataFrame, api_key: str, criteria: str, progress=None) -> pd.DataFrame:
    """Score each lead 1-10 for fit, in batches, and append score plus reason."""
    scores = {index: (None, "") for index in df.index}
    indexes = list(df.index)

    for start in range(0, len(indexes), SCORING_BATCH_SIZE):
        chunk = indexes[start:start + SCORING_BATCH_SIZE]
        lines = [f"{index}: {lead_to_summary(df.loc[index])}" for index in chunk]

        try:
            response = requests.post(
                XAI_URL,
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={
                    "model": XAI_MODEL,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content":
                            "You score B2B sales leads. Reply with a JSON array only - no "
                            'prose, no fences. Each element: {"id": <number>, '
                            '"score": <1-10>, "reason": "<12 words max>"}.'},
                        {"role": "user", "content":
                            f"Scoring criteria:\\n{criteria}\\n\\nScore each lead 1 (poor) "
                            f"to 10 (excellent). Use the id at the start of each line.\\n\\n"
                            + "\\n".join(lines)},
                    ],
                },
                timeout=120,
            )
            response.raise_for_status()
            scores.update(parse_scores(response.json()["choices"][0]["message"]["content"]))
        except Exception:
            pass  # leave this batch unscored; the UI reports the gap

        if progress is not None:
            progress.progress(min((start + SCORING_BATCH_SIZE) / len(indexes), 1.0))

    scored = pd.DataFrame(
        [{"Fit Score": scores[i][0], "Score Reason": scores[i][1]} for i in df.index],
        index=df.index,
    )
    return df.join(scored).sort_values("Fit Score", ascending=False, na_position="last")


# ===========================================================================
# UI
# ===========================================================================

def download_row(df: pd.DataFrame, label: str, filename: str, primary: bool = False) -> None:
    """One download button, sized to the frame it's given."""
    st.download_button(
        f"{label} ({len(df):,})",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        type="primary" if primary else "secondary",
    )


def stage_one_size(column_df: pd.DataFrame):
    """Stage 1: headcount threshold."""
    st.subheader("1. Company size")

    if SIZE_COLUMN not in column_df.columns:
        st.info(f"No '{SIZE_COLUMN}' column, so size filtering is off.")
        return column_df

    left, right = st.columns([1, 2])
    minimum = left.number_input("Minimum employees", min_value=0,
                                value=DEFAULT_MIN_EMPLOYEES, step=1)
    include_unknown = right.checkbox("Keep companies with no size listed", value=False)

    result, removed = filter_by_size(column_df, minimum, include_unknown)

    a, b = st.columns(2)
    a.metric("Leads at or above threshold", f"{len(result):,}")
    b.metric("Dropped", f"{removed:,}")

    if result.empty:
        st.info("Nothing left. Try a lower minimum.")
        return result

    with st.expander("Size breakdown"):
        st.dataframe(
            result[SIZE_COLUMN].value_counts(dropna=False)
            .rename_axis("Employee size range").reset_index(name="Companies"),
            use_container_width=True, hide_index=True,
        )

    return result


def stage_two_address(working_df: pd.DataFrame, file_key: str) -> pd.DataFrame:
    """Stage 2: verify addresses, keep commercial."""
    st.subheader("2. Mailable commercial address")

    if not SMARTY_AVAILABLE:
        st.info("Smarty SDK not installed. Add smartystreets_python_sdk to requirements.txt.")
        return working_df
    if not SMARTY_AUTH_ID or not SMARTY_AUTH_TOKEN:
        st.info("No Smarty credentials. Add SMARTY_AUTH_ID and SMARTY_AUTH_TOKEN to Secrets.")
        return working_df
    if STREET_COLUMN not in working_df.columns:
        st.info(f"No '{STREET_COLUMN}' column, so there's nothing to verify.")
        return working_df

    has_street = int(
        working_df[STREET_COLUMN].fillna("").astype(str).str.strip().ne("").sum()
    )
    cache_key = f"addr::{file_key}::{len(working_df)}"

    st.caption(f"{has_street:,} rows have a street address. One Smarty lookup each.")

    if st.button(f"Verify {has_street:,} addresses", type="primary"):
        progress = st.progress(0.0)
        try:
            st.session_state[cache_key] = verify_addresses(
                working_df, SMARTY_AUTH_ID, SMARTY_AUTH_TOKEN, progress
            )
        except Exception as err:
            st.error(f"Verification failed: {err}")
            st.caption("401 means the key is wrong; 402 means the lookup balance is empty.")
            return working_df
        finally:
            progress.empty()

    if cache_key not in st.session_state:
        return working_df

    verified = st.session_state[cache_key]
    status = verified["Address Status"].value_counts()
    property_type = verified["Property Type"].value_counts()

    a, b, c, d = st.columns(4)
    a.metric("Verified", f"{status.get('Verified', 0):,}")
    b.metric("Corrected", f"{status.get('Corrected', 0):,}")
    c.metric("Commercial", f"{property_type.get('Commercial', 0):,}")
    d.metric("Residential", f"{property_type.get('Residential', 0):,}")

    drop_po = st.checkbox("Also drop PO boxes and general delivery", value=True)
    commercial = keep_commercial(verified, drop_po)

    st.caption(
        f"{len(commercial):,} leads carried forward. "
        f"Excludes {property_type.get('Residential', 0):,} residential"
        + (" and any PO boxes." if drop_po else ".")
    )
    download_row(commercial, "Download non-residential list", f"commercial_{file_key}")

    return commercial


def stage_three_industry(working_df: pd.DataFrame, file_key: str) -> pd.DataFrame:
    """
    Stage 3: prioritise blue collar work, exclude only what you choose.

    Deliberately permissive - the picker is built from the industries actually
    in this file, so exclusions get added as they turn up rather than guessed
    at up front.
    """
    st.subheader("3. Industry fit")

    if "Primary SIC Code" not in working_df.columns and \
            "Primary SIC Code Description" not in working_df.columns:
        st.info("No SIC code or description in this file, so industry sorting is off.")
        return working_df

    inventory = group_inventory(working_df)

    # Only offer defaults that exist in this file, so the picker stays short.
    present = list(inventory["Group"]) if not inventory.empty else []
    defaults = [group for group in DEFAULT_EXCLUDED_GROUPS if group in present]

    excluded_groups = st.multiselect(
        "Industries to exclude",
        options=present,
        default=defaults,
        format_func=lambda group: f"{group} - {SIC_GROUP_LABELS.get(group, 'Unlabelled')} "
                                  f"({int(inventory.loc[inventory['Group'] == group, 'Companies'].iloc[0])})",
        help="Only these are dropped. Everything else carries forward - add to "
             "this list as you spot industries that aren't worth working.",
    )

    keyword_text = st.text_input(
        "Also exclude if the description contains (comma separated)",
        value=", ".join(DEFAULT_EXCLUDED_KEYWORDS),
        help="Catches rows whose SIC code is missing or misfiled.",
    )
    excluded_keywords = [
        word.strip().lower() for word in keyword_text.split(",") if word.strip()
    ]

    tiered = apply_industry_tiers(working_df, tuple(excluded_groups), tuple(excluded_keywords))
    counts = tiered["Industry Tier"].value_counts()

    a, b, c = st.columns(3)
    a.metric("Priority", f"{counts.get(TIER_PRIORITY, 0):,}",
             help="Construction, manufacturing, trucking, warehousing, trades, utilities. "
                  "Sorted to the top.")
    b.metric("Kept", f"{counts.get(TIER_NEUTRAL, 0):,}",
             help="Everything else you didn't exclude - retail, hospitality, services.")
    c.metric("Excluded", f"{counts.get(TIER_EXCLUDED, 0):,}",
             help="Only what your picker and keywords matched.")

    result = tiered[tiered["Industry Tier"] != TIER_EXCLUDED].copy()
    dropped = tiered[tiered["Industry Tier"] == TIER_EXCLUDED]

    with st.expander("Industries in this file"):
        st.dataframe(inventory, use_container_width=True, hide_index=True)
        st.caption(
            "Use this to decide what to add to the exclude list. Priority is a "
            "sort order, not a filter - nothing is dropped for being Neutral."
        )

    if not dropped.empty:
        with st.expander(f"Review the {len(dropped):,} excluded companies"):
            st.dataframe(
                dropped[["Company Name", SIZE_COLUMN, "Primary SIC Code",
                         "Primary SIC Code Description", "Industry Basis"]],
                use_container_width=True, hide_index=True,
            )
            st.caption("If something belongs here, remove its group from the picker above.")

    st.caption(
        f"{len(result):,} of {len(tiered):,} leads carried forward, blue collar "
        "first and largest employers first within each tier."
    )
    download_row(result, "Download industry-sorted list", f"industry_{file_key}")

    return result


def stage_four_contacts(working_df: pd.DataFrame, file_key: str):
    """Stage 3: verify, update, or set aside the decision maker."""
    st.subheader("4. Decision maker")

    if not XAI_API_KEY:
        st.info("No xAI key. Add XAI_API_KEY to Secrets to enable contact lookups.")
        return working_df, None

    st.caption(
        "Searches company websites and state business filings for whoever can "
        "approve a benefits decision. Confirms the name on file, replaces it "
        "when it's stale, and fills one in when the record has none."
    )

    rates = dict(DEFAULT_RATES)
    with st.expander("API rates (edit if xAI's pricing changes)"):
        r1, r2, r3, r4 = st.columns(4)
        rates["input_per_million"] = r1.number_input(
            "$ / 1M input", value=DEFAULT_RATES["input_per_million"], step=0.25, format="%.2f")
        rates["cached_input_per_million"] = r2.number_input(
            "$ / 1M cached", value=DEFAULT_RATES["cached_input_per_million"], step=0.10, format="%.2f")
        rates["output_per_million"] = r3.number_input(
            "$ / 1M output", value=DEFAULT_RATES["output_per_million"], step=0.25, format="%.2f")
        rates["per_thousand_searches"] = r4.number_input(
            "$ / 1k searches", value=DEFAULT_RATES["per_thousand_searches"], step=0.50, format="%.2f")
        st.caption(
            "Defaults are xAI's published grok-4.5 rates as of August 2026. "
            "Search calls usually dominate the bill, not tokens."
        )

    cached_companies = len(st.session_state.get("grok_contact_cache", {}))

    left, right = st.columns([2, 1])
    limit = left.slider(
        "How many companies to check", min_value=5,
        max_value=min(200, max(5, len(working_df))),
        value=min(25, len(working_df)), step=5,
        help="Largest employers first. Batched several per call to keep token use down.",
    )
    right.metric("Already cached", f"{cached_companies:,}")

    calls = -(-limit // CONTACT_BATCH_SIZE)

    # Rough forecast: ~700 input and ~250 output tokens per call, plus about
    # two searches per company. Deliberately not optimistic.
    estimate = (
        calls * 700 / 1_000_000 * rates["input_per_million"]
        + calls * 250 / 1_000_000 * rates["output_per_million"]
        + limit * 2 / 1_000 * rates["per_thousand_searches"]
    )
    st.caption(
        f"About {calls} call(s) for {limit} companies ({CONTACT_BATCH_SIZE} per call). "
        f"Rough estimate ${estimate:.2f}, roughly ${estimate / max(limit, 1):.3f} per "
        "company. Companies already checked are free."
    )

    cache_key = f"contacts::{file_key}::{len(working_df)}::{limit}"

    if st.button(f"Check {limit} companies", type="primary"):
        progress = st.progress(0.0)
        try:
            st.session_state[cache_key] = resolve_contacts(
                working_df, XAI_API_KEY, limit, rates, progress
            )
        except Exception as err:
            st.error(f"Contact lookup failed: {err}")
            return working_df, None
        finally:
            progress.empty()

    if cache_key not in st.session_state:
        return working_df, None

    resolved, diagnostics = st.session_state[cache_key]
    counts = resolved["Contact Status"].value_counts()

    a, b, c, d = st.columns(4)
    a.metric("Verified", f"{counts.get(STATUS_CONFIRMED, 0):,}",
             help="Name on file confirmed at the company")
    b.metric("Updated", f"{counts.get(STATUS_CORRECTED, 0):,}",
             help="A different current decision maker was found")
    c.metric("Unverified", f"{counts.get(STATUS_UNVERIFIED, 0):,}",
             help="Business exists but publishes no names")
    d.metric("Missing", f"{counts.get(STATUS_MISSING, 0):,}",
             help="No web presence and no name to work with")

    checked = max(diagnostics["checked"], 1)
    session_spend = st.session_state.get("xai_spend", 0.0)

    cost_a, cost_b, cost_c = st.columns(3)
    cost_a.metric("This run", f"${diagnostics['cost']:.2f}")
    cost_b.metric("Per company", f"${diagnostics['cost'] / checked:.3f}",
                  help="Cost divided by companies newly checked. Cached ones cost nothing.")
    cost_c.metric("Session total", f"${session_spend:.2f}",
                  help="Every contact run since this page was loaded.")

    with st.expander("Cost breakdown"):
        token_cost = (
            diagnostics["input_tokens"] / 1_000_000 * rates["input_per_million"]
            + diagnostics["cached_tokens"] / 1_000_000 * rates["cached_input_per_million"]
            + diagnostics["output_tokens"] / 1_000_000 * rates["output_per_million"]
        )
        search_cost = diagnostics["searches"] / 1_000 * rates["per_thousand_searches"]

        st.dataframe(pd.DataFrame([
            {"Item": "Input tokens", "Count": f"{diagnostics['input_tokens']:,}",
             "Cost": f"${diagnostics['input_tokens'] / 1_000_000 * rates['input_per_million']:.4f}"},
            {"Item": "Cached input tokens", "Count": f"{diagnostics['cached_tokens']:,}",
             "Cost": f"${diagnostics['cached_tokens'] / 1_000_000 * rates['cached_input_per_million']:.4f}"},
            {"Item": "Output tokens", "Count": f"{diagnostics['output_tokens']:,}",
             "Cost": f"${diagnostics['output_tokens'] / 1_000_000 * rates['output_per_million']:.4f}"},
            {"Item": "Web searches", "Count": f"{diagnostics['searches']:,}",
             "Cost": f"${search_cost:.4f}"},
        ]), use_container_width=True, hide_index=True)

        st.caption(
            f"Tokens ${token_cost:.4f}, searches ${search_cost:.4f}. "
            f"{diagnostics['calls']} call(s), {diagnostics['checked']} companies checked, "
            f"{diagnostics['cached']} served from cache."
        )

        if not diagnostics["searches_reported"]:
            st.warning(
                "xAI didn't report search counts on at least one call, so the "
                "search line is understated. Treat the total as a floor and "
                "check the xAI console for the authoritative number."
            )

        if st.button("Reset session total"):
            st.session_state["xai_spend"] = 0.0
            st.session_state["xai_calls"] = 0

        for message in diagnostics["errors"]:
            st.error(message)

    updated = resolved[resolved["Contact Status"] == STATUS_CORRECTED]
    if not updated.empty:
        with st.expander(f"Review the {len(updated):,} updated contacts"):
            st.dataframe(
                updated[[
                    "Company Name", "Executive First Name", "Executive Last Name",
                    "Executive Title", "Current First Name", "Current Last Name",
                    "Current Title", "Contact Source",
                ]],
                use_container_width=True, hide_index=True,
            )
        st.caption(
            "Open a source link before a setter calls - these are web findings, "
            "not a maintained database."
        )

    # Which statuses carry forward is a judgement call, so make it one.
    keep_unverified = st.checkbox(
        "Also keep unverified companies (name on file, nothing found online)",
        value=True,
        help="These still have your original contact. Turn off to work only "
             "names that were confirmed or corrected.",
    )

    allowed = list(ACTIONABLE_STATUSES)
    if keep_unverified:
        allowed.append(STATUS_UNVERIFIED)

    actionable = resolved[resolved["Contact Status"].isin(allowed)].copy()
    set_aside = resolved[~resolved["Contact Status"].isin(allowed)].copy()

    st.caption(f"{len(actionable):,} leads carried into scoring, {len(set_aside):,} set aside.")

    if not set_aside.empty:
        download_row(set_aside, "Download set-aside leads", f"set_aside_{file_key}")

    return actionable, set_aside


def stage_five_score(working_df: pd.DataFrame, file_key: str) -> pd.DataFrame:
    """Stage 4: score what's left."""
    st.subheader("5. Score the remaining leads")

    if not XAI_API_KEY:
        st.info("No xAI key. Add XAI_API_KEY to Secrets to enable scoring.")
        return working_df

    if not st.toggle("Enable scoring", value=False,
                     help="Off by default so it can't spend tokens unattended."):
        return working_df

    criteria = st.text_area("What makes a good lead?",
                            value=DEFAULT_SCORING_CRITERIA, height=160)

    calls = -(-len(working_df) // SCORING_BATCH_SIZE)
    st.caption(f"{len(working_df):,} leads, about {calls} API calls.")

    cache_key = f"scored::{file_key}::{len(working_df)}"

    if st.button(f"Score {len(working_df):,} leads", type="primary"):
        progress = st.progress(0.0)
        try:
            st.session_state[cache_key] = score_leads(
                working_df, XAI_API_KEY, criteria, progress
            )
        except Exception as err:
            st.error(f"Scoring failed: {err}")
            return working_df
        finally:
            progress.empty()

    if cache_key not in st.session_state:
        return working_df

    scored = st.session_state[cache_key]
    unscored = int(scored["Fit Score"].isna().sum())
    if unscored:
        st.warning(f"{unscored:,} leads came back unscored. Run again to retry.")

    floor = st.slider("Minimum fit score", 1, 10, 1)
    if floor > 1:
        scored = scored[scored["Fit Score"].fillna(0) >= floor].copy()
        st.caption(f"{len(scored):,} leads at {floor} or above.")

    return scored


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.title(APP_TITLE)
    st.write(
        "Five stages, in order: filter by size, verify the address is a mailable "
        "commercial location, keep the industries worth working, confirm or "
        "correct the decision maker, then score what's left."
    )

    uploaded_file = st.file_uploader("Upload a lead list", type=["csv"])
    if uploaded_file is None:
        st.info("Choose a CSV to get started.")
        return

    try:
        raw_df = load_csv(uploaded_file)
    except ValueError as err:
        st.error(str(err))
        return
    except Exception as err:
        st.error(f"Couldn't read that file: {err}")
        return

    column_df, found, missing_columns = filter_columns(raw_df)
    if not found:
        st.error("None of the expected columns were found. Is this a raw lead export?")
        st.write("Columns in your file:", list(raw_df.columns))
        return
    if missing_columns:
        st.warning("Not in this file: " + ", ".join(missing_columns))

    st.caption(f"{len(raw_df):,} rows in, {len(found)} of {len(KEEP_COLUMNS)} columns kept.")
    st.divider()

    working_df = stage_one_size(column_df)
    if working_df.empty:
        return

    st.divider()
    working_df = stage_two_address(working_df, uploaded_file.name)

    st.divider()
    working_df = stage_three_industry(working_df, uploaded_file.name)
    if working_df.empty:
        st.info("Nothing left after industry filtering.")
        return

    st.divider()
    working_df, set_aside = stage_four_contacts(working_df, uploaded_file.name)

    st.divider()
    working_df = stage_five_score(working_df, uploaded_file.name)

    # --- Final list ---
    st.divider()
    st.subheader("Working list")

    if set_aside is not None and not set_aside.empty:
        st.caption(
            f"{len(working_df):,} leads carried forward, "
            f"{len(set_aside):,} set aside."
        )

    st.dataframe(color_status(working_df.head(MAX_PREVIEW_ROWS)), use_container_width=True)

    if len(working_df) > MAX_PREVIEW_ROWS:
        st.caption(f"Showing {MAX_PREVIEW_ROWS:,} rows. The download has all of them.")

    download_row(working_df, "Download working list", f"leads_{uploaded_file.name}", primary=True)


if __name__ == "__main__":
    main()
