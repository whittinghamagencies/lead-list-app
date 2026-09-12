"""
Lead List Prep - internal prospecting pipeline
==============================================

Four stages, run in order. Each one narrows the list:

    1. Employees - drop companies below a headcount threshold        (free)
    2. Industry  - prioritise blue collar, exclude what you choose    (free)
    3. Smarty    - business data: headcount, SIC, executive contact   (metered)
    4. Grok      - find or confirm the decision maker for the rest    (metered)

The two free stages run first so nothing metered is spent on a company that
was going to be discarded anyway. Stage 3 writes resolved columns that stage 4
reads, so vetted data carries forward.
                     green  = listed person confirmed still there
                     yellow = updated with a new name found online
                     grey   = business found, no names published
                     red    = nothing found, set aside

The output is a working call list, sorted blue collar and largest first.

Run with:  streamlit run app.py
Requires:  streamlit, pandas, requests, smartystreets_python_sdk
"""

import json

import pandas as pd
import streamlit as st

try:
    from smartystreets_python_sdk import ClientBuilder, StaticCredentials, Batch
    from smartystreets_python_sdk import us_street, us_enrichment
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

# Statuses that carry a usable name into the working list.
ACTIONABLE_STATUSES = (STATUS_CONFIRMED, STATUS_CORRECTED)

XAI_RESPONSES_URL = "https://api.x.ai/v1/responses"   # agentic endpoint, has web_search

# Contact lookup is a retrieval task, not a reasoning one, so the cheaper
# long-context model is the right default. Rates are per million tokens.
CONTACT_MODELS = {
    "grok-4-fast - cheapest": {
        "id": "grok-4-fast-non-reasoning",
        "input_per_million": 0.20,
        "cached_input_per_million": 0.05,
        "output_per_million": 0.50,
        "note": "Roughly 6x cheaper on tokens than the 4.x tier. No reasoning "
                "step, which suits a lookup task. Least capable at following "
                "the JSON format, so expect the odd unparsed batch.",
    },
    "grok-4.20-non-reasoning - cheap, 1M context": {
        "id": "grok-4.20-0309-non-reasoning",
        "input_per_million": 1.25,
        "cached_input_per_million": 0.20,
        "output_per_million": 2.50,
        "note": "Skips the reasoning step but keeps 4.x-tier comprehension.",
    },
    "grok-4.3 - balanced, 1M context": {
        "id": "grok-4.3",
        "input_per_million": 1.25,
        "cached_input_per_million": 0.20,
        "output_per_million": 2.50,
        "note": "Reliable middle option. Good if the cheap models miss results.",
    },
    "grok-4.5 - pricier flagship": {
        "id": "grok-4.5",
        "input_per_million": 2.00,
        "cached_input_per_million": 0.30,
        "output_per_million": 6.00,
        "note": "Best comprehension, ~10x the token cost of grok-4-fast. "
                "Only worth it if the cheaper models are clearly wrong.",
    },
}

DEFAULT_CONTACT_MODEL = "grok-4-fast - cheapest"

# Grok bills DOUBLE on the entire request once it crosses this size, so
# keeping each request small matters more than batching many into one.
LONG_CONTEXT_THRESHOLD = 200_000
LONG_CONTEXT_MULTIPLIER = 2.0

# --- Cost model -------------------------------------------------------------
# Web search bills per call regardless of model, and is usually the largest
# line on the bill. Editable in the UI since xAI's rates change.
DEFAULT_SEARCH_RATE = 5.00   # $ per 1,000 web search tool calls

# Two companies per call, not more. Every web search injects its results into
# the request context, so a large batch balloons input tokens and can tip the
# whole request into the double-rate tier. Several small requests cost less in
# total than one large one.
CONTACT_BATCH_SIZE = 2
CONTACT_MAX_TOKENS = 500      # replies are compact JSON, so cap them hard

# Reasoning effort defaults to "high" on Grok 4.x models that have a reasoning
# step, and reasoning tokens bill at the output rate. This is a lookup task,
# so low is plenty. Non-reasoning models reject this parameter entirely.
CONTACT_REASONING_EFFORT = "low"

# Ceiling on searches per company, stated in the prompt. Searches are the
# dominant cost, so this is the main dial for spend.
CONTACT_MAX_SEARCHES = 2

# Below this headcount a business rarely has anyone between the owner and
# payroll, so the executive IS the benefits contact and a separate search
# would spend money to rediscover the same person.
BENEFITS_TARGET_MIN_EMPLOYEES = 20

# Roles that administer employee benefits, in the order to prefer them.
BENEFITS_TITLES = (
    "HR Director, HR Manager, Benefits Director, Benefits Manager, "
    "Benefits Coordinator, People Director, Personnel Manager, Payroll Manager, "
    "Office Manager, Business Manager, Practice Manager, Controller, "
    "Finance Manager"
)





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
# Optional. Only needed when the account holds several Smarty subscriptions
# and the default one isn't US Street. Accepts a comma-separated list, which
# Smarty evaluates in order until it finds an active subscription -
# e.g. "us-core-cloud, us-rooftop-geocoding-cloud".
SMARTY_LICENSE = get_secret("SMARTY_LICENSE")

# Which Smarty product the address stage uses. US Street verifies deliverability
# and flags residential vs commercial. US Enrichment's business dataset returns
# far more - employee counts, SIC codes, and executive contacts - but says
# nothing about mail deliverability.
SMARTY_MODE_STREET = "US Street - address verification"
SMARTY_MODE_BUSINESS = "US Enrichment - business data"
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
# Stage 0 - Row identity and deduplication
# ===========================================================================
#
# Runs before anything metered. A duplicate row costs a Smarty lookup and a
# web search every time it appears, so removing them here is free money.
# Nothing is deleted silently - duplicates are flagged and the user decides.

DUPLICATE_UNIQUE = "Unique"
DUPLICATE_STRONG = "Strong duplicate"
DUPLICATE_PROBABLE = "Probable duplicate"
DUPLICATE_POSSIBLE = "Possible duplicate"

DUPLICATE_COLORS = {
    DUPLICATE_UNIQUE: "",
    DUPLICATE_STRONG: "#fbd5d5",
    DUPLICATE_PROBABLE: "#fdf3c8",
    DUPLICATE_POSSIBLE: "#e6e8eb",
}

# Street suffixes that vary between exports of the same address.
STREET_ABBREVIATIONS = {
    "street": "st", "avenue": "ave", "road": "rd", "drive": "dr",
    "boulevard": "blvd", "lane": "ln", "court": "ct", "circle": "cir",
    "place": "pl", "parkway": "pkwy", "highway": "hwy", "suite": "ste",
    "building": "bldg", "north": "n", "south": "s", "east": "e", "west": "w",
}


def normalize_address(address: str) -> str:
    """
    Reduce an address to a comparable form.

    "955 Chenault Road # A" and "955 Chenault Rd #A" are the same place, and
    a purchased list will contain both spellings.
    """
    if not isinstance(address, str):
        return ""

    cleaned = "".join(
        char if char.isalnum() else " " for char in address.lower()
    )

    words = []
    for word in cleaned.split():
        words.append(STREET_ABBREVIATIONS.get(word, word))

    return " ".join(words)


def normalize_phone(phone: str) -> str:
    """Last ten digits, so formatting differences don't matter."""
    return "".join(char for char in str(phone or "") if char.isdigit())[-10:]


def add_row_identity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Give every row a stable id and normalized keys.

    Source Row ID survives every later stage, so a lead can always be traced
    back to its line in the original file.
    """
    result = df.copy()
    result.insert(0, "Source Row ID", [f"R{index + 1:05d}" for index in range(len(result))])
    result["Normalized Company"] = result["Company Name"].map(
        lambda name: " ".join(sorted(normalize_company(name)))
    )
    result["Normalized Address"] = result.get(
        STREET_COLUMN, pd.Series("", index=result.index)
    ).map(normalize_address)
    return result


def find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag duplicate rows in three strengths, strongest first.

    Adds Duplicate Status, Duplicate Group ID and Duplicate Reason. The first
    row of each group stays Unique so that keeping "one of each" is a simple
    filter rather than a judgement call.
    """
    result = df.copy()
    result["Duplicate Status"] = DUPLICATE_UNIQUE
    result["Duplicate Group ID"] = ""
    result["Duplicate Reason"] = ""

    city = result.get(CITY_COLUMN, pd.Series("", index=result.index)).fillna("").astype(str)
    state = result.get(STATE_COLUMN, pd.Series("", index=result.index)).fillna("").astype(str)
    phone = result.get(
        "Phone Number Combined", pd.Series("", index=result.index)
    ).map(normalize_phone)

    # Each tier is checked in order; a row already flagged isn't re-flagged,
    # so the strongest evidence is what gets recorded.
    tiers = [
        (DUPLICATE_STRONG, "same name and street address",
         result["Normalized Company"] + "|" + result["Normalized Address"],
         lambda index: bool(result.at[index, "Normalized Company"])
                       and bool(result.at[index, "Normalized Address"])),
        (DUPLICATE_PROBABLE, "same name and phone",
         result["Normalized Company"] + "|" + phone,
         lambda index: bool(result.at[index, "Normalized Company"]) and bool(phone.at[index])),
        (DUPLICATE_POSSIBLE, "same name, city and state",
         result["Normalized Company"] + "|" + city.str.lower() + "|" + state.str.lower(),
         lambda index: bool(result.at[index, "Normalized Company"])),
    ]

    group_counter = 0

    for status, reason, keys, is_valid in tiers:
        for key, indexes in keys.groupby(keys).groups.items():
            indexes = [i for i in indexes if is_valid(i)]
            if len(indexes) < 2:
                continue

            # Skip the group if these rows are already in a stronger group.
            unflagged = [i for i in indexes
                         if result.at[i, "Duplicate Status"] == DUPLICATE_UNIQUE
                         and not result.at[i, "Duplicate Group ID"]]
            if len(unflagged) < 2:
                continue

            group_counter += 1
            group_id = f"D{group_counter:04d}"

            for position, index in enumerate(unflagged):
                result.at[index, "Duplicate Group ID"] = group_id
                result.at[index, "Duplicate Reason"] = reason
                # First row of the group is the keeper.
                if position > 0:
                    result.at[index, "Duplicate Status"] = status

    return result


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


def build_smarty_client(auth_id: str, auth_token: str, license_name: str = ""):
    """
    Build a US Street API client.

    A license is only specified when one is configured. Accounts holding
    several Smarty subscriptions may need it pointed at the right product,
    otherwise the request is rejected as having no active subscription.
    """
    builder = ClientBuilder(StaticCredentials(auth_id, auth_token))

    licenses = [name.strip() for name in license_name.split(",") if name.strip()]
    if licenses:
        # Smarty tries these in order and uses the first with an active
        # subscription, so listing several is a safe fallback.
        builder = builder.with_licenses(licenses)

    return builder.build_us_street_api_client()


def verify_addresses(df: pd.DataFrame, auth_id: str, auth_token: str,
                     license_name: str = "", progress=None) -> pd.DataFrame:
    """Send mailing addresses to Smarty in batches of 100 and append results."""
    client = build_smarty_client(auth_id, auth_token, license_name)

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


def blank_business() -> dict:
    """The shape of one business enrichment result, all empty."""
    return {
        "Business Status": "",
        "Smarty Employee Count": "",
        "Smarty SIC Code": "",
        "Smarty SIC Description": "",
        "Smarty Contact First Name": "",
        "Smarty Contact Last Name": "",
        "Smarty Contact Title": "",
        "Smarty Executive Level": "",
        "Smarty Phone": "",
        "Years In Business": "",
    }


def business_to_columns(attributes) -> dict:
    """Map the fields we care about out of a business detail record."""
    def value(name):
        result = getattr(attributes, name, None)
        return "" if result is None else str(result)

    return {
        "Business Status": value("business_status"),
        # The actual headcount at this location, not a purchased range.
        "Smarty Employee Count": value("location_employee_count")
                                 or value("location_employee_count_range"),
        "Smarty SIC Code": value("primary_sic_code"),
        "Smarty SIC Description": value("primary_sic_description"),
        "Smarty Contact First Name": value("contact_first_name"),
        "Smarty Contact Last Name": value("contact_last_name"),
        "Smarty Contact Title": value("contact_professional_title"),
        "Smarty Executive Level": value("executive_level"),
        "Smarty Phone": value("phone"),
        "Years In Business": value("number_of_years_in_business"),
    }


# Words that don't help decide whether two company names are the same business.
COMPANY_NOISE = {
    "inc", "llc", "ltd", "co", "corp", "corporation", "company", "the",
    "svc", "svcs", "service", "services", "group", "holdings", "and", "of",
}


def normalize_company(name: str) -> set:
    """Reduce a company name to its meaningful words for comparison."""
    if not isinstance(name, str):
        return set()

    cleaned = "".join(char if char.isalnum() else " " for char in name.lower())
    return {word for word in cleaned.split() if word and word not in COMPANY_NOISE}


def same_company(ours: str, theirs: str) -> bool:
    """
    Whether two company names refer to the same business.

    Compares meaningful words rather than exact strings, so "Aska USA" matches
    "Aska USA Inc." but "Apollo Aviation" does not match "Apollo Roofing".
    """
    ours_words, theirs_words = normalize_company(ours), normalize_company(theirs)
    if not ours_words or not theirs_words:
        return False

    overlap = len(ours_words & theirs_words)
    return overlap / min(len(ours_words), len(theirs_words)) >= 0.6


def pick_business_id(entries, company_name: str) -> str:
    """
    Choose which business at this address is ours.

    One address can hold several businesses - a strip mall, a shared office -
    so prefer a name match and only fall back to the single result when
    there's exactly one.
    """
    if not entries:
        return ""

    for entry in entries:
        if same_company(company_name, getattr(entry, "company_name", "") or ""):
            return getattr(entry, "business_id", "") or ""

    if len(entries) == 1:
        return getattr(entries[0], "business_id", "") or ""

    return ""   # several businesses, none matching by name - don't guess


def enrich_businesses(df: pd.DataFrame, auth_id: str, auth_token: str,
                      license_name: str = "", limit: int = 0, progress=None):
    """
    Look each company up in Smarty's business dataset.

    Two steps, because the detail record is keyed by business_id:
      1. search by address and company name  -> a list of businesses there
      2. fetch the detail for the matching business_id

    That means up to two lookups per company, so it is capped. Returns
    (dataframe, diagnostics).
    """
    builder = ClientBuilder(StaticCredentials(auth_id, auth_token))
    licenses = [name.strip() for name in license_name.split(",") if name.strip()]
    if licenses:
        builder = builder.with_licenses(licenses)
    client = builder.build_us_enrichment_api_client()

    results = {index: blank_business() for index in df.index}
    diagnostics = {
        "searched": 0, "found_at_address": 0, "matched": 0,
        "ambiguous": 0, "errors": [],
    }

    targets = list(df.index)[:limit] if limit else list(df.index)

    for position, index in enumerate(targets):
        company = str(df.at[index, "Company Name"] or "")

        try:
            # Step 1 - what businesses are at this address?
            search = us_enrichment.Lookup(
                street=str(df.at[index, STREET_COLUMN] or ""),
                city=str(df.at[index, CITY_COLUMN] or ""),
                state=str(df.at[index, STATE_COLUMN] or ""),
                zipcode=str(df.at[index, ZIP_COLUMN] or ""),
                business_name=company,
            )
            diagnostics["searched"] += 1
            found = client.send_business_lookup(search)

            entries = []
            for response in found or []:
                entries.extend(getattr(response, "businesses", []) or [])

            if entries:
                diagnostics["found_at_address"] += 1

            business_id = pick_business_id(entries, company)
            if not business_id:
                if len(entries) > 1:
                    diagnostics["ambiguous"] += 1
                    results[index]["Business Status"] = (
                        f"{len(entries)} businesses at this address, none matching by name"
                    )
                continue

            # Step 2 - the full record for that business.
            detail = client.send_business_detail_lookup(
                us_enrichment.BusinessDetailLookup(business_id)
            )
            if detail:
                record = detail[0] if isinstance(detail, list) else detail
                attributes = getattr(record, "attributes", record)
                results[index] = business_to_columns(attributes)
                diagnostics["matched"] += 1

        except Exception as err:
            if len(diagnostics["errors"]) < 3:
                diagnostics["errors"].append(str(err)[:140])

        if progress is not None:
            progress.progress((position + 1) / max(len(targets), 1))

    contacts = pd.DataFrame.from_dict(results, orient="index")

    # Location Type and Basis already exist from the derived signal, so joining
    # would collide. Keep Grok's value where it gave one, our derived value
    # otherwise, and join only the genuinely new columns.
    merged = df.copy()
    for column in list(contacts.columns):
        if column not in merged.columns:
            continue
        incoming = contacts[column].fillna("").astype(str).str.strip()
        merged[column] = incoming.where(incoming.ne(""), merged[column])
        contacts = contacts.drop(columns=[column])

    return merged.join(contacts), diagnostics



# --- Comparing your file against what Smarty returned ----------------------
#
# Four outcomes per field, so the table shows at a glance what changed:
#   Agrees   - both sources say the same thing
#   Changed  - Smarty has something different; worth a look before calling
#   Filled   - your file was blank, Smarty supplied a value
#   No data  - Smarty returned nothing for this field

DIFF_AGREES = "Agrees"
DIFF_CHANGED = "Changed"
DIFF_FILLED = "Filled"
DIFF_NONE = "No data"

DIFF_COLORS = {
    DIFF_AGREES: "#d4f4dd",    # green
    DIFF_CHANGED: "#fdf3c8",   # yellow
    DIFF_FILLED: "#d6e9fb",    # blue
    DIFF_NONE: "#f2f3f5",      # grey
}

# Which original column each Smarty column is compared against.
COMPARISON_FIELDS = [
    ("Employees", SIZE_COLUMN, "Smarty Employee Count"),
    ("Contact name", "Executive Last Name", "Smarty Contact Last Name"),
    ("Contact title", "Executive Title", "Smarty Contact Title"),
    ("SIC code", "Primary SIC Code", "Smarty SIC Code"),
    ("Phone", "Phone Number Combined", "Smarty Phone"),
]


def names_match(our_first: str, our_last: str, their_first: str, their_last: str) -> bool:
    """
    Same person? Surname must match; first names match on initial.

    The initial rule catches Mike/Michael and Bob/Robert, which differ
    constantly between a purchased list and a maintained database.
    """
    our_last, their_last = str(our_last).strip().lower(), str(their_last).strip().lower()
    if not our_last or our_last != their_last:
        return False

    our_first, their_first = str(our_first).strip().lower(), str(their_first).strip().lower()
    if not our_first or not their_first:
        return True   # surname matched and one first name is blank

    return our_first[0] == their_first[0]


def is_blank(value) -> bool:
    """Treat None, NaN and the string forms of them as empty."""
    text = str(value or "").strip()
    return text == "" or text.lower() in ("none", "nan", "not available")


def parse_employee_range(text):
    """
    Lower and upper bound of a size band. "10 to 19" -> (10, 19),
    "500+" -> (500, None), a bare number -> (n, n).
    """
    if is_blank(text):
        return None, None

    digits = "".join(char if char.isdigit() else " " for char in str(text).replace(",", ""))
    numbers = [int(part) for part in digits.split() if part]

    if not numbers:
        return None, None
    if len(numbers) == 1:
        return numbers[0], (None if "+" in str(text) else numbers[0])
    return numbers[0], numbers[-1]


def employees_agree(original, smarty) -> bool:
    """
    Whether Smarty's headcount sits inside the band your file claims.

    Your file has ranges; Smarty returns an exact count, so "within the
    range" is the honest comparison rather than string equality.
    """
    low, high = parse_employee_range(original)
    smarty_low, smarty_high = parse_employee_range(smarty)

    if low is None or smarty_low is None:
        return False
    if high is None:
        return smarty_low >= low
    return low <= smarty_low <= high or (smarty_high is not None and low <= smarty_high <= high)


def digits_only(value: str) -> str:
    return "".join(char for char in str(value or "") if char.isdigit())


def field_agrees(label: str, original, smarty) -> bool:
    """Whether the two sources agree, compared the way that field deserves."""
    if label == "Employees":
        return employees_agree(original, smarty)

    if label == "Phone":
        # Compare the last ten digits so formatting differences don't count.
        return digits_only(original)[-10:] == digits_only(smarty)[-10:]

    if label == "SIC code":
        # Same major group is close enough; exports differ on trailing digits.
        return digits_only(original)[:2] == digits_only(smarty)[:2]

    if label == "Contact title":
        ours, theirs = str(original).strip().lower(), str(smarty).strip().lower()
        return ours in theirs or theirs in ours

    return str(original).strip().lower() == str(smarty).strip().lower()


def compare_field(label: str, original, smarty) -> str:
    """One of the four outcomes above."""
    if is_blank(smarty):
        return DIFF_NONE
    if is_blank(original):
        return DIFF_FILLED
    return DIFF_AGREES if field_agrees(label, original, smarty) else DIFF_CHANGED


def build_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    A row per company showing your value, Smarty's value, and the verdict
    for each compared field.
    """
    rows = []
    for index, row in df.iterrows():
        entry = {"Company Name": row.get("Company Name", "")}

        for label, ours_col, theirs_col in COMPARISON_FIELDS:
            if ours_col not in df.columns or theirs_col not in df.columns:
                continue

            ours, theirs = row.get(ours_col), row.get(theirs_col)

            # Names compare on first initial plus surname, so Mike matches
            # Michael - handled here rather than in field_agrees because it
            # needs both name columns.
            if label == "Contact name":
                ours_display = f"{row.get('Executive First Name', '')} {ours}".strip()
                theirs_display = f"{row.get('Smarty Contact First Name', '')} {theirs}".strip()

                if is_blank(theirs):
                    verdict = DIFF_NONE
                elif is_blank(ours):
                    verdict = DIFF_FILLED
                elif names_match(str(row.get("Executive First Name", "")), str(ours),
                                 str(row.get("Smarty Contact First Name", "")), str(theirs)):
                    verdict = DIFF_AGREES
                else:
                    verdict = DIFF_CHANGED

                entry[f"{label} (yours)"] = ours_display
                entry[f"{label} (Smarty)"] = theirs_display
                entry[label] = verdict
                continue

            entry[f"{label} (yours)"] = "" if is_blank(ours) else str(ours)
            entry[f"{label} (Smarty)"] = "" if is_blank(theirs) else str(theirs)
            entry[label] = compare_field(label, ours, theirs)

        rows.append(entry)

    return pd.DataFrame(rows, index=df.index)


def color_comparison(comparison: pd.DataFrame):
    """Shade each verdict column, and the Smarty value beside it."""
    verdict_columns = [label for label, _, _ in COMPARISON_FIELDS
                       if label in comparison.columns]

    def shade(column):
        return [f"background-color: {DIFF_COLORS.get(value, '')}" for value in column]

    def shade_from(verdict_column):
        def inner(column):
            return [
                f"background-color: {DIFF_COLORS.get(verdict, '')}"
                for verdict in comparison[verdict_column]
            ]
        return inner

    styler = comparison.style
    for label in verdict_columns:
        styler = styler.apply(shade, subset=[label])
        smarty_column = f"{label} (Smarty)"
        if smarty_column in comparison.columns:
            styler = styler.apply(shade_from(label), subset=[smarty_column])

    return styler



# --- Resolved values -------------------------------------------------------
#
# Later stages must read the best value we hold, not the original upload.
# These columns are written once enrichment has run and every downstream
# stage reads them, so vetted data actually propagates.

RESOLVED_EMPLOYEES = "Employees (resolved)"
RESOLVED_SIC = "SIC (resolved)"
RESOLVED_SIC_DESC = "SIC description (resolved)"
RESOLVED_SOURCE = "Data source"


def apply_resolved_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add resolved columns that prefer Smarty's data over the uploaded file.

    Leaves the originals untouched so the comparison view still works and
    nothing is silently overwritten.
    """
    result = df.copy()

    has_smarty_size = "Smarty Employee Count" in result.columns
    has_smarty_sic = "Smarty SIC Code" in result.columns

    employees, sics, descriptions, sources = [], [], [], []

    for _, row in result.iterrows():
        smarty_size = row.get("Smarty Employee Count") if has_smarty_size else None
        smarty_sic = row.get("Smarty SIC Code") if has_smarty_sic else None
        smarty_desc = row.get("Smarty SIC Description") if has_smarty_sic else None

        used_smarty = False

        if not is_blank(smarty_size):
            employees.append(str(smarty_size))
            used_smarty = True
        else:
            employees.append(str(row.get(SIZE_COLUMN, "") or ""))

        if not is_blank(smarty_sic):
            sics.append(str(smarty_sic))
            descriptions.append("" if is_blank(smarty_desc) else str(smarty_desc))
            used_smarty = True
        else:
            sics.append(str(row.get("Primary SIC Code", "") or ""))
            descriptions.append(str(row.get("Primary SIC Code Description", "") or ""))

        sources.append("Smarty" if used_smarty else "Uploaded list")

    result[RESOLVED_EMPLOYEES] = employees
    result[RESOLVED_SIC] = sics
    result[RESOLVED_SIC_DESC] = descriptions
    result[RESOLVED_SOURCE] = sources

    return result


def recheck_size(df: pd.DataFrame, minimum: int):
    """
    Re-apply the size threshold using resolved headcounts.

    A company the purchased list called "1 to 4" may really have twelve
    people. Returns (still_qualifying, now_failing) so both are visible -
    a company that drops out on better data is a finding, not a silent loss.
    """
    if RESOLVED_EMPLOYEES not in df.columns:
        return df, df.iloc[0:0]

    bounds = df[RESOLVED_EMPLOYEES].map(lambda value: parse_employee_range(value)[0])

    # Unknown headcount keeps its place rather than being dropped on silence.
    keep = bounds.isna() | (bounds >= minimum)
    return df[keep].copy(), df[~keep].copy()



# --- Commercial signal, derived rather than purchased -----------------------
#
# RDI is a USPS-licensed dataset, so every vendor that serves it charges.
# But we don't need RDI to answer the real question, which is "is this a
# workplace worth visiting". A registered business record, a headcount, and
# an address that isn't an apartment answer it for free.

COMMERCIAL_STRONG = "Business confirmed"
COMMERCIAL_LIKELY = "Likely business"
COMMERCIAL_UNCLEAR = "Unclear"
COMMERCIAL_HOME = "Likely home-based"

COMMERCIAL_COLORS = {
    COMMERCIAL_STRONG: "#d4f4dd",
    COMMERCIAL_LIKELY: "#e8f4d9",
    COMMERCIAL_UNCLEAR: "#f2f3f5",
    COMMERCIAL_HOME: "#fdf3c8",
}

# Address fragments that point at a dwelling rather than a workplace.
RESIDENTIAL_MARKERS = ("apt ", "apt.", "apartment", "unit ", "# ", "trlr", "lot ")
COMMERCIAL_MARKERS = ("ste ", "suite", "bldg", "building", "floor", "fl ",
                      "plaza", "park", "center", "centre", "mall", "hwy", "highway")


def commercial_signal(row) -> tuple:
    """
    Judge whether an address is a workplace, from data already in hand.

    Returns (verdict, reason). Nothing here costs a lookup - it reads the
    Smarty business record we already paid for, the headcount, and the shape
    of the address itself.
    """
    reasons = []

    # A registered business record at this address is the strongest signal
    # there is, and stronger than RDI for our purpose - RDI reports what USPS
    # calls the mailbox, not whether people work there.
    has_record = not is_blank(row.get("Smarty Contact Last Name")) or \
        not is_blank(row.get("Smarty Employee Count"))

    employees = parse_employee_range(
        row.get(RESOLVED_EMPLOYEES) or row.get(SIZE_COLUMN)
    )[0]

    street = str(row.get("Verified Address") or row.get(STREET_COLUMN) or "").lower()

    looks_residential = any(marker in street for marker in RESIDENTIAL_MARKERS)
    looks_commercial = any(marker in street for marker in COMMERCIAL_MARKERS)

    if has_record:
        reasons.append("business record on file")
    if employees and employees >= 10:
        reasons.append(f"{employees}+ employees")
    if looks_commercial:
        reasons.append("suite or building in the address")
    if looks_residential:
        reasons.append("apartment or unit number")

    # A business with real headcount is a workplace wherever it sits.
    if has_record and employees and employees >= 10:
        return COMMERCIAL_STRONG, ", ".join(reasons)
    if has_record and not looks_residential:
        return COMMERCIAL_STRONG, ", ".join(reasons)
    if looks_commercial and not looks_residential:
        return COMMERCIAL_LIKELY, ", ".join(reasons)
    if looks_residential and (not employees or employees < 10):
        return COMMERCIAL_HOME, ", ".join(reasons)
    if has_record:
        return COMMERCIAL_LIKELY, ", ".join(reasons)

    # Ten people have to work somewhere, record or not.
    if employees and employees >= 10:
        return COMMERCIAL_LIKELY, ", ".join(reasons)

    return COMMERCIAL_UNCLEAR, ", ".join(reasons) or "nothing to go on"


def apply_commercial_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Add the Location Type and Location Basis columns."""
    result = df.copy()
    verdicts, reasons = [], []

    for _, row in result.iterrows():
        verdict, reason = commercial_signal(row)
        verdicts.append(verdict)
        reasons.append(reason)

    result["Location Type"] = verdicts
    result["Location Basis"] = reasons
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
    column = RESOLVED_SIC if RESOLVED_SIC in df.columns else "Primary SIC Code"
    if column not in df.columns:
        return pd.DataFrame(columns=["Group", "Industry", "Companies"])

    groups = df[column].map(sic_major_group)
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
        # Prefer the resolved SIC so Smarty's classification wins where it
        # differs from the uploaded file.
        sic_code = row.get(RESOLVED_SIC) if RESOLVED_SIC in df.columns \
            else row.get("Primary SIC Code")
        description = row.get(RESOLVED_SIC_DESC) if RESOLVED_SIC_DESC in df.columns \
            else row.get("Primary SIC Code Description")

        tier, reason = classify_industry(
            sic_code, description, excluded_groups, excluded_keywords,
        )
        tiers.append(tier)
        reasons.append(reason)

    result = df.copy()
    result["Industry Tier"] = tiers
    result["Industry Basis"] = reasons

    result["_tier_sort"] = result["Industry Tier"].map(TIER_ORDER)
    size_column = RESOLVED_EMPLOYEES if RESOLVED_EMPLOYEES in result.columns \
        else SIZE_COLUMN
    result["_size_sort"] = result[size_column].map(parse_min_employees) \
        if size_column in result.columns else 0
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
    "You verify business contacts using web search. "
    "Search each company separately. Prefer, in order: (1) the company's own "
    "website - about, team, staff, contact, or leadership pages; "
    "(2) state business filings and Secretary of State registries, which list "
    "owners, officers, and registered agents for small businesses; "
    "(3) local chamber of commerce and licensing board listings; "
    "(4) LinkedIn or business directories. "
    "Use the phone number and address given to confirm you have the right "
    "business, not a same-named company elsewhere. "
    f"Run at most {CONTACT_MAX_SEARCHES} searches per business. If that hasn't "
    "settled it, answer U and move on - do not keep searching. "
    "Keep your reasoning brief; this is a lookup, not an analysis.\n\n"
    "You are looking for TWO different people, and they are often not the "
    "same person:\n"
    "1. EXECUTIVE AUTHORITY - who controls the business. Owner, Founder, CEO, "
    "President, Managing Partner or Member, or Executive Director at a "
    "nonprofit.\n"
    f"2. BENEFITS TARGET - who would actually administer employee benefits: "
    f"{BENEFITS_TITLES}. Prefer them in that order. At a small company this "
    "may genuinely be the owner, but only say so if the evidence shows it - "
    "never assume it to fill the field.\n\n"
    "Never guess a name from the company name. Never invent a person. "
    "Prefer an empty field over a guess.\n"
    "Reply with a JSON array only - no prose, no markdown fences. "
    'Each element: {"i":<id>,"s":"<V|C|U|M>","n":"<exec name or empty>",'
    '"t":"<exec title or empty>","b":"<benefits person name or empty>",'
    '"bt":"<benefits person title or empty>","u":"<source URL or empty>",'
    '"l":"<C|H|U>"}. '
    "Status codes describe the EXECUTIVE only: "
    "V = the person named in the record was found currently at this company. "
    "C = that person was not found, but a different current executive was, "
    "so you are correcting the record - put the new name in n. "
    "U = the business exists online but publishes no personnel information. "
    "M = no evidence the business exists online at all. "
    "Use U, never C or M, when you simply could not find personnel details. "
    "Absence of information is never proof someone left. "
    'The "l" field is the location type, from whatever you already saw while '
    "searching - do not search again for it. "
    "C = a commercial premises: a shop, yard, plant, office or unit. "
    "H = run from a home or residence. U = you could not tell."
)


# Maps the model's single letters back to our status names.
STATUS_CODES = {
    "V": STATUS_CONFIRMED,
    "C": STATUS_CORRECTED,
    "U": STATUS_UNVERIFIED,
    "M": STATUS_MISSING,
}


def best_known_contact(row):
    """
    The freshest contact we already hold for a company, and where it came from.

    Smarty's business record is maintained, so it outranks a purchased list
    when the two differ. Returns (first, last, title, source).
    """
    smarty_last = str(row.get("Smarty Contact Last Name", "") or "").strip()
    if smarty_last and smarty_last.lower() not in ("none", "nan"):
        return (
            str(row.get("Smarty Contact First Name", "") or "").strip(),
            smarty_last,
            str(row.get("Smarty Contact Title", "") or "").strip(),
            "Smarty",
        )

    return (
        str(row.get("Executive First Name", "") or "").strip(),
        str(row.get("Executive Last Name", "") or "").strip(),
        str(row.get("Executive Title", "") or "").strip(),
        "list",
    )


def contact_prompt_line(index, row) -> str:
    """
    One compact line describing a company to check.

    Uses the best contact we hold, not just the original upload - so when
    Smarty has supplied or corrected a name, that's what gets verified.
    Includes phone and address because they disambiguate a local business
    far better than the name alone, and cost only a few tokens.
    """
    first, last, title, source = best_known_contact(row)
    listed = f"{first} {last}".strip()

    parts = [
        f"id={index}",
        str(row.get("Company Name", "") or ""),
        f"{row.get('Verified Address') or row.get(STREET_COLUMN) or ''}",
        f"{row.get(CITY_COLUMN, '')} {row.get(STATE_COLUMN, '')}".strip(),
        str(row.get("Phone Number Combined", "") or row.get("Smarty Phone", "") or ""),
    ]

    if listed:
        descriptor = f"record says: {listed}" + (f", {title}" if title else "")
        # Say when the two sources disagree, so the model resolves which is
        # current rather than just confirming the one we happened to send.
        original_last = str(row.get("Executive Last Name", "") or "").strip()
        if source == "Smarty" and original_last and original_last.lower() != last.lower():
            descriptor += f" (an older record said {row.get('Executive First Name', '')} {original_last})"
        parts.append(descriptor)
    else:
        parts.append(f"record has no name - find a {DM_TITLES.split(',')[0]} or similar")

    return " | ".join(part for part in parts if part)


def extract_usage(payload: dict) -> dict:
    """
    Pull token and tool counts out of a response.

    Field names differ between endpoints and have changed over time, so we
    check every plausible shape. Reasoning tokens are counted separately
    where reported, because they bill at the output rate and are usually the
    reason an agentic call costs more than expected.

    Anything missing comes back as zero, which understates rather than
    overstates - so treat the total as a floor, not a guarantee.
    """
    usage = payload.get("usage") or {}

    def first_of(*keys):
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                return int(value)
        return 0

    cached = first_of("cached_input_tokens", "cache_read_input_tokens",
                      "prompt_cached_tokens")
    total_input = first_of("input_tokens", "prompt_tokens")
    output = first_of("output_tokens", "completion_tokens")

    # Reasoning tokens may be nested in a details object or reported flat.
    reasoning = 0
    for container_key in ("completion_tokens_details", "output_tokens_details"):
        container = usage.get(container_key)
        if isinstance(container, dict):
            value = container.get("reasoning_tokens")
            if isinstance(value, (int, float)):
                reasoning = int(value)
                break
    if not reasoning:
        reasoning = first_of("reasoning_tokens")

    # Tool calls appear under several shapes; count anything search-like.
    searches = 0
    for container in (
        payload.get("server_side_tool_usage"),
        usage.get("server_side_tool_usage"),
        usage.get("tool_usage"),
        payload.get("tool_usage"),
    ):
        if not isinstance(container, dict):
            continue
        for key, value in container.items():
            if "search" not in key.lower():
                continue
            if isinstance(value, (int, float)):
                searches += int(value)
            elif isinstance(value, dict):
                # e.g. {"count": 3, "unit": "call"}
                count = value.get("count") or value.get("calls") or 0
                if isinstance(count, (int, float)):
                    searches += int(count)
    if not searches:
        searches = first_of("num_sources_used", "web_search_calls", "num_searches")

    return {
        "input_tokens": max(total_input - cached, 0),
        "cached_tokens": cached,
        "output_tokens": output,
        "reasoning_tokens": reasoning,   # already inside output_tokens
        "searches": searches,
        "searches_reported": bool(searches),
        "total_tokens": total_input + output,
    }


def usage_cost(usage: dict, rates: dict) -> float:
    """
    Dollar cost of one call.

    Applies the long-context multiplier when the request crossed the
    threshold, since Grok re-rates the whole request rather than just the
    tokens above the line.
    """
    multiplier = (
        LONG_CONTEXT_MULTIPLIER
        if usage.get("total_tokens", 0) >= LONG_CONTEXT_THRESHOLD else 1.0
    )

    token_cost = (
        usage["input_tokens"] / 1_000_000 * rates["input_per_million"]
        + usage["cached_tokens"] / 1_000_000 * rates["cached_input_per_million"]
        + usage["output_tokens"] / 1_000_000 * rates["output_per_million"]
    ) * multiplier

    return token_cost + usage["searches"] / 1_000 * rates["per_thousand_searches"]


def call_grok_search(prompt: str, api_key: str, model_id: str = "grok-4-fast-non-reasoning"):
    """
    One agentic call with web search enabled.

    Uses the /v1/responses endpoint because the older search_parameters API
    was retired in January 2026 and now returns 410.

    Returns (text, usage) so the caller can price the run.
    """
    body = {
        "model": model_id,
        "input": [
            {"role": "system", "content": CONTACT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "tools": [{"type": "web_search"}],
        "max_output_tokens": CONTACT_MAX_TOKENS,
    }

    # Reasoning tokens bill at the output rate, so keep effort low where the
    # model has a reasoning step at all. Non-reasoning models reject the
    # parameter outright.
    if "non-reasoning" not in model_id and "fast" not in model_id:
        body["reasoning_effort"] = CONTACT_REASONING_EFFORT

    response = requests.post(
        XAI_RESPONSES_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=240,
    )

    # Older API versions reject reasoning_effort; retry once without it
    # rather than failing the run.
    if response.status_code == 400 and "reasoning_effort" in body:
        body.pop("reasoning_effort")
        response = requests.post(
            XAI_RESPONSES_URL,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json=body,
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
                "benefits_name": str(item.get("b", "")).strip(),
                "benefits_title": str(item.get("bt", "")).strip(),
                "url": str(item.get("u", "")).strip(),
                # Location type rides along on the search we already paid for.
                "location": {"C": COMMERCIAL_STRONG, "H": COMMERCIAL_HOME}.get(
                    str(item.get("l", "")).upper()[:1], ""
                ),
            }
        except (KeyError, ValueError, TypeError):
            continue
    return parsed


def blank_contact() -> dict:
    """
    Two contacts per company, kept apart on purpose.

    Executive Authority controls the business. Benefits Target administers
    employee benefits. At a small company they are often the same person, but
    assuming that is how a call ends up with someone who can't help.
    """
    return {
        "Contact Status": "",
        "Location Type": "",
        "Location Basis": "",
        "Executive Authority": "",
        "Executive Title": "",
        "Executive Source": "",
        "Benefits Target": "",
        "Benefits Target Title": "",
        "Benefits Target Status": "",
        "Ask For": "",
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


BENEFITS_FOUND = "Found"
BENEFITS_IS_EXECUTIVE = "Executive handles it"
BENEFITS_NOT_FOUND = "Could not find"
BENEFITS_NOT_SEARCHED = "Not searched"

BENEFITS_COLORS = {
    BENEFITS_FOUND: "#d4f4dd",
    BENEFITS_IS_EXECUTIVE: "#e8f4d9",
    BENEFITS_NOT_FOUND: "#fdf3c8",
    BENEFITS_NOT_SEARCHED: "#f2f3f5",
}


def apply_contact_result(row, result: dict, searched_benefits: bool = True) -> dict:
    """
    Turn one parsed result into our contact columns.

    Fills the executive and the benefits target separately, then works out
    which of them the field team should actually ask for.
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

    if status == STATUS_UNVERIFIED and not had_name:
        status = STATUS_MISSING

    executive = f"{first} {last}".strip()
    executive_title = result.get("title", "")

    contact.update({
        "Contact Status": status,
        "Executive Authority": executive,
        "Executive Title": executive_title,
        "Executive Source": "web search",
        "Contact Source": result.get("url", ""),
    })

    # The benefits target, judged on its own evidence.
    benefits_name = result.get("benefits_name", "").strip()
    benefits_title = result.get("benefits_title", "").strip()

    if not searched_benefits:
        contact["Benefits Target Status"] = BENEFITS_NOT_SEARCHED
    elif benefits_name:
        contact["Benefits Target"] = benefits_name
        contact["Benefits Target Title"] = benefits_title
        # The model may legitimately name the owner - but only with evidence.
        contact["Benefits Target Status"] = (
            BENEFITS_IS_EXECUTIVE
            if executive and benefits_name.lower() == executive.lower()
            else BENEFITS_FOUND
        )
    else:
        contact["Benefits Target Status"] = BENEFITS_NOT_FOUND

    # Who the setter asks for: the benefits person when we have one, the
    # executive otherwise, and an honest fallback when we have neither.
    if benefits_name:
        contact["Ask For"] = benefits_name + (f" ({benefits_title})" if benefits_title else "")
    elif executive:
        contact["Ask For"] = executive + (f" ({executive_title})" if executive_title else "")
    else:
        contact["Ask For"] = "whoever handles employee benefits"

    if result.get("location"):
        contact["Location Type"] = result["location"]
        contact["Location Basis"] = "seen during web search"

    if status == STATUS_CONFIRMED:
        contact["Contact Note"] = "Listed executive confirmed"
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


def accept_smarty_contact(row) -> dict:
    """
    Take Smarty's business contact as the EXECUTIVE candidate only.

    It never satisfies the benefits target. A compiled executive record says
    who runs the company, not who processes enrollment, and treating the two
    as one is how a setter ends up asking for someone who can't help.
    """
    contact = blank_contact()

    first = str(row.get("Smarty Contact First Name", "") or "").strip()
    last = str(row.get("Smarty Contact Last Name", "") or "").strip()
    title = str(row.get("Smarty Contact Title", "") or "").strip()

    our_first = str(row.get("Executive First Name", "") or "").strip()
    our_last = str(row.get("Executive Last Name", "") or "").strip()

    agrees = bool(our_last) and names_match(our_first, our_last, first, last)
    executive = f"{first} {last}".strip()

    contact.update({
        "Contact Status": STATUS_CONFIRMED if agrees else STATUS_CORRECTED,
        "Executive Authority": executive,
        "Executive Title": title,
        "Executive Source": "Smarty business record",
        "Benefits Target Status": BENEFITS_NOT_SEARCHED,
        "Ask For": executive + (f" ({title})" if title else ""),
        "Contact Source": "Smarty business record",
        "Contact Note": (
            "Smarty confirms the listed executive" if agrees
            else (f"Smarty has {executive}" +
                  (f", replacing {our_first} {our_last}".strip() if our_last
                   else " - list had no name"))
        ),
    })
    return contact


def needs_benefits_search(row, threshold: int) -> bool:
    """
    Whether a separate benefits-target search is worth running.

    Below the threshold a business rarely has anyone between the owner and
    payroll, so searching would spend money to rediscover the executive. Above
    it, an office manager or controller usually handles enrollment and is a
    different person entirely.

    Unknown headcount gets searched - silence isn't evidence of being small.
    """
    employees = parse_employee_range(
        row.get(RESOLVED_EMPLOYEES) or row.get(SIZE_COLUMN)
    )[0]

    if employees is None:
        return True
    return employees >= threshold


def resolve_contacts(df: pd.DataFrame, api_key: str, limit: int, rates: dict,
                     model_id: str = "grok-4-fast-non-reasoning",
                     use_smarty: bool = True,
                     benefits_threshold: int = BENEFITS_TARGET_MIN_EMPLOYEES,
                     progress=None):
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
        "reasoning_tokens": 0, "searches": 0, "searches_reported": True,
        "cost": 0.0, "long_context_calls": 0, "peak_tokens": 0,
    }

    diagnostics["from_smarty"] = 0
    diagnostics["benefits_searched"] = 0

    cache = st.session_state.setdefault("grok_contact_cache", {})
    targets = list(df.index)[:limit]

    has_smarty = "Smarty Contact Last Name" in df.columns

    # Reuse anything already resolved for the same company name.
    pending = []
    for index in targets:
        company = str(df.at[index, "Company Name"] or "").strip()
        if not company:
            results[index]["Contact Status"] = STATUS_MISSING
            results[index]["Contact Note"] = "No company name"
            continue

        # Smarty answers the executive. Whether that's enough depends on size:
        # at a small company the owner handles benefits, at a larger one
        # someone else does and still needs finding.
        if use_smarty and has_smarty:
            smarty_last = str(df.at[index, "Smarty Contact Last Name"] or "").strip()
            has_smarty_contact = smarty_last and smarty_last.lower() not in ("none", "nan")

            if has_smarty_contact and not needs_benefits_search(df.loc[index], benefits_threshold):
                contact = accept_smarty_contact(df.loc[index])
                contact["Contact Note"] += (
                    f" - under {benefits_threshold} employees, so treated as the "
                    "benefits contact too"
                )
                results[index] = contact
                diagnostics["from_smarty"] += 1
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
            reply, usage = call_grok_search(prompt, api_key, model_id)
            parsed = parse_contact_reply(reply)
            diagnostics["calls"] += 1

            # Track spend as we go, so a cancelled run still reports its cost.
            for field in ("input_tokens", "cached_tokens", "output_tokens",
                          "reasoning_tokens", "searches"):
                diagnostics[field] += usage[field]

            # Flag calls that hit the double-rate tier - that's the signal to
            # cut batch size further.
            diagnostics["peak_tokens"] = max(diagnostics["peak_tokens"],
                                             usage.get("total_tokens", 0))
            if usage.get("total_tokens", 0) >= LONG_CONTEXT_THRESHOLD:
                diagnostics["long_context_calls"] += 1
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

            searched = needs_benefits_search(df.loc[index], benefits_threshold)
            if searched:
                diagnostics["benefits_searched"] += 1

            results[index] = apply_contact_result(df.loc[index], result, searched)
            cache[str(df.at[index, "Company Name"] or "").strip()] = result
            diagnostics["checked"] += 1

        if progress is not None:
            progress.progress(min((start + CONTACT_BATCH_SIZE) / max(len(pending), 1), 1.0))

    # Running total for the session, so the cost of several passes is visible.
    st.session_state["xai_spend"] = st.session_state.get("xai_spend", 0.0) + diagnostics["cost"]
    st.session_state["xai_calls"] = st.session_state.get("xai_calls", 0) + diagnostics["calls"]

    contacts = pd.DataFrame.from_dict(results, orient="index")

    # Location Type and Basis already exist from the derived signal, so joining
    # would collide. Keep Grok's value where it gave one, our derived value
    # otherwise, and join only the genuinely new columns.
    merged = df.copy()
    for column in list(contacts.columns):
        if column not in merged.columns:
            continue
        incoming = contacts[column].fillna("").astype(str).str.strip()
        merged[column] = incoming.where(incoming.ne(""), merged[column])
        contacts = contacts.drop(columns=[column])

    return merged.join(contacts), diagnostics



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
    if "Location Type" in df.columns:
        styler = styler.apply(shade, palette=COMMERCIAL_COLORS, subset=["Location Type"])
    if "Benefits Target Status" in df.columns:
        styler = styler.apply(shade, palette=BENEFITS_COLORS,
                              subset=["Benefits Target Status"])
    if "Duplicate Status" in df.columns:
        styler = styler.apply(shade, palette=DUPLICATE_COLORS, subset=["Duplicate Status"])
    return styler



# ===========================================================================
# Stage 5 - ChatGPT research handoff
# ===========================================================================
#
# The app qualifies; ChatGPT researches. Deep work - current executives, the
# benefits contact, physical vs corporate address, franchise/branch structure,
# rapport notes - is deferred to a ChatGPT subscription the team already pays
# for, rather than billed per search through the API.

HANDOFF_FILENAME = "CHATGPT_RESEARCH_HANDOFF.xlsx"

# Blank columns ChatGPT fills in. Order matters - it's what the prompt names.
RESEARCH_COLUMNS = [
    "Verified Physical Operating Address",
    "Verified Corporate / HQ Address",
    "Corporate Structure",
    "Parent Company",
    "Parent HQ",
    "Benefits Decision Likely",
    "Verified Executive Authority",
    "Verified Executive Title",
    "Benefits Target Name",
    "Benefits Target Title",
    "Executive Verification Status",
    "Benefits Target Verification Status",
    "Location Verification Status",
    "Contact Conflict Notes",
    "Address Conflict Notes",
    "Rapport Building Notes",
    "Research Confidence",
    "Research Source URLs",
]


def build_handoff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble the sheet ChatGPT receives.

    Carries identity, qualification, original data, enrichment and comparison
    verdicts, then leaves the research columns blank. Source Row ID is first
    so the merge back has an unambiguous key.
    """
    def column(name, default=""):
        if name in df.columns:
            return df[name].fillna("").astype(str)
        return pd.Series([default] * len(df), index=df.index)

    handoff = pd.DataFrame(index=df.index)

    # Identity
    handoff["Source Row ID"] = column("Source Row ID")
    handoff["Company Name"] = column("Company Name")
    handoff["City"] = column(CITY_COLUMN)
    handoff["State"] = column(STATE_COLUMN)
    handoff["Source ID"] = column("Infogroup ID")

    # Qualification
    handoff["Industry Tier"] = column("Industry Tier")
    handoff["Employees Resolved"] = column(RESOLVED_EMPLOYEES)
    handoff["Employee Data Source"] = column(RESOLVED_SOURCE)
    handoff["SIC Resolved"] = column(RESOLVED_SIC)
    handoff["SIC Description Resolved"] = column(RESOLVED_SIC_DESC)
    handoff["Business Status"] = column("Business Status")

    # Original data - kept so ChatGPT can see what it's correcting
    handoff["Original Mailing Address"] = column(STREET_COLUMN)
    handoff["Original Employee Range"] = column(SIZE_COLUMN)
    handoff["Original SIC"] = column("Primary SIC Code")
    handoff["Original SIC Description"] = column("Primary SIC Code Description")
    handoff["Original Executive Name"] = (
        column("Executive First Name") + " " + column("Executive Last Name")
    ).str.strip()
    handoff["Original Executive Title"] = column("Executive Title")
    handoff["Original Phone"] = column("Phone Number Combined")

    # Structured enrichment
    handoff["Enriched Employee Count"] = column("Smarty Employee Count")
    handoff["Enriched SIC"] = column("Smarty SIC Code")
    handoff["Enriched Executive Name"] = (
        column("Smarty Contact First Name") + " " + column("Smarty Contact Last Name")
    ).str.strip()
    handoff["Enriched Executive Title"] = column("Smarty Contact Title")
    handoff["Enriched Phone"] = column("Smarty Phone")
    handoff["Years In Business"] = column("Years In Business")

    # Address and location
    handoff["Verified Mailing Address"] = column("Verified Address")
    handoff["Postal Record Type"] = column("Record Type")
    handoff["Preliminary Location Type"] = column("Location Type")
    handoff["Preliminary Location Basis"] = column("Location Basis")

    # Anything the app already found, marked as a lead rather than an answer
    handoff["App Executive Candidate"] = column("Executive Authority")
    handoff["App Executive Candidate Source"] = column("Executive Source")
    handoff["App Benefits Candidate"] = column("Benefits Target")

    # Where the sources disagree, so research starts with the conflicts
    handoff["Duplicate Status"] = column("Duplicate Status")
    needs_resolution = pd.Series("No", index=df.index)
    if "Contact Status" in df.columns:
        needs_resolution = df["Contact Status"].map(
            lambda value: "Yes" if value in (STATUS_CORRECTED, STATUS_UNVERIFIED,
                                             STATUS_MISSING) else "No"
        )
    handoff["Needs Conflict Resolution"] = needs_resolution

    for name in RESEARCH_COLUMNS:
        handoff[name] = ""

    return handoff


def handoff_to_excel(handoff: pd.DataFrame) -> bytes:
    """Write the handoff sheet to XLSX bytes for the download button."""
    from io import BytesIO

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        handoff.to_excel(writer, index=False, sheet_name="Research")
    return buffer.getvalue()


CHATGPT_PROMPT = """I am uploading a file named CHATGPT_RESEARCH_HANDOFF.xlsx.

This file has already gone through structured data cleaning, deduplication,
business enrichment, employee-count resolution, SIC resolution, business-status
enrichment, and qualification filtering.

Do not redo the structured enrichment or re-filter the list. Treat the
structured data as the starting dataset, but verify any field relevant to the
research tasks below.

For EACH company, research and fill the blank research columns.

1. CONFIRM COMPANY IDENTITY
Make sure the public sources you find refer to the same company and location in
the file. If multiple matches exist, flag the ambiguity rather than guessing.

2. VERIFY BUSINESS STATUS
Classify as Active, Closed, or Unclear.

3. VERIFY ADDRESSES
Using the existing address data as clues, identify the actual Physical
Operating Address, the Corporate / HQ Address, whether they are the same, and
whether the physical location is Commercial Storefront, Professional Office,
Industrial/Warehouse, Residential/Home-Based, or Unknown.
Do not use a PO Box as the physical operating address.

4. VERIFY EXECUTIVE AUTHORITY
Use the Original Executive and Enriched Executive fields as leads, but
independently verify the CURRENT Owner, Founder, CEO, President, Managing
Partner/Member, or Executive Director. Do not assume the structured executive
record is current. If structured data conflicts with current public evidence,
preserve the conflict in the notes and use the best-supported current value.

5. INDEPENDENTLY FIND THE BENEFITS TARGET
This step MUST be completed even when an owner/CEO/president has already been
identified. Search in this priority order: HR Director, HR Manager, Benefits
Director/Manager/Coordinator, People/Personnel leader, Payroll Manager, Office
Manager, Business Manager, Practice Manager, Controller/Finance Manager, and
only then Executive Authority if no separate administrative contact exists.
Only use a real named individual with a supported title.
If no specific person can be supported after research, enter exactly:
Could not find

6. RESEARCH CORPORATE STRUCTURE
Determine whether the business is independent, franchise, branch, subsidiary,
division, nonprofit, government/public entity, or part of a larger corporate
group. If part of a larger organization, identify the parent company and parent
HQ, and estimate whether employee-benefits decisions are likely Local,
Corporate, Shared, or Unknown.

7. RESOLVE CONFLICTS
Compare original data, structured enrichment, and current public evidence. Do
not silently overwrite conflicting values. Explain meaningful conflicts in the
conflict-note columns.

8. RAPPORT-BUILDING RESEARCH
For qualified active businesses, add 2-3 specific outreach-relevant facts when
available: family-owned history, years in business, recent expansion, community
involvement, local awards, employee ownership, leadership background, major
local projects, Chamber involvement, notable specialty.
Do not add generic filler.

9. RESEARCH SOURCES
Prioritize: official company website, current government/corporate filings,
company documents/PDFs, Chamber/trade association, nonprofit Form 990 when
applicable, reputable current news, BBB/reputable directories, LinkedIn/public
professional results, other corroborating sources.
When a contact is difficult to find or sources conflict, search deeper --
corporate filings, public directories, PDFs, professional search results --
before marking it unresolved.

10. CONFIDENCE
Assign a Research Confidence score from 1-10 based on the strength and recency
of the evidence. Accuracy is more important than completeness. Do not guess.

OUTPUT REQUIREMENT
Return a downloadable XLSX. Preserve the existing rows and columns, including
Source Row ID exactly as provided. Populate the blank research fields. Do not
delete rows, do not reorder them, and do not add rows. If a business is closed
or its identity is ambiguous, flag it in the research fields rather than
removing it."""


def merge_research(working: pd.DataFrame, returned: pd.DataFrame):
    """
    Merge ChatGPT's returned workbook back in, keyed on Source Row ID.

    Returns (merged, problems). Merging on company name is refused - two
    locations of the same company would silently cross-contaminate.
    """
    problems = []

    if "Source Row ID" not in returned.columns:
        return working, ["The returned file has no Source Row ID column, so rows "
                         "can't be matched. Ask ChatGPT to return the file with "
                         "that column intact."]

    returned = returned.copy()
    returned["Source Row ID"] = returned["Source Row ID"].astype(str).str.strip()
    known = set(working["Source Row ID"].astype(str))

    unknown_ids = [value for value in returned["Source Row ID"] if value not in known]
    if unknown_ids:
        problems.append(
            f"{len(unknown_ids)} returned row(s) carry an ID not in the current "
            f"list (first: {unknown_ids[0]}). Those rows are ignored."
        )

    missing = len(known) - returned["Source Row ID"].isin(known).sum()
    if missing > 0:
        problems.append(
            f"{missing} row(s) sent for research didn't come back. They keep "
            "their existing values."
        )

    present = [name for name in RESEARCH_COLUMNS if name in returned.columns]
    if not present:
        return working, problems + [
            "None of the research columns are present in the returned file."
        ]
    if len(present) < len(RESEARCH_COLUMNS):
        problems.append(
            f"{len(RESEARCH_COLUMNS) - len(present)} research column(s) are "
            "missing from the returned file."
        )

    lookup = returned.set_index("Source Row ID")[present]
    merged = working.copy()
    keys = merged["Source Row ID"].astype(str)

    for name in present:
        merged[name] = keys.map(lookup[name]).fillna("")

    # A returned company name that doesn't match is a sign rows were reordered
    # or edited, which is worth saying out loud rather than merging quietly.
    if "Company Name" in returned.columns:
        returned_names = returned.set_index("Source Row ID")["Company Name"]
        mismatches = 0
        for key, ours in zip(keys, merged["Company Name"].astype(str)):
            theirs = returned_names.get(key)
            if theirs is not None and not same_company(ours, str(theirs)):
                mismatches += 1
        if mismatches:
            problems.append(
                f"{mismatches} row(s) came back with a company name that doesn't "
                "match the ID. Check before trusting those rows."
            )

    return merged, problems


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


def stage_dedupe(df: pd.DataFrame, file_key: str) -> pd.DataFrame:
    """
    Stage 0: flag duplicate rows before anything metered runs.

    A duplicate costs a Smarty lookup and a web search every time it appears,
    so this is the cheapest saving available. Nothing is deleted silently.
    """
    st.subheader("0. Duplicates")

    flagged = find_duplicates(add_row_identity(df))
    counts = flagged["Duplicate Status"].value_counts()
    duplicates = int(len(flagged) - counts.get(DUPLICATE_UNIQUE, 0))

    if not duplicates:
        st.caption(f"No duplicates found across {len(flagged):,} rows.")
        return flagged

    a, b, c = st.columns(3)
    a.metric("Strong", f"{counts.get(DUPLICATE_STRONG, 0):,}",
             help="Same company name and street address")
    b.metric("Probable", f"{counts.get(DUPLICATE_PROBABLE, 0):,}",
             help="Same company name and phone number")
    c.metric("Possible", f"{counts.get(DUPLICATE_POSSIBLE, 0):,}",
             help="Same company name, city and state, different address")

    with st.expander(f"Review the {duplicates:,} flagged rows"):
        st.dataframe(
            flagged[flagged["Duplicate Status"] != DUPLICATE_UNIQUE][[
                "Source Row ID", "Company Name", STREET_COLUMN, CITY_COLUMN,
                "Duplicate Status", "Duplicate Group ID", "Duplicate Reason",
            ]],
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "The first row of each group is kept. Possible duplicates may be "
            "genuinely separate locations of the same company - worth a look "
            "before removing those."
        )

    choice = st.radio(
        "What to do with them",
        ["Remove strong and probable", "Remove all flagged", "Keep everything"],
        horizontal=True,
        help="Every duplicate you keep costs a Smarty lookup and a web search.",
    )

    if choice == "Remove strong and probable":
        result = flagged[~flagged["Duplicate Status"].isin(
            [DUPLICATE_STRONG, DUPLICATE_PROBABLE])].copy()
    elif choice == "Remove all flagged":
        result = flagged[flagged["Duplicate Status"] == DUPLICATE_UNIQUE].copy()
    else:
        result = flagged

    saved = len(flagged) - len(result)
    if saved:
        st.caption(f"{len(result):,} rows carried forward - {saved:,} fewer paid lookups.")

    return result


def stage_employees(column_df: pd.DataFrame):
    """Stage 1: headcount threshold. Free, so it runs first."""
    st.subheader("1. Company size")

    if SIZE_COLUMN not in column_df.columns:
        st.info(f"No '{SIZE_COLUMN}' column, so size filtering is off.")
        return column_df

    left, right = st.columns([1, 2])
    minimum = left.number_input("Minimum employees", min_value=0,
                                value=DEFAULT_MIN_EMPLOYEES, step=1)
    include_unknown = right.checkbox("Keep companies with no size listed", value=False)

    st.session_state["size_minimum"] = minimum
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


def stage_smarty_business(working_df: pd.DataFrame, file_key: str) -> pd.DataFrame:
    """
    Stage 3, Enrichment mode: pull Smarty's business record for each company.

    Returns employee count, SIC code, and an executive contact - which means
    stage 4 has less to find. Says nothing about mail deliverability.
    """
    limit = st.slider(
        "How many companies to look up", min_value=5,
        max_value=min(500, max(5, len(working_df))),
        value=min(50, len(working_df)), step=5,
        help="One Enrichment lookup each. Trial lookups are limited, so start small.",
    )

    cache_key = f"business::{file_key}::{len(working_df)}::{limit}"

    if st.button(f"Look up {limit} companies", type="primary"):
        progress = st.progress(0.0)
        try:
            st.session_state[cache_key] = enrich_businesses(
                working_df, SMARTY_AUTH_ID, SMARTY_AUTH_TOKEN,
                SMARTY_LICENSE, limit, progress
            )
        except Exception as err:
            st.error(f"Enrichment lookup failed: {err}")
            st.caption(
                "If this mentions a subscription, the license name may be wrong - "
                "check the Smarty dashboard for the exact license string on the "
                "Enrichment trial and put it in SMARTY_LICENSE."
            )
            return working_df
        finally:
            progress.empty()

    if cache_key not in st.session_state:
        return working_df

    enriched, diagnostics = st.session_state[cache_key]

    a, b, c, d = st.columns(4)
    a.metric("Searched", f"{diagnostics['searched']:,}")
    b.metric("Businesses at address", f"{diagnostics['found_at_address']:,}",
             help="Smarty knows of a business there, matching by name or not.")
    c.metric("Matched", f"{diagnostics['matched']:,}",
             help="Name matched and the full record came back.")
    d.metric("With a contact name",
             f"{int(enriched['Smarty Contact Last Name'].fillna('').astype(str).str.strip().ne('').sum()):,}")

    if diagnostics["ambiguous"]:
        st.info(
            f"{diagnostics['ambiguous']:,} addresses had several businesses and none "
            "matched your company name - skipped rather than guessed. Shared "
            "buildings and strip malls do this."
        )

    for message in diagnostics["errors"]:
        st.error(message)

    if diagnostics["searched"] and not diagnostics["found_at_address"]:
        st.warning(
            "No businesses found at any of these addresses. That usually means "
            "the license string is wrong rather than the data being absent - "
            "check the exact license on the Enrichment trial in your dashboard."
        )
    elif diagnostics["found_at_address"] and not diagnostics["matched"]:
        st.warning(
            "Nothing matched. Either the license is wrong or the business dataset "
            "has no record for these addresses. Check one company by hand in the "
            "Smarty dashboard before spending more trial lookups."
        )

    # Show the diff rather than two columns side by side - what changed is
    # the question, not what each source says.
    if diagnostics["matched"]:
        comparison = build_comparison(enriched)
        verdict_columns = [label for label, _, _ in COMPARISON_FIELDS
                           if label in comparison.columns]

        st.write("**What Smarty changed**")

        tallies = []
        for label in verdict_columns:
            counts = comparison[label].value_counts()
            tallies.append({
                "Field": label,
                "Agrees": int(counts.get(DIFF_AGREES, 0)),
                "Changed": int(counts.get(DIFF_CHANGED, 0)),
                "Filled a blank": int(counts.get(DIFF_FILLED, 0)),
                "No data": int(counts.get(DIFF_NONE, 0)),
            })
        st.dataframe(pd.DataFrame(tallies), use_container_width=True, hide_index=True)

        st.caption(
            "Green agrees, yellow is different, blue filled a blank in your file, "
            "grey means Smarty had nothing."
        )

        show = st.radio(
            "Show",
            ["Only rows with a change", "Only rows Smarty filled in", "Everything"],
            horizontal=True,
        )

        changed_mask = comparison[verdict_columns].isin([DIFF_CHANGED]).any(axis=1) \
            if verdict_columns else pd.Series(False, index=comparison.index)
        filled_mask = comparison[verdict_columns].isin([DIFF_FILLED]).any(axis=1) \
            if verdict_columns else pd.Series(False, index=comparison.index)

        if show == "Only rows with a change":
            view = comparison[changed_mask]
        elif show == "Only rows Smarty filled in":
            view = comparison[filled_mask]
        else:
            view = comparison

        if view.empty:
            st.info("Nothing in that category.")
        else:
            st.dataframe(color_comparison(view.head(200)),
                         use_container_width=True, hide_index=True)
            if len(view) > 200:
                st.caption(f"Showing 200 of {len(view):,} rows.")

        st.caption(
            f"{int(changed_mask.sum()):,} companies where Smarty disagrees with "
            f"your file, {int(filled_mask.sum()):,} where it filled a gap. "
            "Spot-check a few before trusting the rest."
        )

        # The rows worth a human look before anyone calls them.
        if changed_mask.any():
            download_row(
                enriched[changed_mask], "Download rows where Smarty disagrees",
                f"changed_{file_key}",
            )

    # Write the resolved columns so every later stage reads Smarty's data
    # where it exists rather than the original upload.
    enriched = apply_resolved_values(enriched)

    # Free commercial read from data we already hold - no RDI subscription.
    enriched = apply_commercial_signal(enriched)

    location_counts = enriched["Location Type"].value_counts()
    loc_a, loc_b, loc_c = st.columns(3)
    loc_a.metric("Business confirmed", f"{location_counts.get(COMMERCIAL_STRONG, 0):,}")
    loc_b.metric("Likely business", f"{location_counts.get(COMMERCIAL_LIKELY, 0):,}")
    loc_c.metric("Likely home-based", f"{location_counts.get(COMMERCIAL_HOME, 0):,}")
    st.caption(
        "Derived from the business record, headcount and address shape - no "
        "RDI subscription needed. The Location Basis column shows the reasoning "
        "per row, and Grok refines it for free in the next stage."
    )

    using_smarty = int((enriched[RESOLVED_SOURCE] == "Smarty").sum())
    if using_smarty:
        st.caption(
            f"{using_smarty:,} companies now carry Smarty's headcount or SIC code "
            "into the later stages. The rest fall back to your file."
        )

    # A company the purchased list undercounted may now clear the threshold -
    # and one it overcounted may now fail it.
    minimum = st.session_state.get("size_minimum", DEFAULT_MIN_EMPLOYEES)
    if minimum and RESOLVED_EMPLOYEES in enriched.columns:
        kept, dropped = recheck_size(enriched, minimum)

        if not dropped.empty:
            st.warning(
                f"{len(dropped):,} companies fall below {minimum} employees on "
                "Smarty's count, though your file put them above it."
            )
            with st.expander(f"See the {len(dropped):,} that no longer qualify"):
                st.dataframe(
                    dropped[["Company Name", SIZE_COLUMN, "Smarty Employee Count"]],
                    use_container_width=True, hide_index=True,
                )

            if st.checkbox(f"Drop these {len(dropped):,} from the working list", value=True):
                enriched = kept

    # Industry ran before Smarty, so it judged on your file's SIC code. Now
    # that we have Smarty's, re-check the survivors - a company your file
    # called a contractor may really be a law office.
    excluded_groups = st.session_state.get("excluded_groups", DEFAULT_EXCLUDED_GROUPS)
    excluded_keywords = st.session_state.get("excluded_keywords", DEFAULT_EXCLUDED_KEYWORDS)

    reclassified = []
    for index, row in enriched.iterrows():
        if is_blank(row.get("Smarty SIC Code")):
            continue
        tier, reason = classify_industry(
            row.get(RESOLVED_SIC), row.get(RESOLVED_SIC_DESC),
            tuple(excluded_groups), tuple(excluded_keywords),
        )
        if tier == TIER_EXCLUDED:
            reclassified.append({
                "Company Name": row.get("Company Name", ""),
                "Your SIC": row.get("Primary SIC Code", ""),
                "Smarty SIC": row.get("Smarty SIC Code", ""),
                "Now reads as": reason,
            })

    if reclassified:
        st.warning(
            f"{len(reclassified):,} companies land in an excluded industry on "
            "Smarty's SIC code, though your file's code let them through."
        )
        with st.expander(f"See the {len(reclassified):,} reclassified companies"):
            st.dataframe(pd.DataFrame(reclassified),
                         use_container_width=True, hide_index=True)

        if st.checkbox(f"Drop these {len(reclassified):,} too", value=True):
            drop_names = {entry["Company Name"] for entry in reclassified}
            enriched = enriched[~enriched["Company Name"].isin(drop_names)].copy()

    download_row(enriched, "Download enriched list", f"enriched_{file_key}")
    return enriched


def stage_smarty_address(working_df: pd.DataFrame, file_key: str) -> pd.DataFrame:
    """Stage 3, US Street mode: verify addresses, keep commercial."""

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
                working_df, SMARTY_AUTH_ID, SMARTY_AUTH_TOKEN, SMARTY_LICENSE, progress
            )
        except Exception as err:
            message = str(err)
            st.error(f"Verification failed: {message}")

            if "subscription" in message.lower() or "1588026162" in message:
                st.caption(
                    "Smarty accepted the credentials but found no active US Street "
                    "subscription on the account. The free trial runs 42 days, so "
                    "it may have expired by date rather than by usage - check the "
                    "Smarty dashboard, not the status page. If the account holds "
                    "several Smarty products, add SMARTY_LICENSE to Secrets "
                    "(comma-separated is fine, e.g. 'us-core-cloud')."
                )
            else:
                st.caption(
                    "401 means the credentials are wrong; 402 means the lookup "
                    "balance is empty."
                )
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


def stage_industry(working_df: pd.DataFrame, file_key: str) -> pd.DataFrame:
    """
    Stage 2: prioritise blue collar work, exclude only what you choose.

    Deliberately permissive - the picker is built from the industries actually
    in this file, so exclusions get added as they turn up rather than guessed
    at up front.
    """
    st.subheader("2. Industry fit")

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

    st.session_state["excluded_groups"] = tuple(excluded_groups)
    st.session_state["excluded_keywords"] = tuple(excluded_keywords)

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

    priority_only = st.checkbox(
        f"Priority industries only - drop the {counts.get(TIER_NEUTRAL, 0):,} neutral ones",
        value=False,
        help="Neutral is retail, hospitality, education, services - real payroll "
             "but higher turnover. Tick this to work only construction, "
             "manufacturing, trucking, warehousing and trades.",
    )

    allowed = [TIER_PRIORITY] if priority_only else [TIER_PRIORITY, TIER_NEUTRAL]
    result = tiered[tiered["Industry Tier"].isin(allowed)].copy()
    dropped = tiered[~tiered["Industry Tier"].isin(allowed)]

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

    if not dropped.empty:
        st.caption(
            "This runs on your file's SIC codes, before Smarty has seen them. "
            "A miscoded company is excluded here and won't be recovered later - "
            "so if the excluded list looks wrong, widen it rather than trusting "
            "the code."
        )
    download_row(result, "Download industry-sorted list", f"industry_{file_key}")

    return result


def stage_contacts(working_df: pd.DataFrame, file_key: str):
    """Stage 3: verify, update, or set aside the decision maker."""
    st.subheader("4. Decision maker (optional, in-app)")

    st.caption(
        "Optional. Stage 5 hands this work to ChatGPT on a subscription you "
        "already pay for, which is cheaper and researches more deeply. Use this "
        "only as a lightweight supplement, or when you want the list pre-filled "
        "before the handoff."
    )

    if not st.toggle("Run in-app contact lookups (costs per search)", value=False):
        return working_df, None

    if not XAI_API_KEY:
        st.info("No xAI key. Add XAI_API_KEY to Secrets to enable contact lookups.")
        return working_df, None

    st.caption(
        "Searches company websites and state business filings for whoever can "
        "approve a benefits decision. Confirms the name on file, replaces it "
        "when it's stale, and fills one in when the record has none."
    )

    model_label = st.selectbox(
        "Model", list(CONTACT_MODELS.keys()),
        index=list(CONTACT_MODELS.keys()).index(DEFAULT_CONTACT_MODEL),
        help="Contact lookup is retrieval, not reasoning, so the cheap models "
             "usually do it about as well.",
    )
    model = CONTACT_MODELS[model_label]
    st.caption(model["note"])

    rates = {
        "input_per_million": model["input_per_million"],
        "cached_input_per_million": model["cached_input_per_million"],
        "output_per_million": model["output_per_million"],
        "per_thousand_searches": DEFAULT_SEARCH_RATE,
    }

    with st.expander("API rates (edit if xAI's pricing changes)"):
        r1, r2, r3, r4 = st.columns(4)
        rates["input_per_million"] = r1.number_input(
            "$ / 1M input", value=rates["input_per_million"], step=0.25, format="%.2f")
        rates["cached_input_per_million"] = r2.number_input(
            "$ / 1M cached", value=rates["cached_input_per_million"], step=0.10, format="%.2f")
        rates["output_per_million"] = r3.number_input(
            "$ / 1M output", value=rates["output_per_million"], step=0.25, format="%.2f")
        rates["per_thousand_searches"] = r4.number_input(
            "$ / 1k searches", value=rates["per_thousand_searches"], step=0.50, format="%.2f")
        st.caption(
            f"Requests over {LONG_CONTEXT_THRESHOLD:,} tokens bill at "
            f"{LONG_CONTEXT_MULTIPLIER:g}x on the whole request, which is why "
            f"only {CONTACT_BATCH_SIZE} companies go per call."
        )

    has_smarty = "Smarty Contact Last Name" in working_df.columns
    smarty_contacts = 0
    if has_smarty:
        smarty_contacts = int(
            working_df["Smarty Contact Last Name"].fillna("").astype(str)
            .str.strip().replace({"None": "", "nan": ""}).ne("").sum()
        )

    use_smarty = True
    if smarty_contacts:
        use_smarty = st.checkbox(
            f"Use Smarty's contact where it has one ({smarty_contacts:,} companies)",
            value=True,
            help="Smarty's business record is a maintained database, so it's taken "
                 "as answered and skipped here. Uncheck to search the web for "
                 "these too, which costs more and is unlikely to be better.",
        )

    benefits_threshold = st.number_input(
        "Search for a separate benefits contact at companies with at least",
        min_value=0, value=BENEFITS_TARGET_MIN_EMPLOYEES, step=5,
        help="Below this headcount the owner usually handles benefits, so a "
             "separate search would spend money finding the same person. "
             "Companies with unknown headcount are always searched.",
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

    # Only companies Smarty didn't answer need a search.
    to_search = limit
    if use_smarty and smarty_contacts:
        to_search = max(limit - min(smarty_contacts, limit), 0)

    calls = -(-to_search // CONTACT_BATCH_SIZE)

    # Rough forecast: ~700 input and ~250 output tokens per call, plus about
    # two searches per company. Deliberately not optimistic.
    # Each search injects its results into the request, so input tokens scale
    # with searches, not just with the prompt. Assume ~8k injected per search.
    searches = to_search * CONTACT_MAX_SEARCHES
    estimate = (
        (calls * 600 + searches * 8_000) / 1_000_000 * rates["input_per_million"]
        + calls * 400 / 1_000_000 * rates["output_per_million"]
        + searches / 1_000 * rates["per_thousand_searches"]
    )
    search_floor = searches / 1_000 * rates["per_thousand_searches"]

    if to_search < limit:
        st.caption(
            f"{limit - to_search:,} of {limit} already have a Smarty contact, so "
            f"only {to_search:,} need searching - about {calls} call(s). "
            f"Rough estimate ${estimate:.2f}, of which ${search_floor:.2f} is "
            "web search fees."
        )
    else:
        st.caption(
            f"About {calls} call(s) for {to_search} companies ({CONTACT_BATCH_SIZE} "
            f"per call). Rough estimate ${estimate:.2f}, roughly "
            f"${estimate / max(to_search, 1):.3f} per company - of which "
            f"${search_floor:.2f} is web search fees, which no model change reduces. "
            "Companies already checked are free."
        )

    cache_key = f"contacts::{file_key}::{len(working_df)}::{limit}"

    if st.button(f"Resolve {limit} companies", type="primary"):
        progress = st.progress(0.0)
        try:
            st.session_state[cache_key] = resolve_contacts(
                working_df, XAI_API_KEY, limit, rates, model["id"], use_smarty,
                benefits_threshold, progress
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
            {"Item": "Output tokens (incl. reasoning)",
             "Count": f"{diagnostics['output_tokens']:,}",
             "Cost": f"${diagnostics['output_tokens'] / 1_000_000 * rates['output_per_million']:.4f}"},
            {"Item": "  of which reasoning",
             "Count": f"{diagnostics['reasoning_tokens']:,}", "Cost": "included above"},
            {"Item": "Web searches", "Count": f"{diagnostics['searches']:,}",
             "Cost": f"${search_cost:.4f}"},
        ]), use_container_width=True, hide_index=True)

        st.caption(
            f"Tokens ${token_cost:.4f}, searches ${search_cost:.4f}. "
            f"{diagnostics['calls']} call(s), {diagnostics['checked']} companies checked, "
            f"{diagnostics['cached']} served from cache."
        )

        if diagnostics["long_context_calls"]:
            st.error(
                f"{diagnostics['long_context_calls']} call(s) crossed "
                f"{LONG_CONTEXT_THRESHOLD:,} tokens and billed at "
                f"{LONG_CONTEXT_MULTIPLIER:g}x. Largest was "
                f"{diagnostics['peak_tokens']:,} tokens. Lower CONTACT_BATCH_SIZE "
                "or CONTACT_MAX_SEARCHES to avoid this."
            )
        elif diagnostics["peak_tokens"]:
            st.caption(
                f"Largest request {diagnostics['peak_tokens']:,} tokens, under "
                f"the {LONG_CONTEXT_THRESHOLD:,} double-rate threshold."
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
                    "Executive Authority", "Executive Title", "Benefits Target",
                    "Benefits Target Title", "Contact Source",
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

    st.caption(f"{len(actionable):,} leads on the working list, {len(set_aside):,} set aside.")

    if not set_aside.empty:
        download_row(set_aside, "Download set-aside leads", f"set_aside_{file_key}")

    return actionable, set_aside


def stage_handoff(working_df: pd.DataFrame, file_key: str) -> pd.DataFrame:
    """
    Stage 5: hand the qualified list to ChatGPT, then merge the research back.

    Deep research runs on a ChatGPT subscription the team already pays for,
    rather than per-search through the API.
    """
    st.subheader("5. ChatGPT research handoff")

    st.markdown(
        "**Structured qualification is complete.** These records have been "
        "deduplicated, enriched and filtered on the best structured data "
        "available.\n\n"
        "The remaining work needs judgement rather than lookups: verifying the "
        "current owner, independently finding the benefits contact, telling the "
        "physical site from the mailing address, spotting branches and "
        "franchises, resolving conflicts, and gathering rapport notes."
    )

    handoff = build_handoff(working_df)

    left, right = st.columns([1, 1])

    with left:
        st.download_button(
            f"Download {HANDOFF_FILENAME} ({len(handoff):,} companies)",
            data=handoff_to_excel(handoff),
            file_name=HANDOFF_FILENAME,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )
        st.caption(
            "Upload this to ChatGPT with the prompt beside it. Work in batches "
            "of about 25 - a long list tends to come back truncated or with "
            "rows quietly dropped."
        )

    with right:
        st.caption("Copy this prompt into ChatGPT with the file attached.")

    st.code(CHATGPT_PROMPT, language=None)

    st.divider()
    st.write("**When the research comes back**")

    returned_file = st.file_uploader(
        "Upload the researched XLSX", type=["xlsx"], key=f"research_{file_key}"
    )

    if returned_file is None:
        st.caption(
            "The merge is keyed on Source Row ID, so the file must come back "
            "with that column unchanged."
        )
        return working_df

    try:
        returned = pd.read_excel(returned_file, dtype=str)
    except Exception as err:
        st.error(f"Couldn't read that file: {err}")
        return working_df

    merged, problems = merge_research(working_df, returned)

    for message in problems:
        st.warning(message)

    filled = 0
    for name in RESEARCH_COLUMNS:
        if name in merged.columns:
            filled = max(filled, int(merged[name].fillna("").astype(str).str.strip().ne("").sum()))

    if not filled:
        st.error("Nothing merged - no research values came back populated.")
        return working_df

    st.success(f"Research merged for {filled:,} companies.")

    if "Benefits Target Name" in merged.columns:
        named = merged[merged["Benefits Target Name"].fillna("").astype(str)
                       .str.strip().ne("")]
        named = named[~named["Benefits Target Name"].astype(str)
                      .str.strip().str.lower().eq("could not find")]
        if not named.empty:
            st.caption(f"{len(named):,} companies came back with a named benefits contact.")

    if "Corporate Structure" in merged.columns:
        structures = merged["Corporate Structure"].fillna("").astype(str).str.strip()
        controlled = structures.str.lower().isin(
            ["franchise", "branch", "subsidiary", "division"]
        ).sum()
        if controlled:
            st.warning(
                f"{int(controlled):,} are branches, franchises or subsidiaries - "
                "benefits decisions may sit with the parent, not the local site."
            )

    download_row(merged, "Download the researched list", f"researched_{file_key}",
                 primary=True)

    return merged


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.title(APP_TITLE)
    st.write(
        "Deduplicate, filter by employee count and industry, enrich with Smarty, "
        "optionally check contacts in-app, then hand the qualified list to "
        "ChatGPT for the deep research and merge it back. The free stages run "
        "first so nothing metered is spent on a lead you were going to discard."
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

    # Each stage's surviving count, so the funnel is visible at the end.
    funnel = [("Uploaded", len(raw_df))]

    # 0 - duplicates. Free, and every duplicate removed here is a paid
    # lookup and a web search not spent twice.
    column_df = stage_dedupe(column_df, uploaded_file.name)
    funnel.append(("After deduplication", len(column_df)))

    st.divider()

    # 1 - employees
    working_df = stage_employees(column_df)
    funnel.append(("After size filter", len(working_df)))
    if working_df.empty:
        return

    # 2 - industry. Free, so it runs before anything metered and keeps us
    # from spending lookups on companies we're about to discard.
    st.divider()
    working_df = stage_industry(working_df, uploaded_file.name)
    funnel.append(("After industry filter", len(working_df)))

    if working_df.empty:
        st.info("Nothing left after industry filtering.")
        return

    # 3 - Smarty. Metered, so it only sees what survived the free stages.
    st.divider()
    st.subheader("3. Smarty business data")

    if not SMARTY_AVAILABLE:
        st.info("Smarty SDK not installed. Add smartystreets_python_sdk to requirements.txt.")
    elif not SMARTY_AUTH_ID or not SMARTY_AUTH_TOKEN:
        st.info("No Smarty credentials. Add SMARTY_AUTH_ID and SMARTY_AUTH_TOKEN to Secrets.")
    else:
        smarty_mode = st.radio(
            "Which Smarty subscription is active?",
            [SMARTY_MODE_STREET, SMARTY_MODE_BUSINESS],
            index=1,
            help="US Street verifies deliverability and flags residential. "
                 "US Enrichment returns employee counts, SIC codes and executive "
                 "contacts, but nothing about mail deliverability.",
            horizontal=True,
        )

        if smarty_mode == SMARTY_MODE_STREET:
            working_df = stage_smarty_address(working_df, uploaded_file.name)
        else:
            working_df = stage_smarty_business(working_df, uploaded_file.name)

    funnel.append(("After Smarty", len(working_df)))

    st.divider()
    working_df, set_aside = stage_contacts(working_df, uploaded_file.name)
    funnel.append(("With a usable contact", len(working_df)))

    # 5 - hand the survivors to ChatGPT for the judgement work.
    st.divider()
    working_df = stage_handoff(working_df, uploaded_file.name)


    # --- Final list ---
    st.divider()
    st.subheader("Working list")

    # How many survived each stage, and what each one cost you.
    with st.expander("How the list narrowed"):
        rows = []
        previous = None
        for label, count in funnel:
            rows.append({
                "Stage": label,
                "Leads": f"{count:,}",
                "Removed here": "-" if previous is None else f"{previous - count:,}",
            })
            previous = count
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if RESOLVED_SOURCE in working_df.columns:
            st.caption(
                "Stages after the Smarty step read its headcount and SIC code "
                "where it has them, falling back to your file otherwise. The "
                "'Data source' column shows which was used per row."
            )

    # Every dial that could narrow this further, and what it would cost you
    # right now - so tightening is a decision, not a guess.
    with st.expander("Want a shorter list? What each filter would remove"):
        dials = []

        if SIZE_COLUMN in working_df.columns or RESOLVED_EMPLOYEES in working_df.columns:
            column = RESOLVED_EMPLOYEES if RESOLVED_EMPLOYEES in working_df.columns \
                else SIZE_COLUMN
            bounds = working_df[column].map(lambda v: parse_employee_range(v)[0])
            for threshold in (10, 20, 50):
                below = int((bounds.notna() & (bounds < threshold)).sum())
                if below:
                    dials.append({
                        "Filter": f"Raise minimum employees to {threshold}",
                        "Where": "Stage 1",
                        "Would remove": f"{below:,}",
                        "Leaves": f"{len(working_df) - below:,}",
                    })

        if "Industry Tier" in working_df.columns:
            neutral = int((working_df["Industry Tier"] == TIER_NEUTRAL).sum())
            if neutral:
                dials.append({
                    "Filter": "Priority industries only",
                    "Where": "Stage 2",
                    "Would remove": f"{neutral:,}",
                    "Leaves": f"{len(working_df) - neutral:,}",
                })

        if "Location Type" in working_df.columns:
            home = int((working_df["Location Type"] == COMMERCIAL_HOME).sum())
            if home:
                dials.append({
                    "Filter": "Hide likely home-based",
                    "Where": "Working list below",
                    "Would remove": f"{home:,}",
                    "Leaves": f"{len(working_df) - home:,}",
                })

        if "Contact Status" in working_df.columns:
            unverified = int((working_df["Contact Status"] == STATUS_UNVERIFIED).sum())
            if unverified:
                dials.append({
                    "Filter": "Drop unverified contacts",
                    "Where": "Stage 4",
                    "Would remove": f"{unverified:,}",
                    "Leaves": f"{len(working_df) - unverified:,}",
                })

        if dials:
            st.dataframe(pd.DataFrame(dials), use_container_width=True, hide_index=True)
            st.caption(
                "These overlap - removing the same company twice still only "
                "removes it once, so the combined effect is smaller than the sum."
            )
        else:
            st.caption("Nothing obvious left to trim at the current settings.")

    if set_aside is not None and not set_aside.empty:
        st.caption(
            f"{len(working_df):,} leads carried forward, "
            f"{len(set_aside):,} set aside."
        )

    if "Location Type" in working_df.columns:
        if st.checkbox("Hide likely home-based locations", value=False,
                       help="Home-run businesses can still be good worksite "
                            "prospects, so this is off by default."):
            working_df = working_df[working_df["Location Type"] != COMMERCIAL_HOME].copy()
            st.caption(f"{len(working_df):,} leads shown.")

    st.dataframe(color_status(working_df.head(MAX_PREVIEW_ROWS)), use_container_width=True)

    if len(working_df) > MAX_PREVIEW_ROWS:
        st.caption(f"Showing {MAX_PREVIEW_ROWS:,} rows. The download has all of them.")

    download_row(working_df, "Download working list", f"leads_{uploaded_file.name}", primary=True)


if __name__ == "__main__":
    main()
